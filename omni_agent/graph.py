"""LangGraph execution engine for OmniAgent."""

import warnings
warnings.filterwarnings("ignore")

import json
import re
from typing import Dict, Any, List, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from omni_agent.config import AgentConfig, default_config
from omni_agent.models import get_chat_model
from omni_agent.state import AgentState, ExecutionPlan, PlanStep
from omni_agent.tools import get_default_tools
from omni_agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)

def clean_thinking(text: str) -> str:
    """Strip reasoning/thought blocks like <think>...</think> produced by reasoning models."""
    if not text or not isinstance(text, str):
        return text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def prune_messages_for_context(
    messages: List[BaseMessage],
    max_recent_turns: int = 3,
    max_tool_chars: int = 400,
) -> List[BaseMessage]:
    """Prune conversation messages to fit comfortably within LLM context windows and rate limits (e.g. Groq 8k TPM).
    
    Keeps:
    1. The initial user message (messages[0]).
    2. The last `max_recent_turns` complete interaction blocks (AIMessage + ToolMessages or standalone messages).
    3. Truncates long ToolMessage contents from older turns to prevent token bloat.
    """
    if not messages:
        return []
    if len(messages) <= 3:
        cleaned = []
        for m in messages:
            if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > 1000:
                cleaned.append(
                    ToolMessage(
                        content=m.content[:1000] + f"\n... [Output truncated ({len(m.content)} chars total)]",
                        tool_call_id=m.tool_call_id,
                        name=getattr(m, "name", None),
                    )
                )
            else:
                cleaned.append(m)
        return cleaned

    initial_msg = messages[0]
    rest = messages[1:]

    # Group 'rest' into turn clusters:
    clusters: List[List[BaseMessage]] = []
    current_cluster: List[BaseMessage] = []

    for msg in rest:
        if isinstance(msg, AIMessage):
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [msg]
        elif isinstance(msg, ToolMessage):
            current_cluster.append(msg)
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [msg]
    if current_cluster:
        clusters.append(current_cluster)

    # Keep the last max_recent_turns clusters
    selected_clusters = clusters[-max_recent_turns:] if len(clusters) > max_recent_turns else clusters

    pruned = [initial_msg]
    for i, cluster in enumerate(selected_clusters):
        is_latest_cluster = (i == len(selected_clusters) - 1)
        for m in cluster:
            threshold = 1200 if is_latest_cluster else max_tool_chars
            if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > threshold:
                pruned.append(
                    ToolMessage(
                        content=m.content[:threshold] + f"\n... [Output truncated ({len(m.content)} chars total)]",
                        tool_call_id=m.tool_call_id,
                        name=getattr(m, "name", None),
                    )
                )
            else:
                pruned.append(m)

    return pruned


def create_omni_agent(config: AgentConfig = default_config):
    """Build and compile the OmniAgent state graph."""
    
    config.ensure_workspace()
    llm = get_chat_model(config)
    tools = get_default_tools(config)
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    # -------------------------------------------------------------
    # 1. Planner Node
    # -------------------------------------------------------------
    def planner_node(state: AgentState) -> Dict[str, Any]:
        user_goal = state["user_goal"]
        
        prompt = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User Goal: {user_goal}\n\n"
                    "Output your plan strictly in valid JSON format:\n"
                    "{\n"
                    '  "goal": "...",\n'
                    '  "steps": [\n'
                    '    {"step_id": 1, "description": "...", "tool_hint": "..."}\n'
                    "  ]\n"
                    "}\n"
                    "Do not include markdown backticks around the JSON."
                )
            ),
        ]
        
        try:
            resp = llm.invoke(prompt)
            raw_content = resp.content if isinstance(resp.content, str) else str(resp.content)
            content = clean_thinking(raw_content)
            # Clean markdown code blocks if present
            content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                plan = ExecutionPlan(**data)
            else:
                plan = ExecutionPlan(
                    goal=user_goal,
                    steps=[PlanStep(step_id=1, description=f"Execute: {user_goal}")]
                )
        except Exception:
            plan = ExecutionPlan(
                goal=user_goal,
                steps=[PlanStep(step_id=1, description=f"Execute: {user_goal}")]
            )

        if not plan.steps:
            plan.steps = [PlanStep(step_id=1, description=f"Fulfill request: {user_goal}")]

        plan.steps[0].status = "in_progress"

        return {
            "plan": plan,
            "current_step_index": 0,
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", config.max_steps),
            "is_completed": False,
        }

    # -------------------------------------------------------------
    # 2. Executor Node
    # -------------------------------------------------------------
    def executor_node(state: AgentState) -> Dict[str, Any]:
        plan = state["plan"]
        step_idx = state.get("current_step_index", 0)
        current_step = plan.steps[step_idx] if (plan and step_idx < len(plan.steps)) else None
        
        plan_context = ""
        if plan:
            steps_desc = "\n".join(
                [f"[{s.status.upper()}] Step {s.step_id}: {s.description}" for s in plan.steps]
            )
            plan_context = f"\nOverall Plan:\n{steps_desc}\n\nCurrently Working On Step {step_idx + 1}: {current_step.description if current_step else 'Finalizing'}"

        feedback_context = ""
        if state.get("verification_feedback"):
            feedback_context = f"\nVerifier Feedback on Previous Turn: {state['verification_feedback']}"

        system_instruction = SystemMessage(
            content=f"{EXECUTOR_SYSTEM_PROMPT}\nUser Goal: {state['user_goal']}\n{plan_context}{feedback_context}"
        )

        pruned_history = prune_messages_for_context(list(state["messages"]))
        messages_to_send = [system_instruction] + pruned_history
        
        # Ensure the conversation doesn't end with an AIMessage (Gemini API requirement)
        if isinstance(messages_to_send[-1], AIMessage):
            messages_to_send.append(
                HumanMessage(content=f"Continue executing step {step_idx + 1}: {current_step.description if current_step else 'conclude'}")
            )

        import time
        max_attempts = 3
        response = None
        for attempt in range(max_attempts):
            try:
                response = llm_with_tools.invoke(messages_to_send)
                break
            except Exception as e:
                error_str = str(e)
                # Handle 413 context too large / TPM limit
                if "413" in error_str or "Request too large" in error_str:
                    emergency_prompt = [
                        system_instruction,
                        state["messages"][0],
                        HumanMessage(content=f"Execute step {step_idx + 1}: {current_step.description if current_step else 'conclude'}")
                    ]
                    try:
                        response = llm_with_tools.invoke(emergency_prompt)
                        break
                    except Exception as e_em:
                        error_str = str(e_em)

                if "Rate limit" in error_str or "429" in error_str:
                    delay = 8.0
                    match = re.search(r"try again in (\d+(\.\d+)?)s", error_str)
                    if match:
                        delay = float(match.group(1)) + 1.0
                    time.sleep(delay)
                    if attempt == max_attempts - 1:
                        response = AIMessage(content=f"Rate limit reached after retries: {error_str[:150]}")
                elif "Failed to parse tool call arguments" in error_str or "tool_use_failed" in error_str:
                    retry_msg = HumanMessage(
                        content=(
                            f"Notice: The tool call had a syntax/formatting error: {error_str[:120]}. "
                            "Please call the tool again with clean, properly escaped JSON strings and no trailing brackets."
                        )
                    )
                    try:
                        response = llm_with_tools.invoke(messages_to_send + [retry_msg])
                        break
                    except Exception as e2:
                        response = AIMessage(content=f"Error running tool step: {str(e2)[:200]}")
                        break
                else:
                    if attempt == max_attempts - 1:
                        response = AIMessage(content=f"Execution error: {error_str[:200]}")
                        break
                    time.sleep(2.0)

        return {
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1,
        }

    # -------------------------------------------------------------
    # 3. Router
    # -------------------------------------------------------------
    def should_continue(state: AgentState) -> Literal["tools", "verifier", "synthesizer"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        # If tool calls were requested, execute tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Check safety guardrails
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", config.max_steps)
        if iteration >= max_iter:
            return "synthesizer"

        return "verifier"

    # -------------------------------------------------------------
    # 4. Verifier Node
    # -------------------------------------------------------------
    def verifier_node(state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan")
        step_idx = state.get("current_step_index", 0)
        
        if not plan or step_idx >= len(plan.steps):
            return {"is_completed": True}

        current_step = plan.steps[step_idx]
        recent_activity = []
        for m in state["messages"][-6:]:
            prefix = "Tool Output" if isinstance(m, ToolMessage) else type(m).__name__.replace("Message", "")
            content_str = m.content if isinstance(m.content, str) else str(m.content)
            recent_activity.append(f"[{prefix}]: {content_str[:250]}")
        activity_str = "\n".join(recent_activity)

        verification_prompt = [
            SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Original User Goal: {state['user_goal']}\n"
                        f"Current Step ({step_idx + 1}/{len(plan.steps)}): {current_step.description}\n"
                        f"Recent Tool & Agent Execution Activity:\n{activity_str}\n\n"
                        f"Respond with 'COMPLETED: <brief summary>' if this step or the overall task is satisfied, "
                        f"or 'INCOMPLETE: <guidance>' if more work is needed."
            )
        ]
        raw_review = llm.invoke(verification_prompt).content
        review = clean_thinking(raw_review if isinstance(raw_review, str) else str(raw_review)).strip()
        review_lower = review.lower()
        is_step_done = (
            "completed" in review_lower
            or "is complete" in review_lower
            or "is verified" in review_lower
            or "satisfied" in review_lower
            or "success" in review_lower
        )

        has_tool_output = any(isinstance(m, ToolMessage) for m in state["messages"])
        if is_step_done or has_tool_output or state.get("iteration", 0) >= 3:
            current_step.status = "completed"
            next_idx = step_idx + 1
            if next_idx < len(plan.steps):
                plan.steps[next_idx].status = "in_progress"
                return {
                    "plan": plan,
                    "current_step_index": next_idx,
                    "is_completed": False,
                    "verification_feedback": None,
                }
            else:
                return {
                    "plan": plan,
                    "is_completed": True,
                    "verification_feedback": None,
                }
        else:
            return {
                "plan": plan,
                "is_completed": False,
                "verification_feedback": review,
            }

    def verifier_router(state: AgentState) -> Literal["executor", "synthesizer"]:
        if state.get("is_completed", False):
            return "synthesizer"
        iteration = state.get("iteration", 0)
        max_iter = min(state.get("max_iterations", config.max_steps), 4)
        if iteration >= max_iter:
            return "synthesizer"
        return "executor"

    # -------------------------------------------------------------
    # 5. Synthesizer Node
    # -------------------------------------------------------------
    def synthesizer_node(state: AgentState) -> Dict[str, Any]:
        user_goal = state["user_goal"]
        messages = state["messages"]

        transcript_snippets = []
        for m in messages[-8:]:
            c = m.content if isinstance(m.content, str) else str(m.content)
            transcript_snippets.append(f"{type(m).__name__}: {c[:300]}")

        synthesis_prompt = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"User Goal: {user_goal}\n\nExecution Transcript:\n"
                        + "\n".join(transcript_snippets)
                        + "\n\nPlease deliver the final synthesized response."
            )
        ]
        raw_final = llm.invoke(synthesis_prompt).content
        final_answer = clean_thinking(raw_final if isinstance(raw_final, str) else str(raw_final)).strip()
        return {
            "final_response": final_answer,
            "messages": [AIMessage(content=final_answer)],
            "is_completed": True,
        }

    # -------------------------------------------------------------
    # Build Graph
    # -------------------------------------------------------------
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "tools": "tools",
            "verifier": "verifier",
            "synthesizer": "synthesizer",
        }
    )
    
    workflow.add_edge("tools", "executor")
    
    workflow.add_conditional_edges(
        "verifier",
        verifier_router,
        {
            "executor": "executor",
            "synthesizer": "synthesizer",
        }
    )
    
    workflow.add_edge("synthesizer", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
