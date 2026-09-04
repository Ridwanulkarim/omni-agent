#!/usr/bin/env python3
"""Hugging Face Spaces Web App for OmniAgent using Gradio."""

import os
import uuid
import warnings
from pathlib import Path
from typing import Generator, Tuple, List

import gradio as gr
from dotenv import load_dotenv

# Suppress harmless warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from langchain_core.messages import HumanMessage
from omni_agent.config import AgentConfig
from omni_agent.graph import create_omni_agent
from omni_agent.tools import get_default_tools

load_dotenv()

WORKSPACE_PATH = Path("./workspace").resolve()
WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)


def list_workspace_files() -> List[str]:
    """Return a list of relative file paths currently in the workspace."""
    if not WORKSPACE_PATH.exists():
        return []
    files = []
    for p in sorted(WORKSPACE_PATH.rglob("*")):
        if p.is_file():
            files.append(str(p.relative_to(WORKSPACE_PATH)))
    return files


def read_workspace_file(selected_file: str) -> str:
    """Read contents of a selected workspace file."""
    if not selected_file:
        return "Select a file to inspect its content."
    target = (WORKSPACE_PATH / selected_file).resolve()
    try:
        target.relative_to(WORKSPACE_PATH)
        if target.exists() and target.is_file():
            return target.read_text(encoding="utf-8", errors="replace")
        return f"File '{selected_file}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def run_agent_stream(
    goal: str,
    provider: str,
    model: str,
    custom_api_key: str,
    max_steps: int,
) -> Generator[Tuple[str, str, str, gr.Dropdown], None, None]:
    """Execute an agent goal and stream real-time updates to Gradio components."""
    if not goal or not goal.strip():
        yield "", "⚠️ Please enter a goal or task description.", "", gr.Dropdown(choices=list_workspace_files())
        return

    # Set up configuration
    config = AgentConfig()
    config.workspace_dir = WORKSPACE_PATH
    config.max_steps = int(max_steps)

    provider_clean = "groq" if "groq" in provider.lower() else "gemini"
    config.provider = provider_clean
    config.model = model.strip()

    # If user provided a custom key, inject it into environment
    if custom_api_key and custom_api_key.strip():
        key_val = custom_api_key.strip()
        if provider_clean == "groq":
            os.environ["GROQ_API_KEY"] = key_val
        elif provider_clean == "gemini":
            os.environ["GEMINI_API_KEY"] = key_val
            config.gemini_api_key = key_val

    plan_markdown = "⏳ *Master Planner is analyzing the goal and breaking it into steps...*"
    execution_trace = "🚀 *Agent execution initialized...*\n"
    final_result = ""

    yield plan_markdown, execution_trace, final_result, gr.Dropdown(choices=list_workspace_files())

    try:
        app = create_omni_agent(config)
    except Exception as e:
        err_msg = f"❌ **Initialization Error:** {str(e)}"
        yield plan_markdown, err_msg, err_msg, gr.Dropdown(choices=list_workspace_files())
        return

    initial_state = {
        "user_goal": goal.strip(),
        "messages": [HumanMessage(content=goal.strip())],
        "iteration": 0,
        "max_iterations": config.max_steps,
        "is_completed": False,
    }

    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    try:
        for event in app.stream(initial_state, thread_config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "planner":
                    plan = node_output.get("plan")
                    if plan:
                        rows = ["| Step | Status | Description |", "| :---: | :---: | :--- |"]
                        for s in plan.steps:
                            status_icon = "🟢 Done" if s.status == "completed" else ("🟡 Working" if s.status == "in_progress" else "⚪ Pending")
                            rows.append(f"| **{s.step_id}** | {status_icon} | {s.description} |")
                        plan_markdown = f"### 📋 Execution Plan\n**Goal:** {plan.goal}\n\n" + "\n".join(rows)

                elif node_name == "executor":
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                args_repr = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                if len(args_repr) > 120:
                                    args_repr = args_repr[:120] + "..."
                                execution_trace += f"\n- ⚙️ **Calling Tool:** `{tc['name']}({args_repr})`"
                        elif last_msg.content:
                            thought = str(last_msg.content).strip()
                            if len(thought) > 200:
                                thought = thought[:200] + "..."
                            execution_trace += f"\n- 💭 *Agent:* {thought}"

                elif node_name == "tools":
                    messages = node_output.get("messages", [])
                    for m in messages:
                        output_snippet = str(m.content).strip()
                        if len(output_snippet) > 250:
                            output_snippet = output_snippet[:250] + "..."
                        execution_trace += f"\n  - ✓ *Tool Output:* `{output_snippet}`"

                elif node_name == "verifier":
                    feedback = node_output.get("verification_feedback")
                    if feedback:
                        execution_trace += f"\n- 🔍 **Verifier Feedback:** {feedback[:180]}"
                    elif node_output.get("is_completed"):
                        execution_trace += "\n- 🎯 **Verifier:** Goal satisfied!"

                elif node_name == "synthesizer":
                    final_result = node_output.get("final_response", "")

                yield plan_markdown, execution_trace, final_result, gr.Dropdown(choices=list_workspace_files())

        if not final_result:
            final_result = "Task execution reached completion limit. Check the activity log and workspace files."
            yield plan_markdown, execution_trace, final_result, gr.Dropdown(choices=list_workspace_files())

    except Exception as e:
        execution_trace += f"\n\n❌ **Execution error:** {str(e)}"
        yield plan_markdown, execution_trace, f"Error: {str(e)}", gr.Dropdown(choices=list_workspace_files())


# Build Gradio Interface
custom_css = """
#main-container { max-width: 1100px; margin: 0 auto; }
.output-box { border-radius: 8px; }
"""

with gr.Blocks(title="OmniAgent — Autonomous AI Agent", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 OmniAgent
        ### *Autonomous General-Purpose Problem Solving AI Agent*
        Built with **LangGraph**, self-correcting verification loops, and dynamic tools:
        `filesystem`, `web_search`, `python_sandbox`, and `shell_execution`.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            goal_input = gr.Textbox(
                label="Enter your Goal / Instruction",
                placeholder="e.g. Build a sleek interactive stopwatch web app in workspace with lap times and reset features",
                lines=3,
            )

            with gr.Row():
                run_btn = gr.Button("🚀 Execute Goal", variant="primary", scale=2)
                clear_btn = gr.Button("🧹 Clear", scale=1)

            gr.Examples(
                examples=[
                    ["Build a modern stopwatch web app in the workspace with start, pause, lap times, and sleek styling."],
                    ["Calculate the first 25 Fibonacci numbers and save them in a formatted table to fibonacci.txt."],
                    ["Write a Python script to compute prime numbers up to 100, verify it by executing it, and save to primes.txt."],
                    ["Search the web for the latest updates on Python 3.13 and summarize the top new features."],
                ],
                inputs=goal_input,
                label="Quick Examples",
            )

        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Model & API Settings", open=True):
                provider_dropdown = gr.Dropdown(
                    label="LLM Provider",
                    choices=["Groq (Ultra-Fast Free Tier)", "Google Gemini (Free Tier)"],
                    value="Groq (Ultra-Fast Free Tier)",
                )
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "gemini-3.6-flash"],
                    value="llama-3.3-70b-versatile",
                )
                api_key_input = gr.Textbox(
                    label="API Key (Optional if Space Secret is set)",
                    placeholder="gsk_... or AIzaSy...",
                    type="password",
                )
                steps_slider = gr.Slider(
                    label="Max Execution Steps",
                    minimum=5,
                    maximum=30,
                    value=15,
                    step=1,
                )

    with gr.Tabs():
        with gr.TabItem("✨ Final Response"):
            result_output = gr.Markdown(label="Final Response", value="*Results will appear here after the agent completes the task.*")

        with gr.TabItem("📋 Plan & Live Execution Log"):
            plan_output = gr.Markdown(label="Plan", value="*Execution plan will appear here...*")
            trace_output = gr.Markdown(label="Real-time Trace", value="*Agent activity will appear here...*")

        with gr.TabItem("📂 Workspace File Explorer"):
            gr.Markdown("Inspect and read files created by OmniAgent inside the `./workspace` sandbox:")
            with gr.Row():
                file_picker = gr.Dropdown(label="Select Created File", choices=list_workspace_files())
                refresh_files_btn = gr.Button("🔄 Refresh File List", scale=0)
            file_content_viewer = gr.Code(label="File Content", language="python")

            refresh_files_btn.click(
                fn=lambda: gr.Dropdown(choices=list_workspace_files()),
                outputs=file_picker,
            )
            file_picker.change(
                fn=read_workspace_file,
                inputs=file_picker,
                outputs=file_content_viewer,
            )

    # Wire event handlers
    run_btn.click(
        fn=run_agent_stream,
        inputs=[goal_input, provider_dropdown, model_dropdown, api_key_input, steps_slider],
        outputs=[plan_output, trace_output, result_output, file_picker],
    )

    clear_btn.click(
        fn=lambda: ("", "*Execution plan will appear here...*", "*Agent activity will appear here...*", "*Results will appear here...*"),
        outputs=[goal_input, plan_output, trace_output, result_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
