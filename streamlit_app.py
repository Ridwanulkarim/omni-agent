#!/usr/bin/env python3
"""Streamlit Community Cloud Web Application for OmniAgent."""

import os
import uuid
import warnings
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Suppress harmless warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from langchain_core.messages import HumanMessage
from omni_agent.config import AgentConfig
from omni_agent.graph import create_omni_agent

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="OmniAgent — Autonomous AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKSPACE_PATH = Path("./workspace").resolve()
WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)


def get_secret(key: str, default: str = "") -> str:
    """Safely fetch key from Streamlit secrets, environment, or default."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def list_workspace_files():
    """Return a list of relative file paths in workspace."""
    if not WORKSPACE_PATH.exists():
        return []
    files = []
    for p in sorted(WORKSPACE_PATH.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            files.append(str(p.relative_to(WORKSPACE_PATH)))
    return files


# -----------------------------------------------------------------------------
# Sidebar: Settings & Workspace Browser
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 OmniAgent")
    st.caption("Autonomous General-Purpose AI Agent")
    st.divider()

    st.subheader("⚙️ Model Settings")
    provider = st.selectbox(
        "LLM Provider",
        options=["Groq (Ultra-Fast Free Tier)", "Google Gemini (Free Tier)"],
        index=0,
    )

    provider_code = "groq" if "groq" in provider.lower() else "gemini"

    if provider_code == "groq":
        model = st.selectbox(
            "Model",
            options=["llama-3.3-70b-versatile", "qwen/qwen3.8-27b"],
            index=0,
        )
        default_key = get_secret("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API Key",
            value=default_key,
            type="password",
            help="Free at console.groq.com/keys (No credit card needed)",
        )
    else:
        model = st.selectbox(
            "Model",
            options=["gemini-3.6-flash"],
            index=0,
        )
        default_key = get_secret("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Gemini API Key",
            value=default_key,
            type="password",
            help="Free at aistudio.google.com/app/apikey (No credit card needed)",
        )

    max_steps = st.slider("Max Steps per Goal", min_value=5, max_value=30, value=15)

    st.divider()
    st.subheader("📂 Workspace Files")
    files = list_workspace_files()
    if files:
        selected_file = st.selectbox("View File", options=files)
        if selected_file:
            file_path = WORKSPACE_PATH / selected_file
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                ext = file_path.suffix.lstrip(".") or "txt"
                st.code(content, language=ext)
                st.download_button(
                    label=f"⬇️ Download {selected_file}",
                    data=content,
                    file_name=selected_file,
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        st.info("No files generated yet. When OmniAgent writes code or documents, they will appear here.")

    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main Chat & Execution Area
# -----------------------------------------------------------------------------
st.title("🤖 OmniAgent Web Hub")
st.markdown(
    "OmniAgent breaks goals into structured plans, writes & executes code, searches the web, "
    "runs shell commands, and verifies results in real time."
)

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "plan" in msg and msg["plan"]:
            with st.expander("📋 Execution Plan", expanded=False):
                st.markdown(msg["plan"])
        if "trace" in msg and msg["trace"]:
            with st.expander("⚙️ Execution Trace", expanded=False):
                st.markdown(msg["trace"])

# Quick example chips
if not st.session_state.messages:
    st.markdown("##### 💡 Quick Start Examples:")
    c1, c2 = st.columns(2)
    examples = [
        "Build a modern stopwatch web app in workspace with start, pause, and lap times",
        "Calculate the first 25 Fibonacci numbers and save them formatted in fibonacci.txt",
        "Write a Python script to compute prime numbers up to 100 and verify it",
        "Search the web for the latest updates on Python 3.13 and summarize key features",
    ]
    for i, ex in enumerate(examples):
        target_col = c1 if i % 2 == 0 else c2
        if target_col.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.submitted_prompt = ex
            st.rerun()

prompt = st.chat_input("What would you like OmniAgent to accomplish?")
if hasattr(st.session_state, "submitted_prompt") and st.session_state.submitted_prompt:
    prompt = st.session_state.submitted_prompt
    del st.session_state.submitted_prompt

if prompt:
    # Append user goal to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Configure agent
    config = AgentConfig()
    config.workspace_dir = WORKSPACE_PATH
    config.max_steps = int(max_steps)
    config.provider = provider_code
    config.model = model.strip()

    if api_key_input and api_key_input.strip():
        if provider_code == "groq":
            os.environ["GROQ_API_KEY"] = api_key_input.strip()
        else:
            os.environ["GEMINI_API_KEY"] = api_key_input.strip()
            config.gemini_api_key = api_key_input.strip()

    with st.chat_message("assistant"):
        status_box = st.status("🧠 OmniAgent is planning and solving...", expanded=True)
        plan_placeholder = status_box.empty()
        trace_placeholder = status_box.empty()
        final_placeholder = st.empty()

        plan_text = ""
        trace_text = ""
        final_text = ""

        try:
            app = create_omni_agent(config)
            initial_state = {
                "user_goal": prompt.strip(),
                "messages": [HumanMessage(content=prompt.strip())],
                "iteration": 0,
                "max_iterations": config.max_steps,
                "is_completed": False,
            }
            thread_id = str(uuid.uuid4())
            thread_config = {"configurable": {"thread_id": thread_id}}

            for event in app.stream(initial_state, thread_config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "planner":
                        plan = node_output.get("plan")
                        if plan:
                            lines = [f"**Goal:** {plan.goal}\n"]
                            for s in plan.steps:
                                icon = "🟢" if s.status == "completed" else ("🟡" if s.status == "in_progress" else "⚪")
                                lines.append(f"- {icon} **Step {s.step_id}**: {s.description}")
                            plan_text = "\n".join(lines)
                            plan_placeholder.markdown("### 📋 Generated Plan\n" + plan_text)

                    elif node_name == "executor":
                        messages = node_output.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                    if len(args_str) > 100:
                                        args_str = args_str[:100] + "..."
                                    trace_text += f"\n- ⚙️ **Calling Tool**: `{tc['name']}({args_str})`"
                            elif last_msg.content:
                                c = str(last_msg.content).strip()
                                if len(c) > 180:
                                    c = c[:180] + "..."
                                trace_text += f"\n- 💭 *Thought*: {c}"
                            trace_placeholder.markdown("### ⚙️ Execution Activity\n" + trace_text)

                    elif node_name == "tools":
                        messages = node_output.get("messages", [])
                        for m in messages:
                            snippet = str(m.content).strip()
                            if len(snippet) > 180:
                                snippet = snippet[:180] + "..."
                            trace_text += f"\n  - ✓ *Output*: `{snippet}`"
                        trace_placeholder.markdown("### ⚙️ Execution Activity\n" + trace_text)

                    elif node_name == "verifier":
                        feedback = node_output.get("verification_feedback")
                        if feedback:
                            trace_text += f"\n- 🔍 *Verifier*: {feedback[:150]}"
                        elif node_output.get("is_completed"):
                            trace_text += "\n- 🎯 *Verifier*: Step satisfied!"
                        trace_placeholder.markdown("### ⚙️ Execution Activity\n" + trace_text)

                    elif node_name == "synthesizer":
                        final_text = node_output.get("final_response", "")
                        final_placeholder.markdown(final_text)

            status_box.update(label="✅ Task Completed!", state="complete", expanded=False)
            if not final_text:
                final_text = "Task reached completion. Check the execution details and workspace."
                final_placeholder.markdown(final_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "plan": plan_text,
                "trace": trace_text,
            })
            st.rerun()

        except Exception as e:
            status_box.update(label="❌ Execution Error", state="error", expanded=True)
            err_msg = f"**Error encountered:** {str(e)}"
            final_placeholder.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
