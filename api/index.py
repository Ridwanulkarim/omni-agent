import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path so omni_agent can be imported
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="OmniAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_init_error = None
try:
    from langchain_core.messages import HumanMessage
    from omni_agent.config import AgentConfig
    from omni_agent.graph import create_omni_agent
except Exception as e:
    import traceback
    _init_error = f"Import Error: {str(e)}\n{traceback.format_exc()}"


class AgentRequest(BaseModel):
    goal: str
    provider: Optional[str] = "groq"
    model: Optional[str] = "qwen/qwen3.8-27b"
    api_key: Optional[str] = None
    max_steps: Optional[int] = 8


@app.get("/")
@app.get("/api")
@app.get("/api/health")
@app.get("/health")
@app.get("/{full_path:path}")
def handle_get(full_path: str = ""):
    if _init_error:
        return {"status": "error", "detail": _init_error}
    return {
        "status": "ok",
        "service": "OmniAgent on Vercel",
        "path": full_path,
    }


@app.post("/")
@app.post("/api")
@app.post("/api/run")
@app.post("/run")
@app.post("/{full_path:path}")
def run_goal(req: AgentRequest, full_path: str = ""):
    if _init_error:
        raise HTTPException(status_code=500, detail=f"Backend startup issue: {_init_error}")
    if not req.goal or not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")

    # On Vercel, the only writable directory is /tmp
    is_vercel = os.getenv("VERCEL", "0") == "1"
    workspace_path = Path("/tmp/omni_workspace").resolve() if is_vercel else Path("./workspace").resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    raw_provider = (req.provider or "groq").strip().lower()
    raw_model = (req.model or "qwen/qwen3.8-27b").strip()

    # Map deprecated or unavailable models to active ones
    if "llama" in raw_model.lower():
        raw_model = "qwen/qwen3.8-27b"

    # Smart auto-detection to prevent mismatches
    if "gemini" in raw_model.lower():
        provider_clean = "gemini"
    elif "qwen" in raw_model.lower() or "gpt-oss" in raw_model.lower() or raw_provider == "groq":
        provider_clean = "groq"
    else:
        provider_clean = raw_provider

    config = AgentConfig()
    config.workspace_dir = workspace_path
    config.max_steps = min(req.max_steps or 4, 6)
    config.provider = provider_clean
    config.model = raw_model

    # API key resolution: request body, or environment variables
    if req.api_key and req.api_key.strip():
        key_val = req.api_key.strip()
        if provider_clean == "groq":
            os.environ["GROQ_API_KEY"] = key_val
        else:
            os.environ["GEMINI_API_KEY"] = key_val
            config.gemini_api_key = key_val

    # Verify that the required API key is available
    if provider_clean == "groq" and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="Missing Groq API Key. Please enter your Groq API key in the 'API Key' field or set GROQ_API_KEY in Vercel Environment Variables.",
        )
    elif provider_clean == "gemini" and not (config.gemini_api_key or os.getenv("GEMINI_API_KEY")):
        raise HTTPException(
            status_code=400,
            detail="Missing Gemini API Key. Please enter your Gemini API key in the 'API Key' field or set GEMINI_API_KEY in Vercel Environment Variables.",
        )

    try:
        app_engine = create_omni_agent(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent initialization error: {str(e)}")

    initial_state = {
        "user_goal": req.goal.strip(),
        "messages": [HumanMessage(content=req.goal.strip())],
        "iteration": 0,
        "max_iterations": config.max_steps,
        "is_completed": False,
    }
    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    steps_log = []
    final_response = ""
    plan_info = None

    try:
        for event in app_engine.stream(initial_state, thread_config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "planner":
                    plan = node_output.get("plan")
                    if plan:
                        plan_info = {
                            "goal": plan.goal,
                            "steps": [
                                {
                                    "step_id": s.step_id,
                                    "description": s.description,
                                    "status": s.status,
                                }
                                for s in plan.steps
                            ],
                        }
                elif node_name == "executor":
                    msgs = node_output.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        if hasattr(last, "tool_calls") and last.tool_calls:
                            for tc in last.tool_calls:
                                steps_log.append({
                                    "type": "tool_call",
                                    "name": tc["name"],
                                    "args": tc["args"],
                                })
                        elif last.content:
                            steps_log.append({
                                "type": "thought",
                                "content": str(last.content)[:250],
                            })
                elif node_name == "tools":
                    msgs = node_output.get("messages", [])
                    for m in msgs:
                        steps_log.append({
                            "type": "tool_output",
                            "content": str(m.content)[:300],
                        })
                elif node_name == "verifier":
                    steps_log.append({
                        "type": "verifier",
                        "is_completed": node_output.get("is_completed", False),
                        "feedback": node_output.get("verification_feedback"),
                    })
                elif node_name == "synthesizer":
                    final_response = node_output.get("final_response", "")

        # Inspect any generated files
        created_files = []
        if workspace_path.exists():
            for f in sorted(workspace_path.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        created_files.append({
                            "name": str(f.relative_to(workspace_path)),
                            "content": f.read_text(encoding="utf-8", errors="replace")[:4000],
                        })
                    except Exception:
                        pass

        return {
            "success": True,
            "plan": plan_info,
            "steps": steps_log,
            "final_response": final_response or "Goal executed.",
            "files": created_files,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
