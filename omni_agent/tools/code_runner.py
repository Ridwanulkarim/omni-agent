"""Python code execution tool for OmniAgent."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from omni_agent.config import default_config, AgentConfig

_active_config: AgentConfig = default_config


def set_active_config(config: AgentConfig):
    global _active_config
    _active_config = config


@tool
def execute_python(
    code: Optional[str] = None,
    script: Optional[str] = None,
    python_code: Optional[str] = None,
    timeout: int = 15,
) -> str:
    """Execute Python code in an isolated subprocess and return the output (stdout and stderr).
    
    The code runs inside the agent workspace directory. Use this tool for data analysis,
    math calculations, testing logic, verifying algorithms, or processing data.
    
    Args:
        code: Complete Python script or code block to run.
        script: Alias for code.
        python_code: Alias for code.
        timeout: Maximum execution time in seconds (default: 15).
    """
    target_code = code or script or python_code
    if not target_code:
        return "Error: Missing required Python code to execute."
    workspace = _active_config.workspace_dir.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    import io
    import contextlib

    old_cwd = os.getcwd()
    try:
        os.chdir(workspace)
        f_out = io.StringIO()
        f_err = io.StringIO()
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            globs = {"__name__": "__main__"}
            exec(target_code, globs)
        stdout = f_out.getvalue().strip()
        stderr = f_err.getvalue().strip()
        output_parts = []
        if stdout:
            output_parts.append(f"Output:\n{stdout}")
        if stderr:
            output_parts.append(f"Stderr:\n{stderr}")
        if not output_parts:
            output_parts.append("Execution completed successfully with no output.")
        return "\n\n".join(output_parts)
    except Exception as e:
        return f"Error executing Python code: {str(e)}"
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
