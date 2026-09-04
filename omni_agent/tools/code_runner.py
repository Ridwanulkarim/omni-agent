"""Python code execution tool for OmniAgent."""

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

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=workspace, delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(target_code)
        temp_file_path = Path(temp_file.name)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_file_path.resolve())],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output_parts = []
        if result.stdout.strip():
            output_parts.append(f"Output:\n{result.stdout.strip()}")
        if result.stderr.strip():
            output_parts.append(f"Stderr:\n{result.stderr.strip()}")
        if not output_parts:
            output_parts.append("Execution completed successfully with no output.")
        
        output_parts.append(f"[Exit code: {result.returncode}]")
        return "\n\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing Python code: {str(e)}"
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError:
                pass
