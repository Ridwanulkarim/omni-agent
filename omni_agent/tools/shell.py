"""Safe shell execution tool for OmniAgent."""

import shutil
import subprocess
from typing import Optional
from langchain_core.tools import tool
from omni_agent.config import default_config, AgentConfig

_active_config: AgentConfig = default_config


def set_active_config(config: AgentConfig):
    global _active_config
    _active_config = config


@tool
def run_shell_command(
    command: Optional[str] = None,
    cmd: Optional[str] = None,
    timeout: int = 15,
) -> str:
    """Execute a bash shell command inside the workspace directory.
    
    Use this tool to run command-line utilities, install workspace packages, git commands,
    file inspections, or running external scripts.
    
    Args:
        command: The shell command line to execute.
        cmd: Alias for command.
        timeout: Maximum duration in seconds before terminating the process (default: 15).
    """
    target_cmd = command or cmd
    if not target_cmd:
        return "Error: Missing required shell command to execute."

    workspace = _active_config.workspace_dir.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    shell_bin = shutil.which("bash") or "/bin/sh"

    try:
        result = subprocess.run(
            target_cmd,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable=shell_bin,
        )
        output_parts = []
        if result.stdout.strip():
            output_parts.append(f"Stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            output_parts.append(f"Stderr:\n{result.stderr.strip()}")
        if not output_parts:
            output_parts.append("Command completed with empty output.")
            
        output_parts.append(f"[Exit code: {result.returncode}]")
        return "\n\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing shell command: {str(e)}"
