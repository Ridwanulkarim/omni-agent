"""Filesystem tools for OmniAgent with workspace sandboxing."""

import os
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from omni_agent.config import default_config, AgentConfig

_active_config: AgentConfig = default_config


def set_active_config(config: AgentConfig):
    global _active_config
    _active_config = config


def _resolve_safe_path(target_path: str) -> Path:
    """Resolve a path relative to the workspace directory and prevent path traversal."""
    workspace = _active_config.workspace_dir.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Clean target
    clean_target = Path(target_path).expanduser()
    if not clean_target.is_absolute():
        resolved = (workspace / clean_target).resolve()
    else:
        resolved = clean_target.resolve()

    # Safety check: allow paths within workspace
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise PermissionError(
            f"Access denied: path '{target_path}' is outside the authorized workspace directory '{workspace}'."
        )

    return resolved


@tool
def read_file(file_path: Optional[str] = None, filepath: Optional[str] = None, path: Optional[str] = None) -> str:
    """Read and return the complete text contents of a file inside the workspace.
    
    Args:
        file_path: Relative path to the file within the workspace.
        filepath: Alias for file_path.
        path: Alias for file_path.
    """
    target = file_path or filepath or path
    if not target:
        return "Error: Missing required file path parameter."
    try:
        safe_path = _resolve_safe_path(target)
        if not safe_path.exists():
            return f"Error: File '{target}' does not exist."
        if not safe_path.is_file():
            return f"Error: Path '{target}' is a directory, not a file."
        return safe_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(
    file_path: Optional[str] = None,
    content: Optional[str] = None,
    filepath: Optional[str] = None,
    path: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """Write content to a file in the workspace. Creates parent directories if needed.
    
    Args:
        file_path: Relative path to the destination file.
        content: The text content to write.
        filepath: Alias for file_path.
        path: Alias for file_path.
        text: Alias for content.
    """
    target_path = file_path or filepath or path
    if not target_path:
        return "Error: Missing required file path parameter."
    target_content = content if content is not None else (text if text is not None else "")

    try:
        safe_path = _resolve_safe_path(target_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(target_content, encoding="utf-8")
        return f"Successfully wrote {len(target_content)} characters to '{target_path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_directory(
    dir_path: Optional[str] = None,
    directory: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """List files and subdirectories inside the workspace directory.
    
    Args:
        dir_path: Relative path to the directory (default: workspace root '.').
        directory: Alias for dir_path.
        path: Alias for dir_path.
    """
    target_dir = dir_path or directory or path or "."
    try:
        safe_path = _resolve_safe_path(target_dir)
        if not safe_path.exists():
            return f"Error: Directory '{target_dir}' does not exist."
        if not safe_path.is_dir():
            return f"Error: '{target_dir}' is not a directory."

        entries = []
        for item in sorted(safe_path.iterdir()):
            kind = "[DIR]" if item.is_dir() else "[FILE]"
            size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            entries.append(f"{kind} {item.name}{size}")

        if not entries:
            return f"Directory '{target_dir}' is empty."
        return "\n".join(entries)
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def file_search(
    pattern: Optional[str] = None,
    query: Optional[str] = None,
    dir_path: Optional[str] = None,
    directory: Optional[str] = None,
) -> str:
    """Search for files matching a glob pattern (e.g., '*.py' or '**/*.json') within the workspace.
    
    Args:
        pattern: The glob search pattern.
        query: Alias for pattern.
        dir_path: Starting subdirectory (default: ".").
        directory: Alias for dir_path.
    """
    target_pattern = pattern or query or "*"
    target_dir = dir_path or directory or "."
    try:
        safe_path = _resolve_safe_path(target_dir)
        matches = list(safe_path.glob(target_pattern))
        if not matches:
            return f"No files matched pattern '{target_pattern}' in '{target_dir}'."
        workspace = _active_config.workspace_dir.resolve()
        rel_paths = [str(p.relative_to(workspace)) for p in matches[:50]]
        return "\n".join(rel_paths)
    except Exception as e:
        return f"Error searching files: {str(e)}"
