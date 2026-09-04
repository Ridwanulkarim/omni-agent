"""Tools suite for OmniAgent."""

from typing import List
from langchain_core.tools import BaseTool

from omni_agent.config import AgentConfig, default_config
from omni_agent.tools.filesystem import (
    read_file,
    write_file,
    list_directory,
    file_search,
    set_active_config as set_fs_config,
)
from omni_agent.tools.code_runner import (
    execute_python,
    set_active_config as set_code_config,
)
from omni_agent.tools.shell import (
    run_shell_command,
    set_active_config as set_shell_config,
)
from omni_agent.tools.web import (
    web_search,
    fetch_web_page,
)


def get_default_tools(config: AgentConfig = default_config) -> List[BaseTool]:
    """Return the suite of default tools enabled in configuration."""
    set_fs_config(config)
    set_code_config(config)
    set_shell_config(config)

    tools: List[BaseTool] = [
        read_file,
        write_file,
        list_directory,
        file_search,
    ]

    if config.enable_python_exec:
        tools.append(execute_python)

    if config.enable_shell:
        tools.append(run_shell_command)

    if config.enable_web:
        tools.extend([web_search, fetch_web_page])

    return tools


__all__ = [
    "get_default_tools",
    "read_file",
    "write_file",
    "list_directory",
    "file_search",
    "execute_python",
    "run_shell_command",
    "web_search",
    "fetch_web_page",
]
