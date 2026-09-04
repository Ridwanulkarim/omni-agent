"""OmniAgent: Autonomous General-Purpose AI Agent."""

import warnings

# Suppress harmless platform SSL warnings and langchain deprecation notices
warnings.filterwarnings("ignore")

from omni_agent.config import AgentConfig, default_config
from omni_agent.graph import create_omni_agent
from omni_agent.tools import get_default_tools

__version__ = "0.1.0"
__all__ = ["AgentConfig", "default_config", "create_omni_agent", "get_default_tools"]
