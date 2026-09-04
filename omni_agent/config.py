"""Configuration management for OmniAgent."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class AgentConfig(BaseModel):
    """OmniAgent configuration parameters."""

    # Model settings
    provider: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_PROVIDER", "gemini").lower()
    )
    model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "gemini-3.6-flash")
    )
    temperature: float = 0.2

    # API Keys
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Runtime and Sandbox constraints
    workspace_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve()
    )
    max_steps: int = Field(
        default_factory=lambda: int(os.getenv("MAX_STEPS", "15"))
    )
    tool_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("TOOL_TIMEOUT_SECONDS", "60"))
    )
    enable_shell: bool = True
    enable_web: bool = True
    enable_python_exec: bool = True

    def ensure_workspace(self) -> Path:
        """Ensure the agent's sandbox workspace directory exists."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir


default_config = AgentConfig()
