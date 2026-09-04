"""Model provider factory for OmniAgent supporting Free & Paid providers:
Gemini (Free tier), Groq (Free tier), GitHub Models (Free tier), Ollama (100% Local/Free), OpenAI, Anthropic.
"""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from omni_agent.config import AgentConfig


def get_chat_model(config: Optional[AgentConfig] = None) -> BaseChatModel:
    """Instantiate and return the configured chat model."""
    if config is None:
        from omni_agent.config import default_config
        config = default_config

    provider = config.provider.lower().strip()
    model_name = config.model

    # 1. Google Gemini (Has a 100% free tier at aistudio.google.com with NO credit card)
    if provider in ("gemini", "google"):
        api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found.\n"
                "Tip: You can get a 100% FREE Gemini API key with NO billing/credit card at:\n"
                "👉 https://aistudio.google.com/app/apikey"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-3.6-flash",
                google_api_key=api_key,
                temperature=config.temperature,
                convert_system_message_to_human=False,
            )
        except ImportError:
            raise ImportError("Run: pip install langchain-google-genai")

    # 2. Groq (100% Free tier, ultra-fast, NO credit card needed at console.groq.com)
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key not found.\n"
                "Tip: Groq offers free, ultra-fast access to open models with NO credit card.\n"
                "👉 Get a free key at https://console.groq.com/keys"
            )
        from langchain_openai import ChatOpenAI
        target_model = model_name or "qwen/qwen3.8-27b"
        if "llama" in target_model.lower():
            target_model = "qwen/qwen3.8-27b"
        return ChatOpenAI(
            model=target_model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=config.temperature,
            max_retries=2,
        )

    # 3. GitHub Models (100% Free with standard GitHub Personal Access Token)
    elif provider in ("github", "github_models"):
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if not token:
            raise ValueError(
                "GitHub Token not found.\n"
                "Tip: GitHub offers free model inference using your GitHub account token.\n"
                "👉 Create a free token at https://github.com/settings/tokens"
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            api_key=token,
            base_url="https://models.inference.ai.azure.com",
            temperature=config.temperature,
        )

    # 4. Ollama (100% Local on your machine, ZERO accounts, ZERO keys, ZERO billing)
    elif provider in ("ollama", "local"):
        from langchain_openai import ChatOpenAI
        base_url = config.ollama_base_url
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        return ChatOpenAI(
            model=model_name or "llama3.2",
            base_url=base_url,
            api_key="ollama",
            temperature=config.temperature,
        )

    # 5. OpenAI
    elif provider == "openai":
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            api_key=api_key,
            temperature=config.temperature,
        )

    # 6. Anthropic
    elif provider == "anthropic":
        api_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in .env.")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or "claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=config.temperature,
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Available options: 'gemini' (free tier), 'groq' (free), 'github' (free), 'ollama' (free/local), 'openai', 'anthropic'."
        )
