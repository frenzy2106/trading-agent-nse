"""
LLM provider factory.

Selects the chat model based on the LLM_PROVIDER env var:
    - deepseek (default)  : DeepSeek V4 Flash via their OpenAI-compatible endpoint
    - groq                : Groq's hosted models (legacy fallback)
    - anthropic           : Claude API (premium fallback)

Each provider reads its own API key + model name env vars so multiple can be
configured at once and switched per-run via LLM_PROVIDER=<name>.

Usage:
    from llm_factory import get_llm, get_provider_and_model
    llm = get_llm(tools=TOOLS, temperature=0)        # backtest path
    llm = get_llm(tools=TOOLS)                        # live path, default temperature
    provider, model = get_provider_and_model()
"""

import os
import sys


DEFAULTS = {
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat", "https://api.deepseek.com"),
    "groq":     ("GROQ_API_KEY",     "GROQ_MODEL",     "openai/gpt-oss-120b", None),
    "anthropic":("ANTHROPIC_API_KEY","ANTHROPIC_MODEL","claude-sonnet-4-5-20250929", None),
}


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "deepseek").lower().strip()


def get_provider_and_model() -> tuple[str, str]:
    """Resolve (provider, model) without instantiating the LLM. Used for trace logging."""
    provider = get_provider()
    if provider not in DEFAULTS:
        return provider, "unknown"
    _, model_var, default_model, _ = DEFAULTS[provider]
    return provider, os.getenv(model_var, default_model)


def get_llm(tools, temperature: float | None = None):
    """Build a tool-bound chat model. `temperature=None` uses the provider default."""
    provider = get_provider()
    if provider not in DEFAULTS:
        sys.exit(
            f"ERROR: Unknown LLM_PROVIDER='{provider}'. "
            f"Use one of: {', '.join(DEFAULTS)}"
        )

    key_var, model_var, default_model, base_url = DEFAULTS[provider]
    api_key = os.getenv(key_var)
    if not api_key:
        sys.exit(f"ERROR: {key_var} not set in .env (required when LLM_PROVIDER={provider})")
    model = os.getenv(model_var, default_model)

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model, "api_key": api_key, "base_url": base_url}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs).bind_tools(tools)

    if provider == "groq":
        from langchain_groq import ChatGroq
        kwargs = {"model": model, "api_key": api_key}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatGroq(**kwargs).bind_tools(tools)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = {"model": model, "api_key": api_key}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs).bind_tools(tools)

    raise RuntimeError(f"unreachable: provider={provider}")
