"""Single place that constructs the chat model.

Provider-agnostic on purpose: every provider worth using here speaks the same
OpenAI-compatible API, so switching is three environment variables rather than a
code change. That matters because free tiers run out — when one provider starts
returning 402, you want to move without touching the graph.

Pick a provider by setting LLM_PROVIDER, or point LLM_BASE_URL anywhere yourself.

    # Hugging Face (default)
    $env:HF_TOKEN = "hf_..."

    # Groq — generous free tier, very fast
    $env:LLM_PROVIDER = "groq"
    $env:LLM_API_KEY  = "gsk_..."

    # Ollama — local, unlimited, no key, no account
    $env:LLM_PROVIDER = "ollama"
"""

import os

from langchain_openai import ChatOpenAI

from configuration import Configuration

# base_url, default model, env var holding the key, whether a key is required
PROVIDERS = {
    "huggingface": (
        "https://router.huggingface.co/v1",
        "meta-llama/Llama-3.1-8B-Instruct",
        "HF_TOKEN",
        True,
    ),
    # Groq rotates its catalogue often — run `python list_models.py` if this 404s.
    "groq": (
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "GROQ_API_KEY",
        True,
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.0-flash",
        "GEMINI_API_KEY",
        True,
    ),
    # Local: no account, no credits, no rate limit. Slower on CPU.
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", "", False),
}


def _resolve() -> tuple[str, str, str]:
    """Return (base_url, model, api_key), letting explicit env vars win."""
    name = os.environ.get("LLM_PROVIDER", "huggingface").lower()
    if name not in PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER {name!r}. Choose one of: {', '.join(PROVIDERS)}"
        )

    base_url, default_model, key_var, needs_key = PROVIDERS[name]
    base_url = os.environ.get("LLM_BASE_URL", base_url)
    model = os.environ.get("LLM_MODEL") or os.environ.get("HF_MODEL") or default_model

    # LLM_API_KEY overrides, so a provider can be used without its usual var name.
    api_key = os.environ.get("LLM_API_KEY") or (os.environ.get(key_var, "") if key_var else "")
    if needs_key and not api_key:
        raise RuntimeError(
            f"{key_var} is not set, so provider {name!r} can't be used.\n"
            f"Either set {key_var}, or switch provider — e.g. "
            "$env:LLM_PROVIDER = 'ollama' to run locally with no key."
        )

    return base_url, model, api_key or "not-needed"


def describe() -> str:
    """Human-readable current provider, for the UI sidebar."""
    try:
        base_url, model, _ = _resolve()
        return f"{os.environ.get('LLM_PROVIDER', 'huggingface')} · {model}"
    except RuntimeError as exc:
        return f"not configured ({exc.args[0].splitlines()[0]})"


def get_llm(temperature: float = 0) -> ChatOpenAI:
    base_url, model, api_key = _resolve()
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_retries=2,
    )


# Kept so Configuration.model still reflects whatever is actually in use.
def current_model() -> str:
    return _resolve()[1]
