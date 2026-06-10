"""
Configuration module for the Autonomous Cognitive Engine.
Loads environment variables and provides centralized config access.
Supports: Groq LLM + LangSmith Tracing
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Load environment variables from .env file (local dev)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


def _get_secret(key, default=""):
    """Get a config value from os.environ OR Streamlit Cloud secrets."""
    # 1. Try os.environ first (works locally with .env)
    val = os.environ.get(key, "")
    if val:
        return val
    # 2. Try Streamlit Cloud secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# ──────────────────────────────────────────────
# LLM Provider: Groq (fast, free tier)
# ──────────────────────────────────────────────
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
LLM_MODEL = _get_secret("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(_get_secret("LLM_TEMPERATURE", "0.4"))

# ──────────────────────────────────────────────
# LangSmith Observability
# ──────────────────────────────────────────────
LANGCHAIN_TRACING_V2 = _get_secret("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = _get_secret("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = _get_secret("LANGCHAIN_PROJECT", "autonomous-cognitive-engine")

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ──────────────────────────────────────────────
# Agent Configuration
# ──────────────────────────────────────────────
MAX_AGENT_ITERATIONS = int(_get_secret("MAX_AGENT_ITERATIONS", "25"))
RECURSION_LIMIT = int(_get_secret("RECURSION_LIMIT", "50"))


def validate_config():
    """Validate that required configuration is present."""
    global GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
    global LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

    # Re-read fresh (secrets may have been injected after module import)
    GROQ_API_KEY = _get_secret("GROQ_API_KEY")
    LLM_MODEL = _get_secret("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE = float(_get_secret("LLM_TEMPERATURE", "0.4"))
    LANGCHAIN_TRACING_V2 = _get_secret("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY = _get_secret("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = _get_secret("LANGCHAIN_PROJECT", "autonomous-cognitive-engine")

    # Ensure env vars are set for langchain internals
    if GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    if LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

    errors = []
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_here":
        errors.append("GROQ_API_KEY is not set. Get one free at https://console.groq.com/keys")
    if errors:
        print("\n⚠️  Configuration Errors:")
        for err in errors:
            print(f"   ❌ {err}")
        print()
        return False
    print("✅ Configuration loaded successfully.")
    print(f"   Provider: Groq | Model: {LLM_MODEL}")
    if LANGCHAIN_TRACING_V2 == "true" and LANGCHAIN_API_KEY:
        print(f"   LangSmith: ✅ Enabled ({LANGCHAIN_PROJECT})")
    else:
        print("   LangSmith: ⚠️ Disabled (set LANGCHAIN_API_KEY to enable)")
    return True


if __name__ == "__main__":
    validate_config()
