"""
Configuration module for the Autonomous Cognitive Engine.
Loads environment variables and provides centralized config access.
Supports: Groq LLM + LangSmith Tracing
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Load environment variables from .env file (local)
# or from Streamlit Cloud secrets (deployed)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# On Streamlit Cloud, inject st.secrets into os.environ
try:
    import streamlit as st
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ[key] = value
except Exception:
    pass


def _load_config():
    """Read all config values from os.environ (called fresh each time)."""
    return {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
        "LLM_MODEL": os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
        "LLM_TEMPERATURE": float(os.environ.get("LLM_TEMPERATURE", "0.4")),
        "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2", "false"),
        "LANGCHAIN_API_KEY": os.environ.get("LANGCHAIN_API_KEY", ""),
        "LANGCHAIN_PROJECT": os.environ.get("LANGCHAIN_PROJECT", "autonomous-cognitive-engine"),
        "MAX_AGENT_ITERATIONS": int(os.environ.get("MAX_AGENT_ITERATIONS", "25")),
        "RECURSION_LIMIT": int(os.environ.get("RECURSION_LIMIT", "50")),
    }


# ──────────────────────────────────────────────
# Module-level config (for backward compatibility)
# ──────────────────────────────────────────────
_cfg = _load_config()
GROQ_API_KEY = _cfg["GROQ_API_KEY"]
LLM_MODEL = _cfg["LLM_MODEL"]
LLM_TEMPERATURE = _cfg["LLM_TEMPERATURE"]
LANGCHAIN_TRACING_V2 = _cfg["LANGCHAIN_TRACING_V2"]
LANGCHAIN_API_KEY = _cfg["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = _cfg["LANGCHAIN_PROJECT"]
MAX_AGENT_ITERATIONS = _cfg["MAX_AGENT_ITERATIONS"]
RECURSION_LIMIT = _cfg["RECURSION_LIMIT"]

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT


def validate_config():
    """Validate that required configuration is present."""
    global GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
    global LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
    
    # Re-read from os.environ (secrets may have been injected after import)
    cfg = _load_config()
    GROQ_API_KEY = cfg["GROQ_API_KEY"]
    LLM_MODEL = cfg["LLM_MODEL"]
    LLM_TEMPERATURE = cfg["LLM_TEMPERATURE"]
    LANGCHAIN_TRACING_V2 = cfg["LANGCHAIN_TRACING_V2"]
    LANGCHAIN_API_KEY = cfg["LANGCHAIN_API_KEY"]
    LANGCHAIN_PROJECT = cfg["LANGCHAIN_PROJECT"]
    
    # Set env vars for langchain
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

