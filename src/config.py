"""
Configuration module for the Autonomous Cognitive Engine.
Loads environment variables and provides centralized config access.
Supports: Groq LLM + LangSmith Tracing
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Load environment variables from .env file
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


# ──────────────────────────────────────────────
# LLM Provider: Groq (fast, free tier)
# ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))

# ──────────────────────────────────────────────
# LangSmith Observability
# ──────────────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "autonomous-cognitive-engine")

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ──────────────────────────────────────────────
# Agent Configuration
# ──────────────────────────────────────────────
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "25"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "50"))


def validate_config():
    """Validate that required configuration is present."""
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
