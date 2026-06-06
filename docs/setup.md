# Setup Guide

## Prerequisites

- **Python 3.11+** — Required for type hint syntax
- **Groq API Key** — Free at [console.groq.com/keys](https://console.groq.com/keys)
- **LangSmith API Key** (optional) — For tracing at [smith.langchain.com](https://smith.langchain.com)

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd POC_PROJECT
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:

```env
# Required — Groq LLM
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.4

# Optional — LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_key_here
LANGCHAIN_PROJECT=autonomous-cognitive-engine

# Agent Config
MAX_AGENT_ITERATIONS=25
RECURSION_LIMIT=80
```

### 5. Run the Application
```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```
POC_PROJECT/
├── frontend/
│   └── app.py              # Streamlit UI
├── src/
│   ├── config.py            # Environment config loader
│   ├── agent/
│   │   ├── graph.py         # LangGraph StateGraph definition
│   │   ├── nodes.py         # Agent & tool node functions
│   │   ├── prompts.py       # System prompts
│   │   ├── state.py         # AgentState schema
│   │   └── sub_agents/
│   │       └── registry.py  # Sub-agent definitions & runner
│   ├── tools/
│   │   ├── planning.py      # write_todos, update_todo, get_todos
│   │   ├── filesystem.py    # read_file, write_file, edit_file, ls
│   │   ├── search.py        # web_search
│   │   └── delegation.py    # delegate_task
│   └── memory/              # Memory management utilities
├── tests/
│   ├── test_planning.py     # Planning tool tests
│   ├── test_filesystem.py   # VFS tool tests
│   ├── test_delegation.py   # Delegation tests
│   ├── test_integration.py  # End-to-end tests
│   └── evaluation.py        # LLM-as-a-judge evaluator
├── docs/
│   ├── architecture.md      # System architecture
│   ├── setup.md             # This file
│   └── api.md               # Tool & API reference
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── README.md                # Project overview
```

## Running Tests

```bash
# Unit tests
python -m pytest tests/test_planning.py tests/test_filesystem.py tests/test_delegation.py -v

# Integration tests
python -m pytest tests/test_integration.py -v

# LLM-as-a-judge evaluation
python tests/evaluation.py --count 5
```

## Troubleshooting

### Rate Limit Errors (429)
**Symptom:** `Rate limit reached for model llama-3.3-70b-versatile`

**Cause:** Groq free tier limits:
- 30 requests/minute
- 100,000 tokens/day

**Fix:**
1. Wait 1-2 minutes between runs
2. If daily limit hit, wait until midnight UTC
3. Consider upgrading to Groq Dev Tier ($0)

### XML Parse Errors
**Symptom:** `🔧 Fixing Groq tool format` in terminal

**Cause:** Groq sometimes returns tool calls in XML format instead of JSON

**Fix:** Automatic — the system includes a 3-layer XML parser that handles this. No action needed.

### Empty Task Plan
**Symptom:** Task Plan sidebar shows "No plan created yet"

**Cause:** write_todos tool call was skipped by the LLM

**Fix:** Automatic — the system includes a programmatic fallback that creates a plan after 2 iterations. Also, the UI reconstructs tasks from delegation activity.

### LangSmith Not Working
**Symptom:** "LangSmith: ⚠️ Disabled" in terminal

**Fix:** Ensure `.env` has:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_key_here
```
