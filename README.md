# 🧠 Autonomous Cognitive Engine for Deep Research and Long-Horizon Tasks

> An advanced AI agent framework built with **LangGraph** that enables autonomous planning, context management, and multi-step task execution.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-orange.svg)](https://aistudio.google.com)

---

## 📖 Overview

This project implements a **Deep Cognitive Task Framework** using LangGraph to create sophisticated, autonomous AI agents capable of executing complex, long-horizon tasks. The agent goes beyond simple tool-calling loops to enable:

- **📋 Structured Task Planning** — Decomposes complex goals into manageable sub-tasks
- **📁 Context Offloading** — Virtual file system for managing large amounts of intermediate data
- **🔍 Web Research** — DuckDuckGo search integration for real-time information gathering
- **🔄 ReAct Loop** — Reason → Act → Observe cycle for intelligent decision-making

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User UI   │────▶│  Agent Node  │────▶│  Tool Node  │
│ (Streamlit) │     │  (LLM + Plan)│     │ (Execute)   │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
                    ┌──────▼───────┐      ┌──────▼──────┐
                    │  State Graph │      │   Tools     │
                    │  - Messages  │      │ - Planning  │
                    │  - TODOs     │      │ - File Sys  │
                    │  - Files     │      │ - Search    │
                    └──────────────┘      └─────────────┘
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd POC_PROJECT
python -m venv venv
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key_here    # Get free at https://aistudio.google.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key       # Optional, for tracing
```

### 3. Run the Demo
```bash
python examples/demo_milestone1.py
```

### 4. Launch the UI
```bash
streamlit run frontend/app.py
```

## 📁 Project Structure

```
POC_PROJECT/
├── src/                    # Core source code
│   ├── config.py          # Configuration & env loading
│   ├── agent/             # Agent logic
│   │   ├── state.py       # LangGraph state schema
│   │   ├── graph.py       # Graph construction
│   │   ├── nodes.py       # Agent & tool nodes
│   │   └── prompts.py     # System prompts
│   ├── tools/             # Agent tools
│   │   ├── planning.py    # write_todos, update_todo, get_todos
│   │   ├── filesystem.py  # write_file, read_file, edit_file, ls
│   │   └── search.py      # DuckDuckGo web search
│   └── memory/            # Virtual file system
│       └── virtual_fs.py  # In-memory file system
├── frontend/              # Streamlit UI
│   └── app.py             # Main application
├── tests/                 # Evaluation suite
│   ├── test_planning.py   # Milestone 1 tests
│   └── test_filesystem.py # Milestone 2 tests
├── examples/              # Demo scripts
│   └── demo_milestone1.py # Quick demo
└── requirements.txt       # Dependencies
```

## 🎯 Milestones

### ✅ Milestone 1: Foundational Agent & Task Planning
- ReAct agent loop with Gemini LLM
- `write_todos` — task decomposition tool
- `update_todo` — progress tracking
- `get_todos` — plan review
- **Success: >80% task decomposition accuracy**

### ✅ Milestone 2: Context Offloading via Virtual File System
- `write_file` — save intermediate results
- `read_file` — retrieve saved content
- `edit_file` — modify existing files
- `ls` — list saved files
- **Success: >80% correct file system usage in multi-step scenarios**

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph + LangChain |
| LLM | Google Gemini 2.0 Flash |
| Search | DuckDuckGo (free) |
| Frontend | Streamlit |
| Observability | LangSmith |
| Language | Python 3.13 |

## 📊 Evaluation

Run the evaluation suites:
```bash
# Milestone 1 — Task Planning
python tests/test_planning.py 5

# Milestone 2 — File System
python tests/test_filesystem.py 3
```

---

Built with ❤️ for Infosys POC Project
