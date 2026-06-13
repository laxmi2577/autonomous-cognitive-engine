"""
Streamlit Frontend for the Autonomous Cognitive Engine.
=========================================================
Gemini-style UI: True black, Google Sans, pill input, capability chips.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.graph import create_agent
from src.agent.state import TodoItem
from src.config import validate_config, LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ACE — Autonomous Cognitive Engine",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Gemini-Style CSS (True Black / Mobile)
# ──────────────────────────────────────────────
st.markdown("""
<div style="display:none">
<link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600&family=Google+Sans+Display:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* ── Global Reset ── */
  *, *::before, *::after { box-sizing: border-box; }

  html, body, [class*="css"], .stApp {
    font-family: 'Google Sans', 'Inter', sans-serif !important;
    background-color: #0D0D0D !important;
    color: #E8EAED !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  .block-container {
    padding: 0 !important;
    max-width: 100% !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #131314 !important;
    border-right: 1px solid #2A2A2E !important;
    padding: 0 !important;
  }
  [data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
  }

  /* Sidebar inner padding */
  .sidebar-inner {
    padding: 1rem 0.75rem;
  }

  /* ACE Brand */
  .ace-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1rem 0.75rem 0.5rem;
    margin-bottom: 0.25rem;
  }
  .ace-brand-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4285F4, #8B5CF6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }
  .ace-brand-name {
    font-size: 1.1rem;
    font-weight: 500;
    color: #E8EAED;
    letter-spacing: -0.01em;
  }

  /* New Chat Button */
  .new-chat-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    background: transparent;
    border: 1px solid #3C3C40;
    border-radius: 24px;
    padding: 0.55rem 1rem;
    margin: 0.5rem 0.75rem 1rem;
    cursor: pointer;
    color: #C4C7CC;
    font-size: 0.875rem;
    font-family: 'Google Sans', sans-serif;
    width: calc(100% - 1.5rem);
    transition: background 0.2s, border-color 0.2s;
    text-align: left;
  }
  .new-chat-btn:hover {
    background: #1E1E22;
    border-color: #5A5A60;
  }

  /* Sidebar Section Label */
  .sb-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #9AA0A6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.75rem 0.75rem 0.35rem;
  }

  /* Plan Step Pills */
  .plan-step {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 0.4rem 0.75rem;
    margin: 0.15rem 0;
    border-radius: 8px;
    font-size: 0.8rem;
    color: #BDC1C6;
    transition: background 0.15s;
  }
  .plan-step:hover { background: #1E1E22; }
  .step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
  }
  .dot-pending  { background: #5F6368; }
  .dot-active   { background: #4285F4; box-shadow: 0 0 6px #4285F4aa; }
  .dot-done     { background: #34A853; }

  /* Agent Chip */
  .agent-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.4rem 0.75rem;
    margin: 0.15rem 0;
    border-radius: 8px;
    font-size: 0.8rem;
    color: #BDC1C6;
    cursor: default;
  }
  .agent-chip:hover { background: #1E1E22; }
  .agent-icon {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
  }
  .icon-search { background: rgba(66,133,244,0.2); }
  .icon-summary { background: rgba(52,168,83,0.2); }
  .icon-code { background: rgba(251,188,4,0.2); }

  /* Sidebar File Item */
  .sb-file {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    color: #9AA0A6;
    border-radius: 6px;
  }
  .sb-file:hover { background: #1E1E22; color: #E8EAED; }

  /* Stat Grid */
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    padding: 0 0.75rem;
    margin-top: 0.25rem;
  }
  .stat-box {
    background: #1A1A1E;
    border-radius: 10px;
    padding: 0.6rem 0.5rem;
    text-align: center;
    border: 1px solid #2A2A2E;
  }
  .stat-val {
    font-size: 1.4rem;
    font-weight: 600;
    background: linear-gradient(135deg, #4285F4, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .stat-lbl {
    font-size: 0.68rem;
    color: #5F6368;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* Sidebar Divider */
  .sb-divider {
    height: 1px;
    background: #2A2A2E;
    margin: 0.75rem 0;
  }

  /* ── Main Chat Area ── */
  .main-area {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 760px;
    margin: 0 auto;
    padding: 0 1rem;
  }

  /* ── Welcome Screen ── */
  .welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 1.5rem 2rem;
    text-align: center;
  }
  .welcome-greeting {
    font-family: 'Google Sans Display', 'Google Sans', sans-serif;
    font-size: clamp(2rem, 6vw, 2.75rem);
    font-weight: 400;
    line-height: 1.2;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #4285F4 0%, #8B5CF6 50%, #EA4335 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: gradientShift 4s ease-in-out infinite alternate;
  }
  @keyframes gradientShift {
    0%   { background-position: 0% center; }
    100% { background-position: 100% center; }
  }
  .welcome-sub {
    font-size: 0.95rem;
    color: #9AA0A6;
    margin-bottom: 2.5rem;
    max-width: 420px;
    line-height: 1.6;
  }

  /* Capability Chips */
  .chips-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    width: 100%;
    max-width: 520px;
    margin-bottom: 2rem;
  }
  .cap-chip {
    background: #1A1A1E;
    border: 1px solid #2A2A2E;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s cubic-bezier(0.2,0,0,1);
    position: relative;
    overflow: hidden;
  }
  .cap-chip::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(66,133,244,0.08), rgba(139,92,246,0.08));
    opacity: 0;
    transition: opacity 0.2s;
  }
  .cap-chip:hover::before { opacity: 1; }
  .cap-chip:hover {
    border-color: #4285F4;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(66,133,244,0.15);
  }
  .chip-icon {
    font-size: 1.3rem;
    margin-bottom: 0.3rem;
    display: block;
  }
  .chip-title {
    font-size: 0.875rem;
    font-weight: 500;
    color: #E8EAED;
    display: block;
  }
  .chip-desc {
    font-size: 0.75rem;
    color: #9AA0A6;
    margin-top: 2px;
    display: block;
  }

  /* ── Chat Messages ── */
  .chat-history {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 0;
    scrollbar-width: thin;
    scrollbar-color: #2A2A2E transparent;
  }

  /* User message */
  .msg-user-wrap {
    display: flex;
    justify-content: flex-end;
    margin: 0.5rem 0 1rem;
    padding: 0 0.5rem;
  }
  .msg-user-bubble {
    background: #1E3A5F;
    border: 1px solid #2D5A8E;
    border-radius: 20px 20px 4px 20px;
    padding: 0.7rem 1.1rem;
    max-width: 75%;
    font-size: 0.9rem;
    line-height: 1.55;
    color: #E8EAED;
    word-wrap: break-word;
  }

  /* AI message */
  .msg-ai-wrap {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 0.5rem 0 1rem;
    padding: 0 0.5rem;
  }
  .ai-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4285F4, #8B5CF6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .msg-ai-content {
    flex: 1;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #E8EAED;
    max-width: calc(100% - 38px);
  }
  .msg-ai-content h1, .msg-ai-content h2, .msg-ai-content h3 {
    color: #E8EAED !important;
    margin-top: 1rem !important;
  }
  .msg-ai-content code {
    background: #1E1E22 !important;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
  }
  .msg-ai-content pre {
    background: #1A1A1E !important;
    border: 1px solid #2A2A2E;
    border-radius: 10px;
    padding: 1rem;
  }
  .msg-ai-content a { color: #4285F4 !important; }

  /* Thinking shimmer */
  .thinking-wrap {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.5rem;
    margin: 0.5rem 0;
  }
  .thinking-lines {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-top: 4px;
  }
  .shimmer-line {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(90deg, #1A1A1E 25%, #2A2A2E 50%, #1A1A1E 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
  }
  .shimmer-line:nth-child(1) { width: 80%; }
  .shimmer-line:nth-child(2) { width: 65%; }
  .shimmer-line:nth-child(3) { width: 45%; }
  @keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  /* Thinking label */
  .thinking-label {
    font-size: 0.8rem;
    color: #4285F4;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 0.5rem;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
  }

  /* ── Input Bar ── */
  .input-container {
    padding: 1rem 0 1.5rem;
    position: sticky;
    bottom: 0;
    background: linear-gradient(to top, #0D0D0D 80%, transparent);
  }

  /* Override Streamlit chat input */
  [data-testid="stChatInput"] {
    background: #1A1A1E !important;
    border: 1px solid #3C3C40 !important;
    border-radius: 28px !important;
    padding: 0.75rem 1.25rem !important;
    font-family: 'Google Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #E8EAED !important;
    transition: border-color 0.2s !important;
  }
  [data-testid="stChatInput"]:focus-within {
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 2px rgba(66,133,244,0.15) !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #E8EAED !important;
    font-family: 'Google Sans', sans-serif !important;
  }
  [data-testid="stChatInput"] textarea::placeholder {
    color: #5F6368 !important;
  }

  /* Send button */
  [data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #4285F4, #8B5CF6) !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
  }

  /* ── Stremlit Widget Overrides ── */
  .stButton > button {
    background: transparent !important;
    border: 1px solid #3C3C40 !important;
    border-radius: 24px !important;
    color: #C4C7CC !important;
    font-family: 'Google Sans', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #1E1E22 !important;
    border-color: #5A5A60 !important;
    color: #E8EAED !important;
  }

  /* Progress bar */
  [data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4285F4, #8B5CF6) !important;
    border-radius: 4px !important;
  }
  [data-testid="stProgress"] > div {
    background: #1A1A1E !important;
    border-radius: 4px !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    border: 1px solid #2A2A2E !important;
    border-radius: 10px !important;
    background: #131314 !important;
  }
  [data-testid="stExpander"] summary {
    color: #BDC1C6 !important;
    font-size: 0.83rem !important;
    font-family: 'Google Sans', sans-serif !important;
  }

  /* Info box */
  .stAlert {
    background: #131314 !important;
    border: 1px solid #2A2A2E !important;
    border-radius: 10px !important;
    font-size: 0.83rem !important;
    color: #9AA0A6 !important;
  }

  /* Metric */
  [data-testid="metric-container"] {
    background: #1A1A1E !important;
    border: 1px solid #2A2A2E !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
  }

  /* Error */
  .stAlert[data-baseweb="notification"] {
    background: rgba(234,67,53,0.1) !important;
    border-color: rgba(234,67,53,0.3) !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2A2A2E; border-radius: 2px; }
</style>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.thread_id = "ace-session-1"
    st.session_state.chat_history = []
    st.session_state.todos = []
    st.session_state.files = {}
    st.session_state.iteration_count = 0
    st.session_state.delegations = []
    st.session_state.initialized = False
    st.session_state.is_thinking = False


def initialize_agent():
    """Initialize the agent if not already done."""
    if st.session_state.agent is None:
        if validate_config():
            st.session_state.agent = create_agent()
            st.session_state.initialized = True
            return True
        return False
    return True


def build_report_from_files(files: dict, ai_response: str) -> str:
    """If the AI's final response is generic, build report from VFS files."""
    generic_phrases = [
        "task completed", "check the sidebar", "check sidebar",
        "results are in", "see the files",
    ]
    is_generic = (
        not ai_response.strip()
        or len(ai_response.strip()) < 100
        or any(phrase in ai_response.lower() for phrase in generic_phrases)
    )
    if not is_generic:
        return ai_response
    if not files:
        return ai_response or "✅ Task completed."
    report_parts = ["## 📋 Comprehensive Research Report\n"]
    report_parts.append("*Generated from collected research files:*\n\n---\n")
    for filepath, content in sorted(files.items()):
        report_parts.append(f"### 📄 {filepath}\n")
        report_parts.append(content)
        report_parts.append("\n\n---\n")
    return "\n".join(report_parts)


# ──────────────────────────────────────────────
# Sidebar — Gemini Style
# ──────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="ace-brand">
      <div class="ace-brand-icon">✦</div>
      <span class="ace-brand-name">ACE</span>
    </div>
    """, unsafe_allow_html=True)

    # New Session button
    if st.button("✎  New chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.chat_history = []
        st.session_state.todos = []
        st.session_state.files = {}
        st.session_state.delegations = []
        st.session_state.iteration_count = 0
        st.session_state.thread_id = f"ace-session-{abs(hash(str(st.session_state)))}"
        st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Task Plan ──
    st.markdown('<div class="sb-label">Task Plan</div>', unsafe_allow_html=True)

    if st.session_state.todos:
        completed = sum(1 for t in st.session_state.todos if t.get("status") == "completed")
        total = len(st.session_state.todos)
        st.progress(completed / total if total > 0 else 0,
                    text=f"{completed}/{total} steps done")
        st.markdown("<div style='margin-top:6px'>", unsafe_allow_html=True)
        for todo in st.session_state.todos:
            status = todo.get("status", "pending")
            dot_class = {"pending": "dot-pending", "in_progress": "dot-active", "completed": "dot-done"}.get(status, "dot-pending")
            task_text = todo.get("task", "")[:55] + ("…" if len(todo.get("task", "")) > 55 else "")
            st.markdown(
                f'<div class="plan-step">'
                f'<div class="step-dot {dot_class}"></div>'
                f'<span style="font-size:0.8rem;color:#BDC1C6">{task_text}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:0 0.75rem;font-size:0.8rem;color:#5F6368">Send a task to see the plan</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Agent Calls ──
    st.markdown('<div class="sb-label">Agent Activity</div>', unsafe_allow_html=True)

    if getattr(st.session_state, "delegations", []):
        agent_meta = {
            "search_agent":  ("🌐", "icon-search",  "Search"),
            "summary_agent": ("📝", "icon-summary", "Summary"),
            "coding_agent":  ("💻", "icon-code",    "Code"),
        }
        for deleg in st.session_state.delegations:
            a_name = deleg["agent"]
            a_task = deleg["prompt"][:45] + ("…" if len(deleg["prompt"]) > 45 else "")
            icon, cls, label = agent_meta.get(a_name, ("🤖", "icon-search", a_name))
            with st.expander(f"{icon} {label} agent"):
                st.markdown(f"<span style='font-size:0.78rem;color:#9AA0A6'>{deleg['prompt'][:200]}</span>", unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:0 0.75rem;font-size:0.8rem;color:#5F6368">No agents called yet</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Virtual Files ──
    st.markdown('<div class="sb-label">Research Files</div>', unsafe_allow_html=True)

    if st.session_state.files:
        for filepath, content in st.session_state.files.items():
            fname = Path(filepath).name
            with st.expander(f"📄 {fname}"):
                st.code(content[:400] + ("…" if len(content) > 400 else ""), language="markdown")
    else:
        st.markdown('<div style="padding:0 0.75rem;font-size:0.8rem;color:#5F6368">No files saved yet</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Stats ──
    st.markdown('<div class="sb-label">Session</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-val">{len(st.session_state.chat_history)}</div>
        <div class="stat-lbl">Messages</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{st.session_state.iteration_count}</div>
        <div class="stat-lbl">Steps</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{len(st.session_state.todos)}</div>
        <div class="stat-lbl">Tasks</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{len(st.session_state.files)}</div>
        <div class="stat-lbl">Files</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # LangSmith
    if LANGCHAIN_TRACING_V2 == "true" and LANGCHAIN_API_KEY:
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:0 0.75rem">
          <a href="https://smith.langchain.com/o/default/projects/p/{LANGCHAIN_PROJECT}"
             target="_blank"
             style="font-size:0.8rem;color:#4285F4;text-decoration:none;display:flex;align-items:center;gap:6px">
            📊 LangSmith Traces →
          </a>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown('<div class="main-area">', unsafe_allow_html=True)

# Check config — show error if not configured
if not initialize_agent():
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:70vh;text-align:center;padding:2rem">
      <div style="font-size:3rem;margin-bottom:1rem">⚠️</div>
      <div style="font-size:1.2rem;font-weight:500;color:#EA4335;margin-bottom:0.5rem">
        Configuration Error
      </div>
      <div style="font-size:0.9rem;color:#9AA0A6;max-width:360px;line-height:1.6">
        <code style="background:#1A1A1E;padding:2px 8px;border-radius:4px;color:#FBBC04">GROQ_API_KEY</code>
        is not set.<br>
        Add it in Streamlit Cloud → Manage App → Secrets,<br>or set it in your local <code>.env</code> file.
      </div>
      <a href="https://console.groq.com/keys" target="_blank"
         style="margin-top:1.5rem;background:linear-gradient(135deg,#4285F4,#8B5CF6);
                color:white;padding:0.6rem 1.5rem;border-radius:24px;
                font-size:0.875rem;text-decoration:none;font-weight:500">
        Get Free API Key →
      </a>
    </div>
    """, unsafe_allow_html=True)
    # Debug info
    try:
        secret_keys = list(st.secrets.keys()) if hasattr(st, "secrets") else []
        if secret_keys:
            st.info(f"🔍 Secrets found: {secret_keys} — but GROQ_API_KEY may be misspelled")
        else:
            st.warning("🔍 No secrets found in Streamlit Cloud settings")
    except Exception as e:
        st.warning(f"🔍 Secrets error: {e}")
    st.stop()


# ── Render chat history ──
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user-wrap">'
                f'<div class="msg-user-bubble">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # Use st.markdown inside the AI bubble div for proper markdown rendering
            st.markdown('<div class="msg-ai-wrap"><div class="ai-avatar">✦</div><div class="msg-ai-content">', unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown('</div></div>', unsafe_allow_html=True)

else:
    # ── Welcome Screen ──
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-greeting">Hello, Researcher ✨</div>
      <div class="welcome-sub">
        Your Autonomous Research Agent — powered by LangGraph + Groq.<br>
        What would you like to explore today?
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Capability Chips
    chip_data = [
        ("🔬", "Deep Research",    "Comprehensive reports on any topic"),
        ("📊", "Compare & Analyze","Side-by-side analysis & insights"),
        ("💡", "Brainstorm Ideas", "Generate creative ideas & strategies"),
        ("📄", "Write Report",     "Structured documents & summaries"),
        ("🌐", "Web Intelligence", "Search & synthesize live information"),
        ("💻", "Code & Build",     "Technical analysis & code generation"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(chip_data):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(
                f"{icon}  {title}\n{desc}",
                key=f"chip_{i}",
                use_container_width=True,
            ):
                prompts = {
                    "Deep Research":    "Research the latest advancements in quantum computing and write a comprehensive report.",
                    "Compare & Analyze":"Compare AWS, Azure, and GCP cloud platforms across AI/ML services, pricing, and developer experience.",
                    "Brainstorm Ideas": "Brainstorm innovative startup ideas in the AI and sustainability space for 2025.",
                    "Write Report":     "Write a detailed executive report on the state of generative AI in enterprise applications.",
                    "Web Intelligence": "Search and summarize the latest breakthroughs in large language model research from the past 3 months.",
                    "Code & Build":     "Analyze the best Python frameworks for building production-grade AI agents in 2025.",
                }
                st.session_state.pending_prompt = prompts.get(title, f"Tell me about {title}")
                st.rerun()

# Handle pending prompt from chips
pending = st.session_state.pop("pending_prompt", None)

# ── Chat Input ──
user_input = st.chat_input("Ask ACE anything — I'll research, plan, and deliver…")

prompt = pending or user_input

if prompt:
    # Add and display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.markdown(
        f'<div class="msg-user-wrap">'
        f'<div class="msg-user-bubble">{prompt}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Thinking animation
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
    <div class="thinking-wrap">
      <div class="ai-avatar">✦</div>
      <div class="thinking-lines">
        <div class="thinking-label">✦ ACE is researching…</div>
        <div class="shimmer-line"></div>
        <div class="shimmer-line"></div>
        <div class="shimmer-line"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Reset delegation counter
        from src.tools.delegation import reset_delegation_counter
        reset_delegation_counter()

        # Run the agent
        result = st.session_state.agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": st.session_state.thread_id},
                "recursion_limit": 80,
            }
        )

        # Update todos
        todos = result.get("todos", [])
        if todos:
            st.session_state.todos = [
                {"id": t.id, "task": t.task, "status": t.status, "result": t.result}
                for t in todos
            ]

        # Update files
        files = result.get("files", {})
        if files:
            st.session_state.files = files

        st.session_state.iteration_count = result.get("iteration_count", 0)

        # Extract delegations
        delegations = []
        messages_list = result.get("messages", [])
        for i, msg in enumerate(messages_list):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "delegate_task":
                        tc_id = tc["id"]
                        tc_args = tc["args"]
                        tool_res = "Executing…"
                        for m in messages_list[i:]:
                            if isinstance(m, ToolMessage) and m.tool_call_id == tc_id:
                                tool_res = m.content
                                break
                        delegations.append({
                            "agent": tc_args.get("agent_name", "unknown"),
                            "prompt": tc_args.get("task_prompt", ""),
                            "result": tool_res
                        })
        st.session_state.delegations = delegations

        # Fallback: synthetic todos from delegations
        if not st.session_state.todos and delegations:
            st.session_state.todos = [
                {"id": i+1, "task": d["prompt"][:100], "status": "completed",
                 "result": f"Handled by {d['agent']}"}
                for i, d in enumerate(delegations)
            ]

        # Extract final AI response
        final_response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_response = msg.content
                break

        final_response = build_report_from_files(files, final_response)

        # Clear thinking, show response
        thinking_placeholder.empty()
        st.markdown('<div class="msg-ai-wrap"><div class="ai-avatar">✦</div><div class="msg-ai-content">', unsafe_allow_html=True)
        st.markdown(final_response)
        st.markdown('</div></div>', unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_response
        })

    except Exception as e:
        thinking_placeholder.empty()
        error_msg = f"❌ **Error:** {str(e)}"
        st.markdown(
            f'<div class="msg-ai-wrap"><div class="ai-avatar">✦</div>'
            f'<div class="msg-ai-content" style="color:#EA4335">{error_msg}</div></div>',
            unsafe_allow_html=True
        )
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
