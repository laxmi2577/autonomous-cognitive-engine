"""
Streamlit Frontend for the Autonomous Cognitive Engine.
=========================================================
Features:
- Real-time chat with the AI agent
- TODO plan visualization with progress bar
- Virtual file system browser
- Delegation activity panel (search, summary, coding agents)
- Final comprehensive report displayed in main chat
- LangSmith tracing link
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
    page_title="🧠 Autonomous Cognitive Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.8rem !important;
        margin: 0 !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.8) !important;
        margin: 0.3rem 0 0 0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    /* Todo cards */
    .todo-item {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid;
        font-size: 0.85rem;
    }
    .todo-pending { border-color: #ffa726; }
    .todo-in-progress { border-color: #42a5f5; }
    .todo-completed { border-color: #66bb6a; }
    
    /* File cards */
    .file-card {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid #ab47bc;
        font-size: 0.85rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Chat styling */
    .stChatMessage {
        border-radius: 12px !important;
    }
    
    /* Delegation cards */
    .delegation-item {
        background: rgba(102, 126, 234, 0.08);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid #667eea;
        font-size: 0.85rem;
    }
    
    /* Final report styling */
    .final-report {
        background: rgba(102, 126, 234, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.thread_id = "streamlit-session-1"
    st.session_state.chat_history = []
    st.session_state.todos = []
    st.session_state.files = {}
    st.session_state.iteration_count = 0
    st.session_state.delegations = []
    st.session_state.initialized = False


def initialize_agent():
    """Initialize the agent if not already done."""
    if st.session_state.agent is None:
        if validate_config():
            st.session_state.agent = create_agent()
            st.session_state.initialized = True
            return True
        return False
    return True


# ──────────────────────────────────────────────
# Helper: Build final report from VFS files
# ──────────────────────────────────────────────
def build_report_from_files(files: dict, ai_response: str) -> str:
    """
    If the AI's final response is empty or generic ('task completed'),
    build a comprehensive report from VFS files instead.
    """
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
    
    # Build a combined report from all VFS files
    report_parts = ["## 📋 Comprehensive Research Report\n"]
    report_parts.append("*Generated from collected research files:*\n\n---\n")
    
    for filepath, content in sorted(files.items()):
        report_parts.append(f"### 📄 {filepath}\n")
        report_parts.append(content)
        report_parts.append("\n\n---\n")
    
    return "\n".join(report_parts)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Cognitive Engine")
    st.markdown("---")
    
    # ── TODO Plan Section ──
    st.markdown("### 📋 Task Plan")
    
    if st.session_state.todos:
        status_emoji = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
        completed = sum(1 for t in st.session_state.todos if t.get("status") == "completed")
        total = len(st.session_state.todos)
        
        progress = completed / total if total > 0 else 0
        st.progress(progress, text=f"{completed}/{total} tasks done")
        
        for todo in st.session_state.todos:
            status = todo.get("status", "pending")
            emoji = status_emoji.get(status, "❓")
            css_class = f"todo-{status.replace('_', '-')}"
            
            result_html = ""
            if todo.get("result"):
                result_html = f'<br><small style="color: #aaa;">└─ {todo["result"][:80]}</small>'
            
            st.markdown(
                f'<div class="todo-item {css_class}">'
                f'{emoji} <strong>#{todo["id"]}</strong> {todo["task"]}'
                f'{result_html}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No plan created yet. Send a complex request to get started!")
    
    st.markdown("---")
    
    # ── Delegation Activity Section ──
    st.markdown("### 🤝 Delegation Activity")
    
    if getattr(st.session_state, "delegations", []):
        for deleg in st.session_state.delegations:
            agent = deleg["agent"]
            task = deleg["prompt"]
            res = deleg["result"]
            
            agent_emoji = {"search_agent": "🌐", "summary_agent": "📝", "coding_agent": "💻"}
            emoji = agent_emoji.get(agent, "🤝")
            
            with st.expander(f"{emoji} `{agent}`"):
                st.markdown(f"**Task:** {task}")
                st.markdown(f"**Output:**\n{res[:500]}{'...' if len(res) > 500 else ''}")
    else:
        st.info("No delegation activity yet.")
        
    st.markdown("---")
    
    # ── Virtual File System Section ──
    st.markdown("### 📂 Virtual Files")
    
    if st.session_state.files:
        for filepath, content in st.session_state.files.items():
            with st.expander(f"📄 {filepath} ({len(content)} chars)"):
                st.code(content[:500] + ("..." if len(content) > 500 else ""), language="markdown")
    else:
        st.info("No files saved yet.")
    
    st.markdown("---")
    
    # ── Stats Section ──
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.chat_history))
    with col2:
        st.metric("Iterations", st.session_state.iteration_count)
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("TODOs", len(st.session_state.todos))
    with col4:
        st.metric("Files", len(st.session_state.files))
    
    # ── LangSmith Link ──
    if LANGCHAIN_TRACING_V2 == "true" and LANGCHAIN_API_KEY:
        st.markdown("---")
        st.markdown("### 🔍 LangSmith Tracing")
        st.markdown(f"[📊 View Traces →](https://smith.langchain.com/o/default/projects/p/{LANGCHAIN_PROJECT})")
        st.success("✅ Tracing enabled")
    
    st.markdown("---")
    
    # ── Controls ──
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.todos = []
        st.session_state.files = {}
        st.session_state.delegations = []
        st.session_state.iteration_count = 0
        st.session_state.thread_id = f"streamlit-session-{hash(str(st.session_state))}"
        st.rerun()


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

# Header
st.markdown(
    '<div class="main-header">'
    '<h1>🧠 Autonomous Cognitive Engine</h1>'
    '<p>Deep Research & Long-Horizon Task Agent — Powered by LangGraph + Groq</p>'
    '</div>',
    unsafe_allow_html=True
)

# Check config
if not initialize_agent():
    st.error(
        "⚠️ **Configuration Error**: Please set your `GROQ_API_KEY`.\n\n"
        "Get an API key at: [Groq Console](https://console.groq.com/keys)"
    )
    # Debug info to diagnose secrets issue
    try:
        secret_keys = list(st.secrets.keys()) if hasattr(st, "secrets") else []
        st.warning(f"🔍 Debug: Streamlit secrets keys found: {secret_keys}")
    except Exception as e:
        st.warning(f"🔍 Debug: st.secrets error: {e}")
    env_groq = os.environ.get("GROQ_API_KEY", "NOT FOUND")
    st.warning(f"🔍 Debug: os.environ GROQ_API_KEY = {'SET ✅' if env_groq and env_groq != 'NOT FOUND' else 'NOT FOUND ❌'}")
    st.stop()

# Display chat history
for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role, avatar="👤" if role == "user" else "🧠"):
        st.markdown(content)

# Example prompts for new sessions
if not st.session_state.chat_history:
    st.markdown("### 💡 Try one of these:")
    
    example_prompts = [
        "Research the current state of quantum computing and write a comprehensive report covering major companies, breakthroughs, and applications.",
        "Compare the top 3 cloud platforms (AWS, Azure, GCP) in terms of AI/ML services, pricing, and developer experience.",
        "Analyze the impact of AI on healthcare, covering diagnostics, drug discovery, and ethical considerations.",
    ]
    
    cols = st.columns(len(example_prompts))
    for i, (col, prompt) in enumerate(zip(cols, example_prompts)):
        with col:
            if st.button(f"📝 Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

# Handle pending prompt from example buttons
pending = st.session_state.pop("pending_prompt", None)

# Chat input
user_input = st.chat_input("Ask me anything — I'll plan, research, and deliver...")

# Use pending prompt if available
prompt = pending or user_input

if prompt:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Show thinking indicator
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🔄 Planning and executing..."):
            try:
                # Reset delegation cooldown counter for new request
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
                
                # Update session state with results
                todos = result.get("todos", [])
                if todos:
                    st.session_state.todos = [
                        {"id": t.id, "task": t.task, "status": t.status, "result": t.result}
                        for t in todos
                    ]
                
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
                                tool_res = "Executing..."
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
                
                # FALLBACK: If no todos, reconstruct plan from delegation activity
                if not st.session_state.todos and delegations:
                    synthetic_todos = []
                    for i, deleg in enumerate(delegations, 1):
                        synthetic_todos.append({
                            "id": i,
                            "task": deleg["prompt"][:100],
                            "status": "completed",
                            "result": f"Handled by {deleg['agent']}"
                        })
                    st.session_state.todos = synthetic_todos
                
                # ── Extract final AI response ──
                final_response = ""
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        final_response = msg.content
                        break
                
                # ── KEY FIX: Build report from files if response is generic ──
                final_response = build_report_from_files(files, final_response)
                
                st.markdown(final_response)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
    
    st.rerun()
