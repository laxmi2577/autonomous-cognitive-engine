"""
Quick Demo Script — Autonomous Cognitive Engine
=================================================
Run this to test that the agent works end-to-end.

Usage:
    python -m examples.demo_milestone1
    
    Or from project root:
    python examples/demo_milestone1.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import create_agent
from src.config import validate_config, RECURSION_LIMIT


def run_demo():
    """Run a simple demonstration of the agent's planning capabilities."""
    print("\n" + "🧠" * 30)
    print("  AUTONOMOUS COGNITIVE ENGINE — Milestone 1 Demo")
    print("🧠" * 30 + "\n")
    
    # Validate configuration
    if not validate_config():
        print("\n⚠️  Please set your GOOGLE_API_KEY in the .env file!")
        print("   Get a free key at: https://aistudio.google.com")
        return
    
    # Create the agent
    print("🔧 Building agent graph...")
    agent = create_agent()
    print("✅ Agent ready!\n")
    
    # Demo prompt
    demo_prompt = (
        "Research the current state of artificial intelligence in healthcare. "
        "Cover the following areas: diagnostic AI, drug discovery, personalized "
        "medicine, and ethical challenges. Provide a comprehensive summary."
    )
    
    print(f"👤 USER: {demo_prompt}\n")
    print("🤖 AGENT WORKING...\n")
    print("-" * 60)
    
    # Run the agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=demo_prompt)]},
        config={
            "configurable": {"thread_id": "demo-milestone1"},
            "recursion_limit": RECURSION_LIMIT,
        }
    )
    
    # Display results
    print("-" * 60)
    
    # Show TODO plan
    todos = result.get("todos", [])
    if todos:
        print("\n📋 TASK PLAN CREATED:")
        status_emoji = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
        for todo in todos:
            emoji = status_emoji.get(todo.status, "❓")
            print(f"   {todo.id}. {emoji} {todo.task}")
            if todo.result:
                print(f"      └─ {todo.result}")
    
    # Show files created
    files = result.get("files", {})
    if files:
        print(f"\n📂 FILES CREATED ({len(files)}):")
        for path, content in files.items():
            print(f"   📄 {path} ({len(content)} chars)")
    
    # Show final response
    print("\n🤖 FINAL RESPONSE:")
    print("-" * 60)
    last_ai_message = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            last_ai_message = msg
            break
    
    if last_ai_message:
        print(last_ai_message.content)
    else:
        print("(Agent completed but no final text response was generated)")
    
    print("\n" + "=" * 60)
    print(f"📊 Stats: {result.get('iteration_count', 0)} iterations")
    print(f"📋 TODOs: {len(todos)} tasks")
    print(f"📂 Files: {len(files)} saved")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
