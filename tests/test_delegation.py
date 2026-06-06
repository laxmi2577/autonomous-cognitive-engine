"""
Milestone 3 Evaluation — Sub-Agent Delegation Tests
===================================================
Tests that the sub-agents execute successfully and that the main agent
can successfully delegate tasks using the delegate_task tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from src.agent.graph import create_agent
from src.agent.sub_agents.registry import run_sub_agent
from src.config import validate_config


def test_isolated_sub_agents():
    """Verify that specialized sub-agents can run directly via run_sub_agent."""
    print("\n--- Testing Isolated Sub-Agents Direct Execution ---")
    
    # 1. Test search_agent
    print("\n🌐 Testing search_agent...")
    search_task = "Search for the latest key statistics about solar energy capacity in 2024."
    files = {}
    try:
        response, updated_files = run_sub_agent(
            agent_name="search_agent",
            task_prompt=search_task,
            virtual_files=files,
            max_iterations=3
        )
        print("✅ search_agent completed successfully!")
        print(f"   Response Length: {len(response)} chars")
        print(f"   VFS files updated: {list(updated_files.keys())}")
        assert len(response) > 0, "Response should not be empty"
    except Exception as e:
        print(f"❌ search_agent test failed: {e}")
        raise e

    # 2. Test summary_agent
    print("\n📝 Testing summary_agent...")
    summary_task = "Read the file 'notes/solar.md' and draft a high-level summary."
    files = {
        "notes/solar.md": "Solar energy is growing at a record pace. In 2024, global solar capacity reached 1.5 Terawatts, representing a 25% increase compared to 2023. Key drivers include lower module costs and high policy support in China and the European Union."
    }
    try:
        response, updated_files = run_sub_agent(
            agent_name="summary_agent",
            task_prompt=summary_task,
            virtual_files=files,
            max_iterations=3
        )
        print("✅ summary_agent completed successfully!")
        print(f"   Response: {response[:150]}...")
        print(f"   VFS files updated: {list(updated_files.keys())}")
        assert len(response) > 0, "Response should not be empty"
    except Exception as e:
        print(f"❌ summary_agent test failed: {e}")
        raise e


def test_main_agent_delegation():
    """Verify that the main agent graph can use delegate_task to delegate work."""
    print("\n--- Testing Graph-Level Sub-Agent Delegation ---")
    
    agent = create_agent()
    prompt = "Research the top renewable energy source today using your search sub-agent. Save the findings to 'energy.md', then use your summary sub-agent to draft a report."
    
    try:
        print("🚀 Invoking main agent graph...")
        # recursion_limit=16 to allow multiple iterations of planning and delegation
        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": "test-delegation-graph"},
                "recursion_limit": 16,
            }
        )
        
        # Verify execution history contains delegate_task calls
        messages = result.get("messages", [])
        used_delegation = any("delegate_task" in str(m) for m in messages)
        files = result.get("files", {})
        
        print("\n📊 Graph-level Results:")
        print(f"   Used delegation: {used_delegation}")
        print(f"   VFS files saved: {list(files.keys())}")
        
        if files:
            for path, content in files.items():
                print(f"   📄 `{path}` ({len(content)} chars)")
                
        assert used_delegation, "Main agent failed to call delegate_task tool"
        print("\n✅ Graph delegation test passed!")
        
    except Exception as e:
        print(f"❌ Graph delegation test failed: {e}")
        raise e


def run_all_tests():
    if not validate_config():
        print("❌ Invalid configuration. Configure your .env first.")
        return
        
    try:
        test_isolated_sub_agents()
        test_main_agent_delegation()
        print("\n🏆 ALL MILESTONE 3 TESTS PASSED SUCCESSFULLY!")
    except Exception:
        print("\n❌ SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
