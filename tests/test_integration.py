"""
Milestone 4 Evaluation — End-to-End Autonomous Research Integration Tests
========================================================================
Tests the fully integrated agent on complex, multi-step research requests,
verifying that the plan-research-delegate-synthesize pipeline completes successfully.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from src.agent.graph import create_agent
from src.config import validate_config

# ──────────────────────────────────────────────
# E2E Test Prompts representing complex research
# ──────────────────────────────────────────────
E2E_SCENARIOS = [
    {
        "id": 1,
        "prompt": "Research the 2024 breakthroughs in nuclear fusion energy. Delegate web search to gather stats, save them, and delegate drafting to summary agent to create a report.",
        "description": "Nuclear fusion breakthroughs research",
        "expected_min_todos": 3,
        "expected_min_files": 2,
    },
    {
        "id": 2,
        "prompt": "Analyze how AI is reshaping software engineering roles in 2025. Save research notes on coding tools, then generate a comprehensive market impact report.",
        "description": "AI impact on software engineering",
        "expected_min_todos": 3,
        "expected_min_files": 2,
    },
    {
        "id": 3,
        "prompt": "Investigate the global microchip supply chain status for AI accelerators in 2024. Extract key figures using search agent, then synthesize an industry outlook.",
        "description": "Microchip supply chain status",
        "expected_min_todos": 3,
        "expected_min_files": 2,
    }
]


def run_e2e_scenario(agent, scenario: dict, verbose: bool = True) -> dict:
    """Run a single end-to-end scenario and evaluate metrics."""
    test_id = scenario["id"]
    prompt = scenario["prompt"]
    desc = scenario["description"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎬 E2E Scenario #{test_id}: {desc}")
        print(f"{'='*60}")
        print(f"   Prompt: {prompt}\n")
        
    try:
        # recursion_limit=30 is enough for a complete planning + multiple delegations run
        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": f"e2e-scenario-{test_id}"},
                "recursion_limit": 30,
            }
        )
        
        todos = result.get("todos", [])
        files = result.get("files", {})
        messages = result.get("messages", [])
        
        # Check success criteria
        plan_created = len(todos) >= scenario["expected_min_todos"]
        files_saved = len(files) >= scenario["expected_min_files"]
        
        # Check for sub-agent delegation
        used_delegation = any("delegate_task" in str(m) for m in messages)
        
        # Check for final synthesized output
        has_final_response = False
        final_response_preview = ""
        from langchain_core.messages import AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                has_final_response = True
                final_response_preview = msg.content[:200] + "..."
                break
                
        passed = plan_created and files_saved and used_delegation and has_final_response
        
        result_data = {
            "test_id": test_id,
            "description": desc,
            "passed": passed,
            "plan_created": plan_created,
            "num_todos": len(todos),
            "files_saved": files_saved,
            "num_files": len(files),
            "used_delegation": used_delegation,
            "has_final_response": has_final_response,
            "preview": final_response_preview,
            "file_paths": list(files.keys())
        }
        
        if verbose:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"\n{status}")
            print(f"  Plan Tasks Created: {len(todos)} (Expected: >={scenario['expected_min_todos']})")
            print(f"  Files Saved:        {len(files)} (Expected: >={scenario['expected_min_files']})")
            print(f"  Used Delegation:    {used_delegation}")
            print(f"  Final Report:       {'Yes' if has_final_response else 'No'}")
            if files:
                print("  Generated Virtual Files:")
                for path, content in files.items():
                    print(f"    📄 `{path}` ({len(content)} chars)")
            if has_final_response:
                print(f"  Report Preview:\n{final_response_preview}")
                
        return result_data
        
    except Exception as e:
        if verbose:
            print(f"\n❌ SCENARIO ERROR: {e}")
        return {
            "test_id": test_id,
            "description": desc,
            "passed": False,
            "error": str(e)
        }


def run_e2e_evaluation(verbose: bool = True):
    """Run the complete end-to-end evaluation suite."""
    print("\n" + "🌟" * 30)
    print("🌟  MILESTONE 4 — End-to-End Autonomous Research Evaluation")
    print("🌟" * 30)
    
    if not validate_config():
        print("❌ Invalid configuration. Configure your .env first.")
        return
        
    agent = create_agent()
    print(f"\n🚀 Running {len(E2E_SCENARIOS)} complex end-to-end scenarios...\n")
    
    results = []
    for scenario in E2E_SCENARIOS:
        res = run_e2e_scenario(agent, scenario, verbose=verbose)
        results.append(res)
        
    # Calculate pass rate
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 MILESTONE 4 E2E EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Scenarios Run: {total}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {total - passed}")
    print(f"  Success Rate:  {pass_rate:.0%}")
    print(f"  Target:        >80%")
    print(f"  Status:        {'✅ TARGET MET' if pass_rate >= 0.8 else '❌ BELOW TARGET (Rotation or network failure)'}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    run_e2e_evaluation()
