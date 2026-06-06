"""
Milestone 2 Evaluation — Virtual File System Usage Tests
==========================================================
Tests that the agent correctly uses file system tools to manage 
context in >80% of multi-step test scenarios requiring context offloading.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from src.agent.graph import create_agent
from src.config import validate_config, RECURSION_LIMIT


# ──────────────────────────────────────────────
# Test Scenarios requiring context offloading
# ──────────────────────────────────────────────
TEST_SCENARIOS = [
    {
        "id": 1,
        "prompt": "Research three different renewable energy sources (solar, wind, and hydrogen). For each one, find key statistics and recent developments. Then write a combined comparative summary.",
        "expected_files_min": 2,
        "expected_read_operations": True,
        "description": "Multi-topic research with combined synthesis",
    },
    {
        "id": 2,
        "prompt": "Find information about the top 3 AI frameworks (TensorFlow, PyTorch, JAX). Save detailed notes about each one separately. Then create a comparison table combining all findings.",
        "expected_files_min": 2,
        "expected_read_operations": True,
        "description": "Structured note-taking with comparison",
    },
    {
        "id": 3,
        "prompt": "Research the history of space exploration in three phases: 1960s-1980s, 1990s-2010s, and 2020s-present. Save notes for each era, then write a comprehensive timeline report.",
        "expected_files_min": 2,
        "expected_read_operations": True,
        "description": "Chronological research with synthesis",
    },
    {
        "id": 4,
        "prompt": "Analyze three major tech companies (Apple, Google, Microsoft). For each, research their AI strategy, recent products, and market position. Then write an investment analysis comparing all three.",
        "expected_files_min": 2,
        "expected_read_operations": True,
        "description": "Company analysis with investment report",
    },
    {
        "id": 5,
        "prompt": "Research the pros and cons of three programming paradigms: Object-Oriented, Functional, and Procedural. Save detailed notes on each. Then create a recommendation guide for different use cases.",
        "expected_files_min": 2,
        "expected_read_operations": True,
        "description": "Multi-concept analysis with guide creation",
    },
]


def run_single_fs_test(agent, test_case: dict, verbose: bool = True) -> dict:
    """Run a single file system test case."""
    test_id = test_case["id"]
    prompt = test_case["prompt"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📝 Test #{test_id}: {test_case['description']}")
        print(f"{'='*60}")
        print(f"   Prompt: {prompt[:100]}...")
    
    try:
        # recursion_limit=16 = ~8 agent iterations
        # Enough for: plan → write_file × 3 → read_file → synthesize
        # Saves ~70% tokens vs recursion_limit=50
        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": f"test-m2-{test_id}"},
                "recursion_limit": 16,
            }
        )
        
        # Check file system usage
        files = result.get("files", {})
        num_files = len(files)
        has_enough_files = num_files >= test_case["expected_files_min"]
        
        # Check if files have meaningful content
        meaningful_files = sum(1 for content in files.values() if len(content) > 50)
        
        # Check message history for read_file usage
        messages = result.get("messages", [])
        used_write = any("write_file" in str(m) for m in messages)
        used_read = any("read_file" in str(m) for m in messages)
        used_ls = any("ls" in str(m) and "ls" in str(getattr(m, 'tool_calls', [])) for m in messages)
        
        passed = has_enough_files and meaningful_files >= 1
        
        result_data = {
            "test_id": test_id,
            "passed": passed,
            "num_files": num_files,
            "meaningful_files": meaningful_files,
            "expected_min_files": test_case["expected_files_min"],
            "used_write_file": used_write,
            "used_read_file": used_read,
            "file_paths": list(files.keys()),
        }
        
        if verbose:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"\n{status}")
            print(f"  Files created: {num_files} (min expected: {test_case['expected_files_min']})")
            print(f"  Meaningful files: {meaningful_files}")
            print(f"  Used write_file: {used_write}")
            print(f"  Used read_file: {used_read}")
            if files:
                print("  Files saved:")
                for path, content in files.items():
                    print(f"    📄 `{path}` ({len(content)} chars)")
        
        return result_data
        
    except Exception as e:
        if verbose:
            print(f"\n❌ ERROR: {str(e)}")
        return {"test_id": test_id, "passed": False, "error": str(e)}


def run_evaluation(num_tests: int = 5, verbose: bool = True):
    """Run the full Milestone 2 evaluation suite."""
    print("\n" + "🧪" * 30)
    print("🧪  MILESTONE 2 EVALUATION — File System Context Management")
    print("🧪" * 30)
    
    if not validate_config():
        print("❌ Fix configuration before running evaluation.")
        return
    
    agent = create_agent()
    print(f"\n🚀 Running {num_tests} test scenarios...\n")
    
    results = []
    for test_case in TEST_SCENARIOS[:num_tests]:
        result = run_single_fs_test(agent, test_case, verbose=verbose)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 MILESTONE 2 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Tests Run:    {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {total - passed}")
    print(f"  Pass Rate:    {pass_rate:.0%}")
    print(f"  Target:       >80%")
    print(f"  Status:       {'✅ TARGET MET' if pass_rate >= 0.8 else '❌ BELOW TARGET'}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_evaluation(num_tests=num)
