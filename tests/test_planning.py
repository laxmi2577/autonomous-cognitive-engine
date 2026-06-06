"""
Milestone 1 Evaluation — Task Decomposition Accuracy Tests
=============================================================
Tests that the agent correctly uses write_todos to create relevant
and structured task plans for >80% of test requests.

TOKEN OPTIMIZATION: Uses recursion_limit=6 (only 3 agent iterations)
instead of 50. We only need to check if the agent creates a plan,
not complete the full research task.
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
# Test Prompts — 10 varied complex requests
# ──────────────────────────────────────────────
TEST_PROMPTS = [
    {
        "id": 1,
        "prompt": "Research the current state of quantum computing and write a comprehensive summary covering major companies, recent breakthroughs, and practical applications.",
        "expected_min_tasks": 3,
        "expected_keywords": ["quantum", "research", "company", "application", "summary"],
    },
    {
        "id": 2,
        "prompt": "Compare the top 5 programming languages for AI development in 2025, covering their strengths, weaknesses, ecosystem, and community support.",
        "expected_min_tasks": 3,
        "expected_keywords": ["programming", "language", "compare", "AI"],
    },
    {
        "id": 3,
        "prompt": "Create a detailed analysis of climate change impacts on global food production, including current data, future projections, and potential solutions.",
        "expected_min_tasks": 3,
        "expected_keywords": ["climate", "food", "impact", "solution"],
    },
    {
        "id": 4,
        "prompt": "Research and summarize the latest developments in autonomous vehicles, covering technology, regulation, safety, and market adoption.",
        "expected_min_tasks": 3,
        "expected_keywords": ["autonomous", "vehicle", "technology", "safety"],
    },
    {
        "id": 5,
        "prompt": "Write a comprehensive guide on microservices architecture, covering design patterns, tools, deployment strategies, and common pitfalls.",
        "expected_min_tasks": 3,
        "expected_keywords": ["microservice", "architecture", "pattern", "deployment"],
    },
    {
        "id": 6,
        "prompt": "Analyze the impact of artificial intelligence on healthcare, covering diagnosis, drug discovery, patient care, and ethical considerations.",
        "expected_min_tasks": 3,
        "expected_keywords": ["AI", "healthcare", "diagnosis", "ethical"],
    },
    {
        "id": 7,
        "prompt": "Research renewable energy trends globally, covering solar, wind, and hydrogen technologies, with cost analysis and adoption rates.",
        "expected_min_tasks": 3,
        "expected_keywords": ["renewable", "energy", "solar", "wind"],
    },
    {
        "id": 8,
        "prompt": "Create a detailed comparison of cloud computing platforms (AWS, Azure, GCP) covering services, pricing, performance, and best use cases.",
        "expected_min_tasks": 3,
        "expected_keywords": ["cloud", "AWS", "Azure", "compare"],
    },
    {
        "id": 9,
        "prompt": "Research the evolution of cybersecurity threats in 2024-2025, covering attack types, defense strategies, and emerging technologies.",
        "expected_min_tasks": 3,
        "expected_keywords": ["cybersecurity", "threat", "defense", "attack"],
    },
    {
        "id": 10,
        "prompt": "Analyze the global electric vehicle market, covering major manufacturers, battery technology, infrastructure, and market projections.",
        "expected_min_tasks": 3,
        "expected_keywords": ["electric", "vehicle", "battery", "market"],
    },
]


def run_single_test(agent, test_case: dict, verbose: bool = True) -> dict:
    """Run a single test case and evaluate if the agent creates a proper plan."""
    test_id = test_case["id"]
    prompt = test_case["prompt"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📝 Test #{test_id}: {prompt[:80]}...")
        print(f"{'='*60}")
    
    try:
        # KEY: recursion_limit=6 means max 3 agent loops
        # That's enough to: agent→tools(write_todos)→agent→tools→agent→end
        # This saves ~90% tokens vs recursion_limit=50
        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": f"test-m1-{test_id}"},
                "recursion_limit": 6,
            }
        )
        
        # Check if todos were created
        todos = result.get("todos", [])
        plan_created = result.get("plan_created", False)
        
        # Evaluate
        has_plan = len(todos) > 0 or plan_created
        has_enough_tasks = len(todos) >= test_case["expected_min_tasks"]
        
        # Check if tasks are relevant (simple keyword matching)
        task_text = " ".join(t.task.lower() for t in todos) if todos else ""
        keywords_found = sum(
            1 for kw in test_case["expected_keywords"]
            if kw.lower() in task_text
        )
        keyword_score = keywords_found / len(test_case["expected_keywords"]) if test_case["expected_keywords"] else 0
        
        passed = has_plan and has_enough_tasks and keyword_score >= 0.3
        
        result_data = {
            "test_id": test_id,
            "passed": passed,
            "has_plan": has_plan,
            "num_todos": len(todos),
            "expected_min": test_case["expected_min_tasks"],
            "keyword_score": round(keyword_score, 2),
            "todos": [{"id": t.id, "task": t.task, "status": t.status} for t in todos],
        }
        
        if verbose:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"\n{status}")
            print(f"  Plan created: {has_plan}")
            print(f"  Tasks: {len(todos)} (min expected: {test_case['expected_min_tasks']})")
            print(f"  Keyword relevance: {keyword_score:.0%}")
            if todos:
                print("  Tasks:")
                for t in todos:
                    print(f"    {t.id}. {t.task}")
        
        return result_data
        
    except Exception as e:
        if verbose:
            print(f"\n❌ ERROR: {str(e)[:200]}")
        return {
            "test_id": test_id,
            "passed": False,
            "error": str(e)[:200],
        }


def run_evaluation(num_tests: int = 5, verbose: bool = True):
    """Run the full Milestone 1 evaluation suite."""
    print("\n" + "🧪" * 30)
    print("🧪  MILESTONE 1 EVALUATION — Task Decomposition Accuracy")
    print("🧪" * 30)
    
    if not validate_config():
        print("❌ Fix configuration before running evaluation.")
        return
    
    agent = create_agent()
    print(f"\n🚀 Running {num_tests} test cases (token-optimized)...\n")
    
    results = []
    for test_case in TEST_PROMPTS[:num_tests]:
        result = run_single_test(agent, test_case, verbose=verbose)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 MILESTONE 1 EVALUATION RESULTS")
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
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_evaluation(num_tests=num)
