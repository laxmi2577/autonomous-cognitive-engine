"""
LLM-as-a-Judge Evaluation Suite for the Autonomous Cognitive Engine.
=====================================================================
Runs test prompts through the agent and uses a separate LLM call
to judge the quality of each output.

Usage:
    python tests/evaluation.py              # Run all 15 prompts
    python tests/evaluation.py --count 5    # Run first 5 prompts
    python tests/evaluation.py --count 3 --verbose  # Verbose mode
"""

import sys
import json
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from src.config import validate_config, GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from src.agent.graph import create_agent


# ──────────────────────────────────────────────
# Evaluation Criteria
# ──────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator. Rate the following AI agent output on these criteria.
The agent was asked: "{query}"

The agent produced this output:
---
{output}
---

Rate EACH criterion from 1-4:
1 = Poor (missing/wrong)
2 = Fair (partial/incomplete)
3 = Good (solid coverage)
4 = Excellent (comprehensive, well-structured)

Criteria:
1. COMPLETENESS: Does the output fully address the user's query?
2. STRUCTURE: Is the output well-organized with headings, lists, and sections?
3. ACCURACY: Are the facts reasonable and not obviously hallucinated?
4. DEPTH: Does it go beyond surface-level and provide meaningful analysis?

ALSO check these AGENT FEATURES:
- PLAN_USED: Did the agent create a task plan? (yes/no)
- DELEGATION_USED: Did the agent delegate to sub-agents? (yes/no)
- FILES_CREATED: Did the agent save files to the virtual file system? (yes/no)

Respond ONLY in this exact JSON format:
{{
    "completeness": <1-4>,
    "structure": <1-4>,
    "accuracy": <1-4>,
    "depth": <1-4>,
    "plan_used": <true/false>,
    "delegation_used": <true/false>,
    "files_created": <true/false>,
    "overall_rating": "<poor|fair|good|excellent>",
    "feedback": "<one sentence summary>"
}}"""


class EvaluationRunner:
    """Runs test prompts through the agent and judges output quality."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.agent = None
        self.judge_llm = None
        self.results = []
        
    def setup(self):
        """Initialize agent and judge LLM."""
        print("🔧 Setting up evaluation environment...")
        
        if not validate_config():
            print("❌ Configuration invalid. Fix .env file first.")
            sys.exit(1)
        
        self.agent = create_agent()
        self.judge_llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,  # Low temperature for consistent judging
        )
        print("✅ Agent and judge LLM ready.\n")
    
    def load_prompts(self, count: int = None) -> list[dict]:
        """Load test prompts from JSON file."""
        prompts_path = Path(__file__).parent / "test_prompts.json"
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
        
        if count:
            prompts = prompts[:count]
        
        print(f"📋 Loaded {len(prompts)} test prompts.\n")
        return prompts
    
    def run_agent(self, prompt: str) -> dict:
        """Run a single prompt through the agent and capture results."""
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={
                    "configurable": {"thread_id": f"eval-{datetime.now().timestamp()}"},
                    "recursion_limit": 80,
                }
            )
            
            # Extract final response
            final_response = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_response = msg.content
                    break
            
            # Extract metadata
            todos = result.get("todos", [])
            files = result.get("files", {})
            
            # Count delegations
            delegation_count = 0
            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] == "delegate_task":
                            delegation_count += 1
            
            return {
                "response": final_response,
                "todo_count": len(todos),
                "file_count": len(files),
                "delegation_count": delegation_count,
                "iteration_count": result.get("iteration_count", 0),
                "error": None,
            }
        except Exception as e:
            return {
                "response": "",
                "todo_count": 0,
                "file_count": 0,
                "delegation_count": 0,
                "iteration_count": 0,
                "error": str(e),
            }
    
    def judge_output(self, query: str, output: str) -> dict:
        """Use LLM-as-a-judge to score the output."""
        if not output or len(output.strip()) < 50:
            return {
                "completeness": 1, "structure": 1, "accuracy": 1, "depth": 1,
                "plan_used": False, "delegation_used": False, "files_created": False,
                "overall_rating": "poor", "feedback": "Output was empty or too short."
            }
        
        try:
            judge_input = JUDGE_PROMPT.format(query=query, output=output[:3000])
            response = self.judge_llm.invoke(judge_input)
            
            # Parse JSON from response
            import re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                return scores
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️ Judge error: {e}")
        
        return {
            "completeness": 2, "structure": 2, "accuracy": 2, "depth": 2,
            "plan_used": False, "delegation_used": False, "files_created": False,
            "overall_rating": "fair", "feedback": "Judge could not parse output."
        }
    
    def run_evaluation(self, count: int = None):
        """Run the full evaluation suite."""
        self.setup()
        prompts = self.load_prompts(count)
        
        print("=" * 60)
        print("🧪 STARTING EVALUATION")
        print("=" * 60)
        
        for i, prompt_data in enumerate(prompts):
            prompt_id = prompt_data["id"]
            category = prompt_data["category"]
            prompt = prompt_data["prompt"]
            
            print(f"\n{'─' * 60}")
            print(f"📝 Test {i+1}/{len(prompts)} [#{prompt_id}] ({category})")
            print(f"   Prompt: {prompt[:80]}...")
            print(f"{'─' * 60}")
            
            # Run agent
            print("   🤖 Running agent...")
            import time
            start_time = time.time()
            agent_result = self.run_agent(prompt)
            elapsed = time.time() - start_time
            
            if agent_result["error"]:
                print(f"   ❌ Agent error: {agent_result['error'][:100]}")
                self.results.append({
                    "id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "status": "error",
                    "error": agent_result["error"],
                    "scores": None,
                    "elapsed_seconds": elapsed,
                })
                # Wait before next to avoid rate limits
                print("   ⏳ Waiting 30s before next test...")
                time.sleep(30)
                continue
            
            print(f"   ✅ Agent completed in {elapsed:.0f}s")
            print(f"      TODOs: {agent_result['todo_count']} | Files: {agent_result['file_count']} | Delegations: {agent_result['delegation_count']}")
            
            if self.verbose:
                print(f"      Response preview: {agent_result['response'][:200]}...")
            
            # Judge output
            print("   ⚖️ Judging output...")
            
            # Wait a bit before judge call to avoid rate limits
            time.sleep(10)
            
            scores = self.judge_output(prompt, agent_result["response"])
            avg_score = (scores["completeness"] + scores["structure"] + scores["accuracy"] + scores["depth"]) / 4
            
            rating = scores.get("overall_rating", "unknown")
            emoji = {"excellent": "🌟", "good": "✅", "fair": "⚠️", "poor": "❌"}.get(rating, "❓")
            
            print(f"   {emoji} Rating: {rating.upper()} (avg: {avg_score:.1f}/4)")
            print(f"      Completeness: {scores['completeness']}/4 | Structure: {scores['structure']}/4")
            print(f"      Accuracy: {scores['accuracy']}/4 | Depth: {scores['depth']}/4")
            print(f"      Feedback: {scores.get('feedback', 'N/A')}")
            
            self.results.append({
                "id": prompt_id,
                "category": category,
                "prompt": prompt,
                "status": "completed",
                "scores": scores,
                "agent_metadata": {
                    "todo_count": agent_result["todo_count"],
                    "file_count": agent_result["file_count"],
                    "delegation_count": agent_result["delegation_count"],
                    "iteration_count": agent_result["iteration_count"],
                },
                "elapsed_seconds": elapsed,
            })
            
            # Wait between tests to avoid rate limits
            if i < len(prompts) - 1:
                print("   ⏳ Waiting 30s before next test...")
                time.sleep(30)
        
        # Generate summary
        self._print_summary()
        self._save_results()
    
    def _print_summary(self):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        completed = [r for r in self.results if r["status"] == "completed"]
        errors = [r for r in self.results if r["status"] == "error"]
        
        print(f"\nTotal tests: {len(self.results)}")
        print(f"Completed: {len(completed)}")
        print(f"Errors: {len(errors)}")
        
        if not completed:
            print("\n❌ No tests completed successfully.")
            return
        
        # Score distribution
        ratings = {}
        for r in completed:
            rating = r["scores"].get("overall_rating", "unknown")
            ratings[rating] = ratings.get(rating, 0) + 1
        
        print(f"\n📈 Rating Distribution:")
        for rating in ["excellent", "good", "fair", "poor"]:
            count = ratings.get(rating, 0)
            bar = "█" * count
            print(f"   {rating:>10}: {bar} ({count})")
        
        # Pass rate (good or excellent)
        pass_count = ratings.get("excellent", 0) + ratings.get("good", 0)
        pass_rate = (pass_count / len(completed)) * 100 if completed else 0
        
        print(f"\n🎯 Pass Rate (good+excellent): {pass_rate:.0f}% ({pass_count}/{len(completed)})")
        
        target = 70
        if pass_rate >= target:
            print(f"   ✅ PASSED — Meets {target}% target!")
        else:
            print(f"   ❌ FAILED — Below {target}% target ({pass_rate:.0f}%)")
        
        # Average scores
        avg_completeness = sum(r["scores"]["completeness"] for r in completed) / len(completed)
        avg_structure = sum(r["scores"]["structure"] for r in completed) / len(completed)
        avg_accuracy = sum(r["scores"]["accuracy"] for r in completed) / len(completed)
        avg_depth = sum(r["scores"]["depth"] for r in completed) / len(completed)
        
        print(f"\n📊 Average Scores:")
        print(f"   Completeness: {avg_completeness:.1f}/4")
        print(f"   Structure:    {avg_structure:.1f}/4")
        print(f"   Accuracy:     {avg_accuracy:.1f}/4")
        print(f"   Depth:        {avg_depth:.1f}/4")
        
        # Agent features
        plan_count = sum(1 for r in completed if r["scores"].get("plan_used"))
        deleg_count = sum(1 for r in completed if r["scores"].get("delegation_used"))
        files_count = sum(1 for r in completed if r["scores"].get("files_created"))
        
        print(f"\n🔧 Agent Feature Usage:")
        print(f"   Plan used:       {plan_count}/{len(completed)}")
        print(f"   Delegation used: {deleg_count}/{len(completed)}")
        print(f"   Files created:   {files_count}/{len(completed)}")
        
        print("\n" + "=" * 60)
    
    def _save_results(self):
        """Save results to JSON file."""
        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"eval_{timestamp}.json"
        
        output = {
            "timestamp": timestamp,
            "model": LLM_MODEL,
            "total_tests": len(self.results),
            "completed": len([r for r in self.results if r["status"] == "completed"]),
            "results": self.results,
        }
        
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation Suite")
    parser.add_argument("--count", type=int, default=None, help="Number of prompts to test (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(verbose=args.verbose)
    runner.run_evaluation(count=args.count)
