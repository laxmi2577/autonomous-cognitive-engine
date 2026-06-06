"""
Delegation Tool for the Autonomous Cognitive Engine.
======================================================
Implements the delegate_task tool allowing the main supervisor agent to invoke
specialized sub-agents and merge their memory/files back into the main state.
Includes rate-limit cooldown between delegations for Groq free tier.
"""

from __future__ import annotations

import json
import time
from typing import Any
from langchain_core.tools import tool

from src.agent.sub_agents.registry import run_sub_agent

# Track delegation count for cooldown
_delegation_count = 0


@tool
def delegate_task(agent_name: str, task_prompt: str) -> str:
    """Delegate a specific sub-task to a specialized sub-agent.
    
    Use this tool to delegate work to specialized workers:
    1. 'search_agent': Web searches, collecting raw facts, research data.
    2. 'summary_agent': Reading VFS files, synthesizing notes, drafting reports.
    3. 'coding_agent': Writing Python scripts, data processing, code snippets.
    
    Args:
        agent_name: The sub-agent name ('search_agent', 'summary_agent', or 'coding_agent').
        task_prompt: Detailed instructions for the sub-agent.
                     
    Returns:
        JSON string representing the delegation request.
    """
    return json.dumps({
        "_action": "delegate_task",
        "agent_name": agent_name,
        "task_prompt": task_prompt,
        "display": f"🤝 Delegating task to `{agent_name}`: {task_prompt}"
    })


def process_delegation_tool_result(
    tool_name: str,
    tool_result: str,
    current_state: dict
) -> tuple[dict, str]:
    """
    Process the result of a delegation tool call by running the sub-agent.
    
    Includes a cooldown between delegations to avoid Groq rate limits.
    After each delegation, waits 20 seconds to let the rate limit window reset.
    """
    global _delegation_count
    
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return {}, tool_result

    action = data.get("_action", "")
    if action != "delegate_task":
        return {}, tool_result

    agent_name = data["agent_name"]
    task_prompt = data["task_prompt"]
    
    # Rate limit cooldown — wait between delegations to avoid Groq 429 errors
    if _delegation_count > 0:
        cooldown = 20  # seconds between delegations
        print(f"   ⏳ Cooldown {cooldown}s before next delegation (avoiding rate limits)...")
        time.sleep(cooldown)
    
    _delegation_count += 1
    
    # Run the sub-agent with a copy of current virtual files
    virtual_files = current_state.get("files", {})
    
    try:
        response, updated_files = run_sub_agent(
            agent_name=agent_name,
            task_prompt=task_prompt,
            virtual_files=virtual_files
        )
        
        # Merge sub-agent files INTO existing files (not replace)
        merged_files = {**virtual_files, **updated_files}
        state_updates = {"files": merged_files}
        formatted_response = f"🤝 **Sub-Agent [{agent_name}] Response:**\n\n{response}"
        return state_updates, formatted_response
        
    except Exception as e:
        return {}, f"❌ Sub-agent delegation failed: {str(e)}"


def reset_delegation_counter():
    """Reset the delegation counter (call at start of new session)."""
    global _delegation_count
    _delegation_count = 0
