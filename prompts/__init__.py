"""
Prompts Module — Autonomous Cognitive Engine.
===============================================
Provides all system prompts used by the supervisor agent and sub-agents.
Re-exports from src.agent.prompts and src.agent.sub_agents.registry.
"""

from src.agent.prompts import MAIN_AGENT_SYSTEM_PROMPT, PLANNING_INSTRUCTIONS

__all__ = [
    "MAIN_AGENT_SYSTEM_PROMPT",
    "PLANNING_INSTRUCTIONS",
]
