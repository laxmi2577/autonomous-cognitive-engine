"""
Agent State Definition for the Autonomous Cognitive Engine.
============================================================
Defines the central state schema used by LangGraph to track:
- Conversation messages
- TODO task plans
- Virtual file system contents
- Agent metadata
"""

from __future__ import annotations

import operator
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ──────────────────────────────────────────────
# TODO Item Model
# ──────────────────────────────────────────────
class TodoItem(BaseModel):
    """Represents a single task in the agent's plan."""
    id: int = Field(description="Unique task ID (1-indexed)")
    task: str = Field(description="Description of what needs to be done")
    status: str = Field(
        default="pending",
        description="Task status: 'pending', 'in_progress', or 'completed'"
    )
    result: str = Field(
        default="",
        description="Result or summary after task completion"
    )


# ──────────────────────────────────────────────
# Agent State (LangGraph State Schema)
# ──────────────────────────────────────────────
class AgentState(MessagesState):
    """
    Central state for the Autonomous Cognitive Engine.
    
    Extends MessagesState (which provides `messages` with add-only semantics)
    and adds custom fields for planning and context management.
    """

    # ── Planning State ──
    todos: list[TodoItem] = Field(
        default_factory=list,
        description="The agent's current task plan — a list of TODO items"
    )
    current_todo_id: int = Field(
        default=0,
        description="ID of the TODO currently being executed (0 = none)"
    )

    # ── Virtual File System State ──
    files: dict[str, str] = Field(
        default_factory=dict,
        description="Virtual file system: {filepath: content} mapping"
    )

    # ── Agent Metadata ──
    iteration_count: int = Field(
        default=0,
        description="Number of agent loop iterations completed"
    )
    plan_created: bool = Field(
        default=False,
        description="Whether the agent has created an initial plan"
    )
