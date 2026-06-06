"""
Planning Tools for the Autonomous Cognitive Engine.
=====================================================
Implements the write_todos, update_todo, and get_todos tools that allow
the agent to decompose complex tasks into structured sub-task plans.
"""

from __future__ import annotations

import json
from typing import Any
from langchain_core.tools import tool
from src.agent.state import TodoItem


@tool
def write_todos(todos: list[dict[str, str]]) -> str:
    """Create a structured task plan by writing a list of TODO items.
    
    Use this tool at the START of any complex task to break it down into 
    manageable sub-tasks. Each todo should have a 'task' field describing 
    what needs to be done.
    
    Args:
        todos: List of dicts, each with a 'task' key describing the sub-task.
               Example: [{"task": "Search for X"}, {"task": "Analyze Y"}]
    
    Returns:
        Confirmation message with the created plan.
    """
    # This function returns the data; the actual state update happens in the graph node
    todo_items = []
    for i, item in enumerate(todos, 1):
        task_desc = item.get("task", item.get("description", str(item)))
        todo_items.append({
            "id": i,
            "task": task_desc,
            "status": "pending",
            "result": ""
        })
    
    # Format the plan for display
    plan_lines = ["📋 **Task Plan Created:**\n"]
    for t in todo_items:
        plan_lines.append(f"  {t['id']}. ⬜ {t['task']}")
    plan_lines.append(f"\n📊 Total tasks: {len(todo_items)}")
    
    # Return serialized data so the graph can update state
    return json.dumps({
        "_action": "write_todos",
        "todos": todo_items,
        "display": "\n".join(plan_lines)
    })


@tool
def update_todo(todo_id: int, status: str, result: str = "") -> str:
    """Update the status of a specific TODO task.
    
    Use this tool to mark a task as 'in_progress' when you start working on it,
    or 'completed' when you finish it (include the result summary).
    
    Args:
        todo_id: The ID of the TODO to update (1-indexed).
        status: New status — must be 'in_progress' or 'completed'.
        result: Brief summary of the result (required when status is 'completed').
    
    Returns:
        Confirmation message with updated status.
    """
    if status not in ("in_progress", "completed", "pending"):
        return f"❌ Invalid status '{status}'. Must be 'pending', 'in_progress', or 'completed'."
    
    status_emoji = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
    emoji = status_emoji.get(status, "❓")
    
    return json.dumps({
        "_action": "update_todo",
        "todo_id": todo_id,
        "status": status,
        "result": result,
        "display": f"{emoji} Task #{todo_id} → **{status}**" + (f" | Result: {result}" if result else "")
    })


@tool
def get_todos() -> str:
    """Get the current task plan with all TODO items and their statuses.
    
    Use this tool to review your progress and decide what to work on next.
    
    Returns:
        Formatted list of all TODO items with their current statuses.
    """
    # The actual todo data will be injected by the graph node
    return json.dumps({
        "_action": "get_todos",
        "display": "📋 Requesting current task plan..."
    })


# ──────────────────────────────────────────────
# Helper: Process tool results and update state
# ──────────────────────────────────────────────
def process_planning_tool_result(tool_name: str, tool_result: str, current_state: dict) -> dict:
    """
    Process the result of a planning tool call and return state updates.
    
    This function is called by the graph node after a planning tool executes.
    It parses the JSON result and returns the state mutations needed.
    """
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return {}
    
    action = data.get("_action", "")
    state_updates = {}
    
    if action == "write_todos":
        # Create new TodoItem objects from the tool output
        state_updates["todos"] = [
            TodoItem(**item) for item in data["todos"]
        ]
        state_updates["plan_created"] = True
        
    elif action == "update_todo":
        # Update a specific todo in the existing list
        todo_id = data["todo_id"]
        new_status = data["status"]
        result = data.get("result", "")
        
        updated_todos = []
        for todo in current_state.get("todos", []):
            if todo.id == todo_id:
                updated = todo.model_copy()
                updated.status = new_status
                if result:
                    updated.result = result
                updated_todos.append(updated)
            else:
                updated_todos.append(todo)
        state_updates["todos"] = updated_todos
        
        # Track current active todo
        if new_status == "in_progress":
            state_updates["current_todo_id"] = todo_id
        elif new_status == "completed":
            state_updates["current_todo_id"] = 0
    
    elif action == "get_todos":
        # Format and return current todos (no state mutation needed)
        todos = current_state.get("todos", [])
        if not todos:
            # Override the tool message with actual data
            pass  # Will be handled in the node
    
    return state_updates


def format_todos_for_agent(todos: list[TodoItem]) -> str:
    """Format the TODO list for inclusion in agent context."""
    if not todos:
        return "📋 No task plan created yet. Use write_todos to create one."
    
    status_emoji = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
    lines = ["📋 **Current Task Plan:**\n"]
    
    completed = 0
    for todo in todos:
        emoji = status_emoji.get(todo.status, "❓")
        line = f"  {todo.id}. {emoji} {todo.task}"
        if todo.result:
            line += f"\n      └─ Result: {todo.result}"
        lines.append(line)
        if todo.status == "completed":
            completed += 1
    
    lines.append(f"\n📊 Progress: {completed}/{len(todos)} tasks completed")
    
    # Find next pending task
    next_pending = next((t for t in todos if t.status == "pending"), None)
    if next_pending:
        lines.append(f"👉 Next task: #{next_pending.id} — {next_pending.task}")
    elif completed == len(todos):
        lines.append("🎉 All tasks completed! Time to synthesize results.")
    
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Fallback: Programmatic plan creation
# ──────────────────────────────────────────────
def create_fallback_plan(user_query: str) -> list[TodoItem]:
    """
    Create a fallback plan when write_todos fails.
    Uses a simple LLM call (no tools) to break the query into 5 tasks.
    Falls back to a generic plan if even this fails.
    """
    try:
        from langchain_groq import ChatGroq
        from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
        
        llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
        
        response = llm.invoke(
            f"Break this user request into exactly 5 short task descriptions. "
            f"Return ONLY a JSON array of strings, nothing else.\n"
            f"Example: [\"Search for X\", \"Analyze Y\", \"Write script for Z\", \"Summarize findings\", \"Compile final report\"]\n\n"
            f"User request: {user_query}"
        )
        
        import re
        # Extract JSON array from response
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            tasks = json.loads(match.group())
            if isinstance(tasks, list) and len(tasks) >= 3:
                return [
                    TodoItem(id=i+1, task=str(t), status="pending", result="")
                    for i, t in enumerate(tasks[:5])
                ]
    except Exception as e:
        print(f"   ⚠️ Fallback plan LLM call failed: {e}")
    
    # Ultimate fallback — generic plan based on query
    return [
        TodoItem(id=1, task=f"Research: {user_query[:80]}", status="pending", result=""),
        TodoItem(id=2, task="Gather additional data and sources", status="pending", result=""),
        TodoItem(id=3, task="Write analysis script or processing code", status="pending", result=""),
        TodoItem(id=4, task="Synthesize findings into structured notes", status="pending", result=""),
        TodoItem(id=5, task="Compile comprehensive final report", status="pending", result=""),
    ]

