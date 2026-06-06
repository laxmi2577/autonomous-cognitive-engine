"""
Agent Node Functions for the Autonomous Cognitive Engine.
============================================================
Uses Groq (ChatGroq) with llama-3.3-70b-versatile.
Includes XML parser for Groq's tool_use_failed errors.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from src.agent.state import AgentState, TodoItem
from src.agent.prompts import MAIN_AGENT_SYSTEM_PROMPT
from src.tools.planning import (
    write_todos,
    update_todo,
    get_todos,
    process_planning_tool_result,
    format_todos_for_agent,
    create_fallback_plan,
)
from src.tools.filesystem import (
    write_file,
    read_file,
    edit_file,
    ls,
    process_filesystem_tool_result,
)
from src.tools.search import web_search
from src.tools.delegation import delegate_task, process_delegation_tool_result


# ──────────────────────────────────────────────
# All available tools
# ──────────────────────────────────────────────
ALL_TOOLS = [
    write_todos,
    update_todo,
    get_todos,
    write_file,
    read_file,
    edit_file,
    ls,
    web_search,
    delegate_task,
]
PLANNING_TOOL_NAMES = {"write_todos", "update_todo", "get_todos"}
FILESYSTEM_TOOL_NAMES = {"write_file", "read_file", "edit_file", "ls"}


def _extract_failed_generation(error_str: str) -> str | None:
    """Extract the 'failed_generation' value from a Groq error string."""
    # Method 1: Try to extract JSON body from error string
    # Format: Error code: 400 - {'error': {..., 'failed_generation': '...'}}
    try:
        json_start = error_str.find(" - {")
        if json_start != -1:
            json_str = error_str[json_start + 3:]
            # Python dict format uses single quotes — convert to double quotes for JSON
            # But the inner values use double quotes, so we need to be careful
            # Just extract failed_generation with regex instead
            pass
    except Exception:
        pass
    
    # Method 2: Regex to extract failed_generation value
    fg_pattern = r"'failed_generation'\s*:\s*'(.*?)'\s*\}"
    match = re.search(fg_pattern, error_str, re.DOTALL)
    if match:
        return match.group(1)
    
    # Method 3: Try with double quotes
    fg_pattern2 = r'"failed_generation"\s*:\s*"(.*?)"\s*\}'
    match2 = re.search(fg_pattern2, error_str, re.DOTALL)
    if match2:
        return match2.group(1)
    
    # Method 4: Just search for <function= directly in the full string
    if "<function=" in error_str:
        return error_str
    
    return None


def _parse_groq_tool_call(error_str: str) -> list[dict] | None:
    """
    Parse Groq's failed_generation XML into proper tool calls.
    
    Step 1: Extract the failed_generation XML string from the error
    Step 2: Parse <function=name>[args]</function> patterns from it
    """
    # First, extract the clean XML from the error
    xml_str = _extract_failed_generation(error_str)
    if not xml_str:
        return None
    
    # Now parse <function=name>ARGS</function> patterns
    patterns = [
        r'<function=(\w+)>\s*(\[[\s\S]*?\])\s*</function>',   # [...] args
        r'<function=(\w+)>\s*(\{[\s\S]*?\})\s*</function>',   # {...} args
    ]
    
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, xml_str)
        matches.extend(found)
    
    if not matches:
        return None
    
    tool_calls = []
    for i, (func_name, args_str) in enumerate(matches):
        try:
            args_str = args_str.strip()
            # Unescape any escaped quotes from the error string
            args_str = args_str.replace("\\'", "'").replace("\\n", "\n")
            parsed_args = json.loads(args_str)
            
            # write_todos sends a list → wrap in {"todos": [...]}
            if isinstance(parsed_args, list) and func_name == "write_todos":
                tool_calls.append({
                    "name": func_name,
                    "args": {"todos": parsed_args},
                    "id": f"groq_parsed_{i}_{func_name}",
                })
            elif isinstance(parsed_args, list):
                tool_calls.append({
                    "name": func_name,
                    "args": parsed_args[0] if len(parsed_args) == 1 and isinstance(parsed_args[0], dict) else {"items": parsed_args},
                    "id": f"groq_parsed_{i}_{func_name}",
                })
            else:
                tool_calls.append({
                    "name": func_name,
                    "args": parsed_args,
                    "id": f"groq_parsed_{i}_{func_name}",
                })
        except json.JSONDecodeError as je:
            print(f"      JSON parse error for {func_name}: {je}")
            continue
    
    return tool_calls if tool_calls else None


# ──────────────────────────────────────────────
# Create LLM
# ──────────────────────────────────────────────
def _create_llm():
    """Create a ChatGroq LLM instance with tool binding."""
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    return llm.bind_tools(ALL_TOOLS)


# ──────────────────────────────────────────────
# Agent Node — with Groq XML error recovery
# ──────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """
    The main agent reasoning node.
    If Groq returns tool_use_failed, parses the XML and creates proper tool calls.
    """
    # Build context
    context_parts = [MAIN_AGENT_SYSTEM_PROMPT]

    todos = state.get("todos", [])
    if todos:
        context_parts.append("\n---\n" + format_todos_for_agent(todos))

    files = state.get("files", {})
    if files:
        file_list = "\n".join(f"  📄 `{path}` ({len(content)} chars)" for path, content in sorted(files.items()))
        context_parts.append(f"\n---\n📂 **Files in Virtual File System:**\n{file_list}")

    system_message = SystemMessage(content="\n".join(context_parts))
    messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    full_messages = [system_message] + messages

    max_retries = 5
    last_error = None
    current_iteration = state.get("iteration_count", 0)

    # FALLBACK: If iteration >= 2 and still no todos, create plan programmatically
    if current_iteration >= 2 and not todos:
        user_query = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        if user_query:
            print("   🔄 No plan after 2 iterations — creating fallback plan...")
            fallback_todos = create_fallback_plan(user_query)
            print(f"   ✅ Fallback plan created: {len(fallback_todos)} tasks")
            # Continue with the LLM call but also inject todos
            todos = fallback_todos
            context_parts_with_plan = [MAIN_AGENT_SYSTEM_PROMPT]
            context_parts_with_plan.append("\n---\n" + format_todos_for_agent(todos))
            if files:
                file_list = "\n".join(f"  📄 `{path}` ({len(content)} chars)" for path, content in sorted(files.items()))
                context_parts_with_plan.append(f"\n---\n📂 **Files in Virtual File System:**\n{file_list}")
            system_message = SystemMessage(content="\n".join(context_parts_with_plan))
            full_messages = [system_message] + messages
            # Return the plan injection + let the agent continue
            try:
                llm = _create_llm()
                response = llm.invoke(full_messages)
                return {
                    "messages": [response],
                    "iteration_count": current_iteration + 1,
                    "todos": fallback_todos,
                    "plan_created": True,
                }
            except Exception:
                pass  # Fall through to normal retry logic

    for attempt in range(max_retries):
        try:
            llm = _create_llm()
            response = llm.invoke(full_messages)

            # SAFETY CHECK: If first iteration and agent didn't call write_todos, force planning
            if current_iteration == 0 and not todos:
                has_write_todos = False
                if response.tool_calls:
                    has_write_todos = any(tc["name"] == "write_todos" for tc in response.tool_calls)
                
                if not has_write_todos:
                    print("   ⚠️ Agent skipped write_todos — forcing plan creation...")
                    nudge = HumanMessage(
                        content="STOP. You MUST call write_todos FIRST before doing anything else. "
                        "Create a plan with 5 sub-tasks NOW using write_todos."
                    )
                    response = llm.invoke(full_messages + [nudge])

            new_iteration = current_iteration + 1
            return {
                "messages": [response],
                "iteration_count": new_iteration,
            }
        except Exception as e:
            last_error = e
            error_str = str(e)

            # Case 1: tool_use_failed — parse XML from error
            if "tool_use_failed" in error_str or "failed_generation" in error_str:
                print(f"   🔧 Fixing Groq tool format (attempt {attempt+1})...")
                parsed_calls = _parse_groq_tool_call(error_str)
                if parsed_calls:
                    print(f"   ✅ Parsed {len(parsed_calls)} tool call(s): {[c['name'] for c in parsed_calls]}")
                    ai_msg = AIMessage(content="", tool_calls=parsed_calls)
                    new_iteration = state.get("iteration_count", 0) + 1
                    return {
                        "messages": [ai_msg],
                        "iteration_count": new_iteration,
                    }
                else:
                    print(f"   ⚠️ XML parse FAILED. Error snippet: {error_str[:200]}")
                # If XML parse failed, retry
                time.sleep(2)
                continue

            # Case 2: tool call validation failed — retry without tools
            if "tool call validation" in error_str.lower():
                print(f"   🔧 Tool validation error, retrying without tools...")
                try:
                    llm_plain = ChatGroq(
                        model=LLM_MODEL,
                        api_key=GROQ_API_KEY,
                        temperature=LLM_TEMPERATURE,
                    )
                    correction = HumanMessage(
                        content="Your previous tool call had an invalid format. "
                        "Please respond with the next action or your findings in plain text."
                    )
                    response = llm_plain.invoke(full_messages + [correction])
                    new_iteration = state.get("iteration_count", 0) + 1
                    return {
                        "messages": [response],
                        "iteration_count": new_iteration,
                    }
                except Exception:
                    time.sleep(2)
                    continue

            # Case 3: rate limit — wait longer for Groq free tier
            if "429" in error_str or "rate" in error_str.lower():
                wait = 30 + (attempt * 20)
                print(f"   ⏳ Rate limited, waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
                continue

            # Other errors — retry
            print(f"   ⚠️ Error: {error_str[:100]}")
            time.sleep(2)
            continue

    raise Exception(f"Agent node failed after {max_retries} retries. Last error: {last_error}")


# ──────────────────────────────────────────────
# Tool Node — Execute tool calls
# ──────────────────────────────────────────────
def tool_node(state: AgentState) -> dict:
    """Execute tool calls from the agent's last message."""
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    tool_map = {t.name: t for t in ALL_TOOLS}
    tool_messages = []
    state_updates = {}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)

                if tool_name in PLANNING_TOOL_NAMES:
                    planning_updates = process_planning_tool_result(tool_name, result, state)
                    state_updates.update(planning_updates)
                    if tool_name == "get_todos":
                        todos = state_updates.get("todos", state.get("todos", []))
                        result = format_todos_for_agent(todos)

                elif tool_name in FILESYSTEM_TOOL_NAMES:
                    fs_updates, result = process_filesystem_tool_result(
                        tool_name, result, {**state, **state_updates}
                    )
                    state_updates.update(fs_updates)

                elif tool_name == "delegate_task":
                    delegation_updates, result = process_delegation_tool_result(
                        tool_name, result, {**state, **state_updates}
                    )
                    state_updates.update(delegation_updates)

                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            except Exception as e:
                tool_messages.append(
                    ToolMessage(content=f"❌ Tool error ({tool_name}): {str(e)}", tool_call_id=tool_id)
                )
        else:
            tool_messages.append(
                ToolMessage(content=f"❌ Unknown tool: {tool_name}", tool_call_id=tool_id)
            )

    return {"messages": tool_messages, **state_updates}


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to continue (call tools) or stop."""
    last_message = state["messages"][-1]

    from src.config import MAX_AGENT_ITERATIONS
    if state.get("iteration_count", 0) >= MAX_AGENT_ITERATIONS:
        _auto_complete_todos(state)
        return "end"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Agent is done — auto-complete any remaining todos
    _auto_complete_todos(state)
    return "end"


def _auto_complete_todos(state: AgentState) -> None:
    """Auto-mark all remaining todos as completed when agent finishes."""
    todos = state.get("todos", [])
    if not todos:
        return
    
    changed = False
    for todo in todos:
        if todo.status != "completed":
            todo.status = "completed"
            if not todo.result:
                todo.result = "Auto-completed at end of session"
            changed = True
    
    if changed:
        print("   ✅ Auto-completed remaining todos")
