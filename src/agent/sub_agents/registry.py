"""
Sub-Agent Registry and Execution Engine for the Autonomous Cognitive Engine.
=============================================================================
Defines specialized sub-agents (search_agent, summary_agent, coding_agent) 
and a centralized ReAct execution loop using Groq (ChatGroq).
Includes robust error recovery for Groq's tool calling issues.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
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
from src.tools.search import web_search
from src.tools.filesystem import (
    write_file,
    read_file,
    edit_file,
    ls,
    process_filesystem_tool_result,
)


# ──────────────────────────────────────────────
# Specialized Sub-Agent Prompts
# ──────────────────────────────────────────────

SEARCH_AGENT_PROMPT = """You are the Search Agent — a specialized worker for web research.
Your goal is to search the web, collect facts, and produce a structured summary.

## TOOLS
- `web_search(query)` — Search the web. The `query` parameter must be a string.

## IMPORTANT RULES
- Call web_search with a SINGLE string argument: web_search(query="your search query here")
- Do NOT pass extra arguments to web_search.
- Use multiple specific search queries to gather comprehensive data.
- After gathering enough info, produce a final structured summary as plain text (no tool calls).
"""

SUMMARY_AGENT_PROMPT = """You are the Summary Agent — a specialized worker for synthesizing research.
Your goal is to read saved research files and produce high-quality structured reports.

## TOOLS
- `ls(directory)` — List files. Pass directory as a string, e.g. ls(directory="research")
- `read_file(filepath)` — Read a file. Pass filepath as a string.
- `write_file(filepath, content)` — Save a file. Pass filepath and content as strings.
- `edit_file(filepath, old_content, new_content)` — Edit a file.

## IMPORTANT RULES
- Read ALL available research files before writing your summary.
- Produce well-structured markdown reports.
- Save your final report to a file.
"""

CODING_AGENT_PROMPT = """You are the Coding Agent — a specialized worker for writing small, low-risk code.
Your goal is to write scripts, data processing code, formatting utilities, or code snippets.

## TOOLS
- `write_file(filepath, content)` — Save code files.
- `read_file(filepath)` — Read existing files.
- `edit_file(filepath, old_content, new_content)` — Modify existing code.
- `ls(directory)` — List available files.

## IMPORTANT RULES
- Write clean, well-commented Python code.
- Keep scripts small and focused.
- Save all code files with appropriate extensions (.py, .js, etc.).
"""


# ──────────────────────────────────────────────
# Sub-Agent Configuration Registry
# ──────────────────────────────────────────────
SEARCH_TOOLS = [web_search]
SUMMARY_TOOLS = [write_file, read_file, edit_file, ls]
CODING_TOOLS = [write_file, read_file, edit_file, ls]

SUB_AGENTS_CONFIG = {
    "search_agent": {
        "prompt": SEARCH_AGENT_PROMPT,
        "tools": SEARCH_TOOLS,
        "description": "Web search, information retrieval, and fact collection.",
    },
    "summary_agent": {
        "prompt": SUMMARY_AGENT_PROMPT,
        "tools": SUMMARY_TOOLS,
        "description": "Reading VFS files, synthesizing notes, and writing reports.",
    },
    "coding_agent": {
        "prompt": CODING_AGENT_PROMPT,
        "tools": CODING_TOOLS,
        "description": "Writing small scripts, data processing, code snippets.",
    },
}


# ──────────────────────────────────────────────
# Groq XML Parser — handles ALL Groq tool errors
# ──────────────────────────────────────────────
def _extract_failed_generation(error_str: str) -> str | None:
    """Extract the 'failed_generation' value from a Groq error string."""
    # Method 1: Regex to extract failed_generation value (single quotes)
    fg_pattern = r"'failed_generation'\s*:\s*'(.*?)'\s*\}"
    match = re.search(fg_pattern, error_str, re.DOTALL)
    if match:
        return match.group(1)
    # Method 2: Double quotes
    fg_pattern2 = r'"failed_generation"\s*:\s*"(.*?)"\s*\}'
    match2 = re.search(fg_pattern2, error_str, re.DOTALL)
    if match2:
        return match2.group(1)
    # Method 3: Direct search
    if "<function=" in error_str:
        return error_str
    return None


def _parse_groq_tool_call(error_str: str) -> list[dict] | None:
    """Parse Groq's failed_generation XML into proper tool calls."""
    xml_str = _extract_failed_generation(error_str)
    if not xml_str:
        return None
    
    patterns = [
        r'<function=(\w+)>\s*(\[[\s\S]*?\])\s*</function>',
        r'<function=(\w+)>\s*(\{[\s\S]*?\})\s*</function>',
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
            args_str = args_str.strip().replace("\\'", "'").replace("\\n", "\n")
            parsed_args = json.loads(args_str)
            if isinstance(parsed_args, list) and func_name == "write_todos":
                tool_calls.append({"name": func_name, "args": {"todos": parsed_args}, "id": f"sub_parsed_{i}_{func_name}"})
            elif isinstance(parsed_args, dict):
                tool_calls.append({"name": func_name, "args": parsed_args, "id": f"sub_parsed_{i}_{func_name}"})
            else:
                tool_calls.append({"name": func_name, "args": {"items": parsed_args}, "id": f"sub_parsed_{i}_{func_name}"})
        except json.JSONDecodeError:
            continue
    return tool_calls if tool_calls else None


def _invoke_groq_safely(
    messages: list[BaseMessage],
    tools: Sequence[Any],
    max_retries: int = 5,
) -> AIMessage:
    """
    Invoke Groq with comprehensive error recovery:
    1. tool_use_failed → parse XML
    2. tool call validation failed → add correction message and retry
    3. rate limit → wait and retry
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            llm = ChatGroq(
                model=LLM_MODEL,
                api_key=GROQ_API_KEY,
                temperature=LLM_TEMPERATURE,
            )
            if tools:
                llm = llm.bind_tools(tools)
            return llm.invoke(messages)
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Case 1: tool_use_failed — parse XML
            if "tool_use_failed" in error_str or "failed_generation" in error_str:
                parsed = _parse_groq_tool_call(error_str)
                if parsed:
                    return AIMessage(content="", tool_calls=parsed)
            
            # Case 2: tool call validation failed — retry WITHOUT tools
            if "tool call validation" in error_str.lower():
                print(f"      🔧 Tool validation error, retrying without tools...")
                try:
                    llm_no_tools = ChatGroq(
                        model=LLM_MODEL,
                        api_key=GROQ_API_KEY,
                        temperature=LLM_TEMPERATURE,
                    )
                    # Add a correction hint
                    correction = HumanMessage(
                        content="Your previous tool call had invalid format. "
                        "Please respond with plain text instead, providing the information directly."
                    )
                    return llm_no_tools.invoke(messages + [correction])
                except Exception:
                    pass
            
            # Case 3: rate limit — wait longer for Groq free tier
            if "429" in error_str or "rate" in error_str.lower():
                wait = 30 + (attempt * 20)
                print(f"      ⏳ Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            
            # Other error — retry with small delay
            time.sleep(2)
            continue
    
    raise Exception(f"Sub-agent LLM failed after {max_retries} retries: {str(last_error)[:200]}")


# ──────────────────────────────────────────────
# Sub-Agent Execution Loop
# ──────────────────────────────────────────────
def run_sub_agent(
    agent_name: str,
    task_prompt: str,
    virtual_files: dict[str, str],
    max_iterations: int = 5,
) -> tuple[str, dict[str, str]]:
    """
    Spins up a specialized sub-agent and runs its isolated ReAct loop.
    
    Returns:
        Tuple of (text_response, updated_virtual_files).
    """
    if agent_name not in SUB_AGENTS_CONFIG:
        return f"❌ Sub-agent '{agent_name}' not registered. Available: {list(SUB_AGENTS_CONFIG.keys())}", virtual_files

    cfg = SUB_AGENTS_CONFIG[agent_name]
    system_prompt = cfg["prompt"]
    tools = cfg["tools"]
    tool_map = {t.name: t for t in tools}

    # Isolated VFS copy
    local_files = dict(virtual_files)

    # Inject VFS listing for summary/coding agents
    full_system_prompt = system_prompt
    if agent_name in ("summary_agent", "coding_agent") and local_files:
        file_list = "\n".join(f"  📄 `{path}` ({len(content)} chars)" for path, content in sorted(local_files.items()))
        full_system_prompt += f"\n📂 **Available files:**\n{file_list}\n"

    messages: list[BaseMessage] = [
        SystemMessage(content=full_system_prompt),
        HumanMessage(content=task_prompt),
    ]

    print(f"\n🚀 [Sub-Agent: {agent_name}] Task: {task_prompt[:100]}...")

    consecutive_errors = 0

    for iteration in range(1, max_iterations + 1):
        print(f"   🤖 [{agent_name}] Iteration {iteration}/{max_iterations}")

        try:
            response = _invoke_groq_safely(messages, tools)
            consecutive_errors = 0  # Reset on success
        except Exception as e:
            consecutive_errors += 1
            print(f"   ⚠️ [{agent_name}] Error #{consecutive_errors}: {str(e)[:100]}")
            
            if consecutive_errors >= 2:
                # After 2 consecutive errors, ask model to respond without tools
                print(f"   🔄 [{agent_name}] Switching to no-tool mode...")
                try:
                    llm_plain = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=LLM_TEMPERATURE)
                    fallback_msg = HumanMessage(
                        content=f"Tool calling is unavailable. Based on what you know, "
                        f"please answer this directly in plain text:\n\n{task_prompt}"
                    )
                    response = llm_plain.invoke([SystemMessage(content=system_prompt), fallback_msg])
                    return response.content, local_files
                except Exception:
                    pass
                break
            continue

        messages.append(response)

        # No tool calls → sub-agent is done
        if not response.tool_calls:
            print(f"   ✅ [{agent_name}] Done.")
            return response.content, local_files

        # Execute tool calls
        print(f"   🛠️ [{agent_name}] Tools: {[tc['name'] for tc in response.tool_calls]}")
        for tool_call in response.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]

            if t_name in tool_map:
                try:
                    tool_result = tool_map[t_name].invoke(t_args)

                    # Intercept filesystem tools to update local VFS
                    if t_name in {"write_file", "read_file", "edit_file", "ls"}:
                        fs_updates, display_res = process_filesystem_tool_result(
                            t_name, tool_result, {"files": local_files}
                        )
                        if "files" in fs_updates:
                            local_files.update(fs_updates["files"])
                        tool_result = display_res

                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=t_id))
                except Exception as e:
                    messages.append(
                        ToolMessage(content=f"❌ Tool error ({t_name}): {str(e)}", tool_call_id=t_id)
                    )
            else:
                messages.append(
                    ToolMessage(content=f"❌ Unknown tool: {t_name}", tool_call_id=t_id)
                )

    # Max iterations reached — get whatever text we have
    print(f"   ⚠️ [{agent_name}] Max iterations ({max_iterations}) reached.")
    last_content = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        "Sub-agent completed but no final text was produced.",
    )
    return last_content, local_files
