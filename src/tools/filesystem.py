"""
Virtual File System Tools for the Autonomous Cognitive Engine.
================================================================
Implements ls, read_file, write_file, and edit_file tools that allow
the agent to manage context by storing and retrieving information
in a dictionary-based virtual file system within the LangGraph state.
"""

from __future__ import annotations

import json
from typing import Any
from langchain_core.tools import tool


@tool
def write_file(filepath: str, content: str) -> str:
    """Save content to a file in the virtual file system.
    
    Use this to store intermediate results, research findings, notes, 
    or any information you want to reference later. Use organized paths
    like 'research/topic.md' or 'notes/summary.md'.
    
    Args:
        filepath: Path for the file (e.g., 'research/quantum_computing.md').
        content: The text content to save.
    
    Returns:
        Confirmation message.
    """
    return json.dumps({
        "_action": "write_file",
        "filepath": filepath.strip("/"),
        "content": content,
        "display": f"📝 File saved: `{filepath}` ({len(content)} characters)"
    })


@tool
def read_file(filepath: str) -> str:
    """Read content from a file in the virtual file system.
    
    Use this to retrieve previously saved information, research notes,
    or intermediate results before synthesizing your final output.
    
    Args:
        filepath: Path of the file to read (e.g., 'research/quantum_computing.md').
    
    Returns:
        The file content, or an error if the file doesn't exist.
    """
    return json.dumps({
        "_action": "read_file",
        "filepath": filepath.strip("/"),
        "display": f"📖 Reading file: `{filepath}`..."
    })


@tool
def edit_file(filepath: str, old_content: str, new_content: str) -> str:
    """Edit a specific section of an existing file in the virtual file system.
    
    Use this to update or modify parts of a file without rewriting the entire content.
    The old_content will be replaced with new_content.
    
    Args:
        filepath: Path of the file to edit.
        old_content: The exact text to find and replace.
        new_content: The replacement text.
    
    Returns:
        Confirmation message, or an error if the file or content is not found.
    """
    return json.dumps({
        "_action": "edit_file",
        "filepath": filepath.strip("/"),
        "old_content": old_content,
        "new_content": new_content,
        "display": f"✏️ Editing file: `{filepath}`"
    })


@tool
def ls(directory: str = "") -> str:
    """List all files in the virtual file system.
    
    Use this to see what files you've already saved and what information
    is available. Optionally filter by directory prefix.
    
    Args:
        directory: Optional directory prefix to filter results (e.g., 'research/').
    
    Returns:
        List of files in the virtual file system.
    """
    return json.dumps({
        "_action": "ls",
        "directory": directory.strip("/"),
        "display": f"📂 Listing files" + (f" in `{directory}`" if directory else "") + "..."
    })


# ──────────────────────────────────────────────
# Helper: Process file system tool results
# ──────────────────────────────────────────────
def process_filesystem_tool_result(tool_name: str, tool_result: str, current_state: dict) -> tuple[dict, str]:
    """
    Process the result of a filesystem tool call.
    
    Returns:
        Tuple of (state_updates_dict, response_message_for_agent)
    """
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return {}, tool_result
    
    action = data.get("_action", "")
    files = dict(current_state.get("files", {}))
    state_updates = {}
    response = data.get("display", "")
    
    if action == "write_file":
        filepath = data["filepath"]
        content = data["content"]
        files[filepath] = content
        state_updates["files"] = files
        response = f"📝 File saved: `{filepath}` ({len(content)} characters)"
        
    elif action == "read_file":
        filepath = data["filepath"]
        if filepath in files:
            content = files[filepath]
            response = f"📖 **File: `{filepath}`**\n\n{content}"
        else:
            # Try partial match
            matches = [f for f in files if filepath in f]
            if matches:
                response = f"❌ File `{filepath}` not found. Did you mean one of these?\n"
                response += "\n".join(f"  • `{m}`" for m in matches)
            else:
                available = list(files.keys())
                response = f"❌ File `{filepath}` not found.\n"
                if available:
                    response += "Available files:\n" + "\n".join(f"  • `{f}`" for f in available)
                else:
                    response += "No files saved yet. Use write_file to save content."
                    
    elif action == "edit_file":
        filepath = data["filepath"]
        if filepath not in files:
            response = f"❌ File `{filepath}` not found. Cannot edit."
        else:
            old_content = data["old_content"]
            new_content = data["new_content"]
            current_content = files[filepath]
            if old_content in current_content:
                files[filepath] = current_content.replace(old_content, new_content, 1)
                state_updates["files"] = files
                response = f"✏️ File `{filepath}` edited successfully."
            else:
                response = f"❌ Could not find the specified text in `{filepath}`."
                
    elif action == "ls":
        directory = data.get("directory", "")
        if directory:
            matching_files = {k: v for k, v in files.items() if k.startswith(directory)}
        else:
            matching_files = files
            
        if matching_files:
            lines = ["📂 **Virtual File System:**\n"]
            for path, content in sorted(matching_files.items()):
                size = len(content)
                lines.append(f"  📄 `{path}` ({size} chars)")
            lines.append(f"\n📊 Total: {len(matching_files)} file(s)")
            response = "\n".join(lines)
        else:
            response = "📂 Virtual file system is empty. No files saved yet."
    
    return state_updates, response
