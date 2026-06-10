"""
Memory System Module — Autonomous Cognitive Engine.
=====================================================
Provides virtual file system tools for context offloading:
ls, read_file, write_file, edit_file.
Re-exports from src.memory.virtual_fs and src.tools.filesystem.
"""

from src.memory.virtual_fs import VirtualFileSystem
from src.tools.filesystem import (
    write_file,
    read_file,
    edit_file,
    ls,
    process_filesystem_tool_result,
)

__all__ = [
    "VirtualFileSystem",
    "write_file",
    "read_file",
    "edit_file",
    "ls",
    "process_filesystem_tool_result",
]
