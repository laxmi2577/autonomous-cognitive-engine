"""
Virtual File System implementation for the Autonomous Cognitive Engine.
========================================================================
Provides a dictionary-based in-memory file system that lives within
the LangGraph state, enabling context offloading and persistence.
"""

from __future__ import annotations

from typing import Any


class VirtualFileSystem:
    """
    A virtual file system that stores files as key-value pairs in memory.
    
    This is used to give the agent a "scratchpad" where it can save
    intermediate results, research findings, and notes — enabling it
    to handle tasks that exceed its context window.
    """
    
    def __init__(self, initial_files: dict[str, str] | None = None):
        """Initialize the virtual file system.
        
        Args:
            initial_files: Optional dict of {filepath: content} to preload.
        """
        self._files: dict[str, str] = dict(initial_files or {})
    
    def write(self, filepath: str, content: str) -> str:
        """Write content to a file path.
        
        Args:
            filepath: The virtual file path (e.g., 'research/topic.md').
            content: The text content to save.
        
        Returns:
            Confirmation message.
        """
        filepath = self._normalize_path(filepath)
        is_new = filepath not in self._files
        self._files[filepath] = content
        action = "Created" if is_new else "Updated"
        return f"📝 {action}: `{filepath}` ({len(content)} chars)"
    
    def read(self, filepath: str) -> str:
        """Read content from a file path.
        
        Args:
            filepath: The virtual file path to read.
        
        Returns:
            The file content, or an error message.
        """
        filepath = self._normalize_path(filepath)
        if filepath in self._files:
            return self._files[filepath]
        
        # Try fuzzy match
        matches = [f for f in self._files if filepath.lower() in f.lower()]
        if matches:
            suggestion = ", ".join(f"`{m}`" for m in matches[:3])
            return f"❌ File not found: `{filepath}`. Did you mean: {suggestion}?"
        
        return f"❌ File not found: `{filepath}`. Use `ls` to see available files."
    
    def edit(self, filepath: str, old_text: str, new_text: str) -> str:
        """Edit a specific section of a file.
        
        Args:
            filepath: The file to edit.
            old_text: The exact text to find and replace.
            new_text: The replacement text.
        
        Returns:
            Confirmation or error message.
        """
        filepath = self._normalize_path(filepath)
        if filepath not in self._files:
            return f"❌ File not found: `{filepath}`"
        
        content = self._files[filepath]
        if old_text not in content:
            return f"❌ Text not found in `{filepath}`. Check the exact text."
        
        self._files[filepath] = content.replace(old_text, new_text, 1)
        return f"✏️ Edited: `{filepath}` successfully"
    
    def list_files(self, directory: str = "") -> list[dict[str, Any]]:
        """List all files, optionally filtered by directory.
        
        Args:
            directory: Optional directory prefix to filter by.
        
        Returns:
            List of file info dicts with path and size.
        """
        directory = self._normalize_path(directory)
        result = []
        for path, content in sorted(self._files.items()):
            if not directory or path.startswith(directory):
                result.append({
                    "path": path,
                    "size": len(content),
                    "preview": content[:100] + "..." if len(content) > 100 else content
                })
        return result
    
    def exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        return self._normalize_path(filepath) in self._files
    
    def delete(self, filepath: str) -> str:
        """Delete a file.
        
        Args:
            filepath: The file to delete.
        
        Returns:
            Confirmation or error message.
        """
        filepath = self._normalize_path(filepath)
        if filepath in self._files:
            del self._files[filepath]
            return f"🗑️ Deleted: `{filepath}`"
        return f"❌ File not found: `{filepath}`"
    
    def to_dict(self) -> dict[str, str]:
        """Export the file system as a dictionary (for state serialization)."""
        return dict(self._files)
    
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a file path — strip slashes, standardize separators."""
        return path.strip("/").strip("\\").replace("\\", "/")
    
    def __len__(self) -> int:
        return len(self._files)
    
    def __repr__(self) -> str:
        return f"VirtualFileSystem({len(self._files)} files)"
