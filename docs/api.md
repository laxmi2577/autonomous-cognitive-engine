# API & Tool Reference

## Planning Tools

### `write_todos(todos)`
Create a structured task plan.

| Parameter | Type | Description |
|-----------|------|-------------|
| `todos` | `list[dict]` | List of dicts with `"task"` key |

**Example:**
```python
write_todos(todos=[
    {"task": "Research quantum computing"},
    {"task": "Analyze findings"},
    {"task": "Write comparison script"},
    {"task": "Draft summary report"},
    {"task": "Compile final output"}
])
```

**Returns:** JSON with `_action: "write_todos"` + todo items with IDs.

---

### `update_todo(todo_id, status, result)`
Update a specific task's status.

| Parameter | Type | Description |
|-----------|------|-------------|
| `todo_id` | `int` | Task ID (1-indexed) |
| `status` | `str` | `"pending"`, `"in_progress"`, or `"completed"` |
| `result` | `str` | Summary of what was done (optional for in_progress) |

**Example:**
```python
update_todo(todo_id=1, status="completed", result="Found 5 key research papers")
```

---

### `get_todos()`
Retrieve the current task plan with all statuses. No parameters.

---

## Virtual File System Tools

### `write_file(filepath, content)`
Save content to the virtual file system.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Virtual path (e.g., `"research/notes.md"`) |
| `content` | `str` | File content to save |

---

### `read_file(filepath)`
Read content from a virtual file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Path to read |

---

### `edit_file(filepath, old_content, new_content)`
Replace specific content within a virtual file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Path to edit |
| `old_content` | `str` | Text to find and replace |
| `new_content` | `str` | Replacement text |

---

### `ls(directory)`
List files in a virtual directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `directory` | `str` | Directory path (default: `"/"`) |

---

## Search Tools

### `web_search(query)`
Perform a web search using DuckDuckGo.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query string |

**Returns:** Top 5 search results with titles, URLs, and snippets.

---

## Delegation Tools

### `delegate_task(agent_name, task_prompt)`
Delegate a sub-task to a specialized agent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | `str` | `"search_agent"`, `"summary_agent"`, or `"coding_agent"` |
| `task_prompt` | `str` | Detailed instructions for the sub-agent |

**Example:**
```python
delegate_task(
    agent_name="search_agent",
    task_prompt="Find the top 5 AI frameworks in 2025 with their GitHub stars and key features"
)
```

---

## Sub-Agent Specifications

### Search Agent (`search_agent`)
- **Tools:** `web_search`
- **Purpose:** Gathers information from the web
- **Max iterations:** 5
- **Best for:** Fact-finding, data collection, market research

### Summary Agent (`summary_agent`)
- **Tools:** `read_file`, `write_file`, `edit_file`, `ls`
- **Purpose:** Reads VFS files and synthesizes reports
- **Max iterations:** 5
- **Best for:** Report writing, data synthesis, comparative analysis

### Coding Agent (`coding_agent`)
- **Tools:** `write_file`, `read_file`, `edit_file`, `ls`
- **Purpose:** Writes scripts and processes data
- **Max iterations:** 5
- **Best for:** Python scripts, data processing, code snippets

---

## Configuration Options

All options are set via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `LLM_TEMPERATURE` | `0.4` | LLM temperature (0-1) |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | (optional) | LangSmith API key |
| `LANGCHAIN_PROJECT` | `autonomous-cognitive-engine` | LangSmith project name |
| `MAX_AGENT_ITERATIONS` | `25` | Max agent loop iterations |
| `RECURSION_LIMIT` | `50` | LangGraph recursion limit |

---

## State Schema

The `AgentState` extends LangGraph's `MessagesState`:

```python
class AgentState(MessagesState):
    todos: list[TodoItem]     # Task plan
    current_todo_id: int      # Active task ID
    files: dict[str, str]     # Virtual file system
    iteration_count: int      # Loop counter
    plan_created: bool        # Plan exists flag
```

### TodoItem
```python
class TodoItem(BaseModel):
    id: int          # 1-indexed ID
    task: str        # Task description
    status: str      # "pending" | "in_progress" | "completed"
    result: str      # Completion summary
```
