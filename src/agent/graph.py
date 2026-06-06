"""
LangGraph Graph Construction for the Autonomous Cognitive Engine.
===================================================================
Builds the main agent graph with the ReAct loop:
  Agent (LLM reasoning) → Router → Tools → Agent → ...
  
This is the central orchestration layer that ties everything together.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import agent_node, tool_node, should_continue
from src.config import RECURSION_LIMIT


def build_agent_graph(with_checkpointer: bool = True) -> StateGraph:
    """
    Build and compile the main agent graph.
    
    The graph follows the ReAct pattern:
    START → agent → (tools → agent)* → END
    
    Args:
        with_checkpointer: Whether to add a memory checkpointer 
                          (enables conversation persistence).
    
    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    # ── Create the graph ──
    graph = StateGraph(AgentState)
    
    # ── Add nodes ──
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    # ── Add edges ──
    # Start → Agent (always begin with the agent reasoning)
    graph.add_edge(START, "agent")
    
    # Agent → Router (decide: call tools or finish?)
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",   # Agent wants to use tools → go to tool node
            "end": END,          # Agent is done → finish
        }
    )
    
    # Tools → Agent (after executing tools, go back to agent for reasoning)
    graph.add_edge("tools", "agent")
    
    # ── Compile the graph ──
    checkpointer = MemorySaver() if with_checkpointer else None
    
    compiled = graph.compile(
        checkpointer=checkpointer,
    )
    
    return compiled


def create_agent():
    """
    Create and return a ready-to-use agent instance.
    
    Usage:
        agent = create_agent()
        result = agent.invoke(
            {"messages": [HumanMessage(content="Research quantum computing")]},
            config={"configurable": {"thread_id": "session-1"}, "recursion_limit": 50}
        )
    """
    return build_agent_graph(with_checkpointer=True)


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import validate_config
    
    print("🧠 Autonomous Cognitive Engine — Graph Builder\n")
    
    if not validate_config():
        print("❌ Fix configuration before running the agent.")
        exit(1)
    
    agent = create_agent()
    print("✅ Agent graph compiled successfully!")
    
    # Print graph structure
    try:
        print("\n📊 Graph Structure:")
        print(agent.get_graph().draw_ascii())
    except Exception:
        print("   (Graph visualization requires additional packages)")
    
    print("\n🚀 Agent is ready! Use create_agent() to get started.")
