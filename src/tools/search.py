"""
Web Search Tool for the Autonomous Cognitive Engine.
======================================================
Uses DuckDuckGo Search — completely free, no API key required.
"""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool


# Create the DuckDuckGo search instance
_ddg_search = DuckDuckGoSearchResults(
    num_results=5,
    output_format="list",
)


@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo.
    
    Use this to find recent information, facts, articles, or data 
    on any topic. Always save important search results to the file 
    system for later reference.
    
    Args:
        query: The search query (be specific for better results).
    
    Returns:
        Search results with titles, snippets, and URLs.
    """
    try:
        results = _ddg_search.invoke(query)
        
        if isinstance(results, list):
            formatted = [f"🔍 **Search Results for:** {query}\n"]
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", result.get("body", "No description"))
                    link = result.get("link", result.get("href", ""))
                    formatted.append(f"**{i}. {title}**")
                    formatted.append(f"   {snippet}")
                    if link:
                        formatted.append(f"   🔗 {link}")
                    formatted.append("")
                else:
                    formatted.append(f"**{i}.** {result}\n")
            return "\n".join(formatted)
        else:
            return f"🔍 **Search Results for:** {query}\n\n{str(results)}"
            
    except Exception as e:
        return f"❌ Search failed: {str(e)}. Try a different query or check your connection."


# Export the tool
search_tool = web_search
