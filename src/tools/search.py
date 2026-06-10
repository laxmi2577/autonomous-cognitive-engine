"""
Web Search Tool for the Autonomous Cognitive Engine.
======================================================
Uses DuckDuckGo Search — completely free, no API key required.
"""

from __future__ import annotations

from langchain_core.tools import tool


# Lazy-initialized search instance (avoids crash at import time)
_ddg_search = None


def _get_ddg_search():
    """Lazily create the DuckDuckGo search instance."""
    global _ddg_search
    if _ddg_search is None:
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            _ddg_search = DuckDuckGoSearchResults(
                num_results=5,
                output_format="list",
            )
        except ImportError:
            _ddg_search = "unavailable"
    return _ddg_search


def _fallback_search(query: str) -> str:
    """Basic fallback search using duckduckgo-search directly."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for: {query}"
        formatted = [f"🔍 **Search Results for:** {query}\n"]
        for i, r in enumerate(results, 1):
            formatted.append(f"**{i}. {r.get('title', 'No title')}**")
            formatted.append(f"   {r.get('body', 'No description')}")
            if r.get('href'):
                formatted.append(f"   🔗 {r['href']}")
            formatted.append("")
        return "\n".join(formatted)
    except Exception as e:
        return f"❌ Search unavailable: {str(e)}"


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
    ddg = _get_ddg_search()
    
    # If langchain wrapper is unavailable, use direct fallback
    if ddg == "unavailable":
        return _fallback_search(query)
    
    try:
        results = ddg.invoke(query)
        
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
        # Try fallback if langchain wrapper fails
        return _fallback_search(query)


# Export the tool
search_tool = web_search

