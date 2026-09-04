"""Web search and URL content fetching tools for OmniAgent."""

import re
from typing import Optional
import urllib.request
import urllib.parse
from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web for current information, facts, documentation, or news.
    
    Args:
        query: The search query string.
        max_results: Number of search results to return (default: 5).
    """
    # Attempt duckduckgo_search library
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No results found for query: '{query}'."
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                href = r.get("href", "")
                body = r.get("body", "")
                formatted.append(f"{i}. [{title}]({href})\n   {body}")
            return "\n\n".join(formatted)
    except Exception as e:
        # Fallback using DuckDuckGo HTML or API
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            # Extract simple snippets
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html)
            titles = re.findall(r'<a class="result__url[^>]*>(.*?)</a>', html)
            if not snippets:
                return f"Search completed with error or no results: {str(e)}"
            formatted = []
            for i, (t, s) in enumerate(zip(titles[:max_results], snippets[:max_results]), 1):
                clean_s = re.sub(r'<[^>]+>', '', s).strip()
                clean_t = re.sub(r'<[^>]+>', '', t).strip()
                formatted.append(f"{i}. {clean_t}\n   {clean_s}")
            return "\n\n".join(formatted)
        except Exception as fallback_err:
            return f"Web search failed: {str(e)} (fallback error: {str(fallback_err)})"


@tool
def fetch_web_page(url: str, max_chars: int = 4000) -> str:
    """Fetch the text content of any public URL and return it in readable format.
    
    Args:
        url: The web URL (must start with http:// or https://).
        max_chars: Maximum characters of text content to return (default: 4000).
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: URL must start with http:// or https://"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Strip scripts, styles, and extract text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text(separator="\n")
        except ImportError:
            clean = re.sub(r"<(script|style).*?>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", clean)

        # Collapse whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + f"\n\n[... truncated to {max_chars} characters]"
        return clean_text if clean_text else "Page returned empty content."

    except Exception as e:
        return f"Error fetching URL '{url}': {str(e)}"
