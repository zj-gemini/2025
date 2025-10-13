import wikipedia
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class WikiPage:
    """A structured dataclass to hold a single Wikipedia page summary."""

    title: str
    summary: str
    url: str
    content: str


@dataclass
class WikiResponse:
    """A structured dataclass for the entire Wikipedia API response."""

    success: bool
    page: Optional[WikiPage] = None
    error: Optional[str] = None
    options: Optional[List[str]] = None  # For disambiguation errors


def get_wiki_summary(query: str) -> WikiResponse:
    """
    Fetches a summary from Wikipedia for a given query.

    This function uses the wikipedia library to find a page, handling common
    issues like pages not being found or disambiguation pages.

    Args:
        query (str): The search term (e.g., "Albert Einstein").

    Returns:
        WikiResponse: An object containing the page data or an error.
    """
    try:
        # auto_suggest helps find pages even with small typos
        # redirect automatically follows page redirects
        page = wikipedia.page(query, auto_suggest=True, redirect=True)

        wiki_page = WikiPage(
            title=page.title, summary=page.summary, url=page.url, content=page.content
        )
        return WikiResponse(success=True, page=wiki_page)

    except wikipedia.exceptions.PageError:
        return WikiResponse(success=False, error=f"Page not found for query: '{query}'")
    except wikipedia.exceptions.DisambiguationError as e:
        return WikiResponse(
            success=False,
            error="Disambiguation page found. Please be more specific.",
            options=e.options[:5],  # Return the first 5 suggestions
        )
    except Exception as e:
        return WikiResponse(success=False, error=str(e))


if __name__ == "__main__":
    # --- Example Usage ---
    search_term = "Albert Einstein"
    print(f"Searching Wikipedia for: '{search_term}'...\n")
    response = get_wiki_summary(search_term)

    if response.success and response.page:
        print(f"Title: {response.page.title}")
        print(f"URL: {response.page.url}")
        print("\n--- Summary ---")
        print(response.page.summary)
        print(f"\n--- Full content length: {len(response.page.content)} characters ---")
    else:
        print(f"Failed to get Wikipedia summary. Error: {response.error}")
        if response.options:
            print("Did you mean one of these?", response.options)
