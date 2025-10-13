import os
from dataclasses import dataclass
from typing import List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load configuration from .env
load_dotenv()
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")


@dataclass
class SearchResult:
    """A structured dataclass to hold a single search result."""

    title: str
    link: str
    snippet: str


@dataclass
class SearchResponse:
    """A structured dataclass for the entire search response."""

    success: bool
    results: List[SearchResult]
    total_results: Optional[str] = None
    error: Optional[str] = None


def google_search(query: str, num_results: int = 2) -> SearchResponse:
    """
    Perform a Google search and return formatted results.

    This function uses Google Custom Search API to search the web based on the provided query.
    It formats the results into a consistent structure and handles potential errors.

    Args:
        query (str): The search query string
        num_results (int, optional): Number of search results to return. Defaults to 5.

    Returns:
        SearchResponse: An object containing the search results and status.
    """
    try:
        # Initialize Google Custom Search API
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)

        # Execute the search
        # pylint: disable=no-member
        result = (
            service.cse()
            .list(q=query, cx=GOOGLE_SEARCH_ENGINE_ID, num=num_results)
            .execute()
        )

        # Format the search results
        formatted_results = []
        if "items" in result:
            for item in result["items"]:
                formatted_results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                    )
                )

        return SearchResponse(
            success=True,
            results=formatted_results,
            total_results=result.get("searchInformation", {}).get("totalResults", "0"),
        )

    except HttpError as error:
        return SearchResponse(
            success=False, error=f"API Error: {str(error)}", results=[]
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        return SearchResponse(success=False, error=str(error), results=[])


if __name__ == "__main__":
    # --- Example Usage ---
    response = google_search("Einstein physist wiki")
    if response.success:
        print(
            f"Search successful! Found approximately {response.total_results} results."
        )
        print(f"Showing first {len(response.results)} results:\n")
        for i, result in enumerate(response.results, 1):
            print(f"{i}. {result.title}")
            print(f"   Link: {result.link}")
            print(f"   Snippet: {result.snippet}\n")
    else:
        print(f"Search failed: {response.error}")
