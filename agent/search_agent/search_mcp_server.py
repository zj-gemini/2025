import logging
import httpx
import os
import pathlib
import sys
from typing import Any, Dict

sys.path.append("../utils")
from search_api import google_search
from mcp.server.fastmcp import FastMCP


LOG_FILE = "/tmp/search_server.log"

# Configure logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s *  %(levelname)s *  %(message)s",
    filename=LOG_FILE,
    filemode="a",  # Append mode
)

mcp = FastMCP("general_search")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "MCP-SEARCH/1.0 (yzj.cpp@gmail.com)"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }
    logging.info(f"Making NWS request to: {url}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logging.exception(f"NWS request failed: {e}")
        return None


def _fmt_alert(feature: dict) -> str:
    p = feature.get("properties", {})
    return (
        f"Event: {p.get('event','Unknown')}\n"
        f"Area: {p.get('areaDesc','Unknown')}\n"
        f"Severity: {p.get('severity','Unknown')}\n"
        f"Description: {p.get('description','(none)')}\n"
        f"Instructions: {p.get('instruction','(none)')}"
    )


@mcp.tool()
def tail_logs(n: int = 100) -> str:
    """Return the last n lines of the server log."""
    p = pathlib.Path(LOG_FILE)
    if not p.exists():
        return "(no log yet)"
    lines = p.read_text().splitlines()[-n:]
    return "\n".join(lines)


@mcp.tool()
def clear_logs() -> str:
    """Clear the server log file."""
    p = pathlib.Path(LOG_FILE)
    try:
        # Overwrite the file with an empty string instead of deleting
        p.write_text("")
        logging.info(f"Log file {LOG_FILE} cleared by overwriting.")
        return "Log file cleared."
    except Exception as e:
        # Log any errors during file clearing
        logging.error(f"Error clearing log file {LOG_FILE}: {e}")
        return f"Error clearing log file: {e}"


@mcp.tool()
async def get_weather_alerts(state: str) -> str:
    """Get active weather alerts for a 2-letter US state (e.g. 'CA').."""
    url = f"{NWS_API_BASE}/alerts/active/area/{state.upper()}"
    data = await make_nws_request(url)
    logging.info(f"Raw alerts data: {data}")
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    feats = data["features"]
    if not feats:
        return "No active alerts for this state."
    return "\n\n---\n\n".join(_fmt_alert(f) for f in feats)


@mcp.tool()
async def get_weather_forecast(latitude: float, longitude: float) -> str:
    """Get a short weather forecast for a US lat/lon (e.g., 37.3382, -121.8863)."""
    # 1) Resolve forecast URL from coordinates
    points = await make_nws_request(f"{NWS_API_BASE}/points/{latitude},{longitude}")
    logging.info(f"Raw points data: {points}")
    if not points:
        return "Unable to fetch grid point for these coordinates (are they inside the US?)."

    forecast_url = points.get("properties", {}).get("forecast")
    if not forecast_url:
        return "NWS points endpoint returned no forecast URL."

    # 2) Fetch the actual forecast
    forecast = await make_nws_request(forecast_url)
    logging.info(f"Raw forecast data: {forecast}")
    if not forecast:
        return "Unable to fetch detailed forecast."

    periods = (forecast.get("properties") or {}).get("periods") or []
    if not periods:
        return "No forecast periods available."

    # Show next 5 periods for brevity
    lines = []
    for p in periods[:5]:
        lines.append(
            f"{p.get('name','Period')}:\n"
            f"  Temp: {p.get('temperature','?')}°{p.get('temperatureUnit','')}\n"
            f"  Wind: {p.get('windSpeed','?')} {p.get('windDirection','')}\n"
            f"  Forecast: {p.get('detailedForecast','(none)')}"
        )
    return "\n\n".join(lines)


@mcp.tool()
async def search_google(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform a Google search and return formatted results.

    This function uses Google Custom Search API to search the web based on the provided query.
    It formats the results into a consistent structure and handles potential errors.

    Args:
        query (str): The search query string
        num_results (int, optional): Number of search results to return. Defaults to 5.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the search was successful
            - results (list): List of dictionaries with title, link, and snippet
            - total_results (str): Total number of results found (when successful)
            - error (str): Error message (when unsuccessful)
    """
    return google_search(query, num_results)


if __name__ == "__main__":
    # Run over stdio so an MCP client can spawn this process and talk JSON-RPC.
    mcp.run(transport="stdio")
