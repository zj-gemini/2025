from typing import Any
import logging
import httpx
from mcp.server.fastmcp import FastMCP

# Never print() from an MCP server using STDIO; use logging to stderr instead.
logging.basicConfig(level=logging.INFO)

mcp = FastMCP("weather")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "MCP-Weather-Colab/1.0 (yzj.cpp@gmail.com)"  # <-- put your email


async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }
    print(f"Making NWS request to: {url}")  # Added print for debugging
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
async def get_alerts(state: str) -> str:
    """Get active weather alerts for a 2-letter US state (e.g. 'CA')."""
    url = f"{NWS_API_BASE}/alerts/active/area/{state.upper()}"
    data = await make_nws_request(url)
    print(f"Raw alerts data: {data}")  # Added print for debugging
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    feats = data["features"]
    if not feats:
        return "No active alerts for this state."
    return "\n\n---\n\n".join(_fmt_alert(f) for f in feats)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get a short weather forecast for a US lat/lon (e.g., 37.3382, -121.8863)."""
    # 1) Resolve forecast URL from coordinates
    points = await make_nws_request(f"{NWS_API_BASE}/points/{latitude},{longitude}")
    print(f"Raw points data: {points}")  # Added print for debugging
    if not points:
        return "Unable to fetch grid point for these coordinates (are they inside the US?)."

    forecast_url = points.get("properties", {}).get("forecast")
    if not forecast_url:
        return "NWS points endpoint returned no forecast URL."

    # 2) Fetch the actual forecast
    forecast = await make_nws_request(forecast_url)
    print(f"Raw forecast data: {forecast}")  # Added print for debugging
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


if __name__ == "__main__":
    # Run over stdio so an MCP client can spawn this process and talk JSON-RPC.
    mcp.run(transport="stdio")
