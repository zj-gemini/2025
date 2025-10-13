import logging
import httpx
import os
import pathlib
import sys
from typing import Any, Dict

sys.path.append("../utils")
from search_api import google_search
from mcp.server.fastmcp import FastMCP


LOG_FILE = "/tmp/faq_server.log"

# Configure logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s *  %(levelname)s *  %(message)s",
    filename=LOG_FILE,
    filemode="a",  # Append mode
)

mcp = FastMCP("faq_service")


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


if __name__ == "__main__":
    # Run over stdio so an MCP client can spawn this process and talk JSON-RPC.
    mcp.run(transport="stdio")
