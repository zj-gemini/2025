from typing import Any
import logging
import httpx
import json
from mcp.server.fastmcp import FastMCP
import os
import pathlib

LOG_FILE = "/Users/zjy/Downloads/scaffold-template-package/scaffold/mcp_server.log"

# Configure logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s *  %(levelname)s *  %(message)s",
    filename=LOG_FILE,
    filemode="a",  # Append mode
)

mcp = FastMCP("code_agent")


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
async def read_file(file_path: str) -> str:
    """
    Reads the content of a specific file and returns it as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at '{file_path}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool()
async def list_directory(directory_path: str) -> str:
    """
    Lists all files and subdirectories within a given directory path.
    Returns a JSON string with 'files' and 'directories' lists.
    """
    try:
        if not os.path.isdir(directory_path):
            return f"Error: Directory not found at '{directory_path}'"

        items = os.listdir(directory_path)
        files = [
            item for item in items if os.path.isfile(os.path.join(directory_path, item))
        ]
        directories = [
            item for item in items if os.path.isdir(os.path.join(directory_path, item))
        ]

        return json.dumps({"files": files, "directories": directories})
    except Exception as e:
        return f"Error listing directory: {str(e)}"


if __name__ == "__main__":
    # Run over stdio so an MCP client can spawn this process and talk JSON-RPC.
    mcp.run(transport="stdio")
