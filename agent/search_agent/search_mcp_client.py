import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool
import sys
from typing import Any, Dict


class SearchClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect_to_server(self):
        if self._session:
            print("Already connected.")
            return

        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=sys.executable, args=[self.server_script]
        )
        stdio = await self._exit_stack.enter_async_context(stdio_client(params))
        read, write = stdio
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        print("Connected to server.")

    async def close_connection(self):
        if self._exit_stack:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None
        print("Connection closed.")

    async def call_tool(self, tool_name: str, params: Dict[str, Any] = {}) -> str:
        """Call a tool on the MCP server."""
        if not self._session:
            return "Error: Not connected to server."

        result = await self._session.call_tool(tool_name, params)
        out_chunks = []
        for c in result.content:
            if isinstance(c, TextContent):
                out_chunks.append(c.text)
            elif isinstance(c, dict) and c.get("type") == "text":
                out_chunks.append(c.get("text") or "")
        return "\n".join(out_chunks)

    async def get_weather_forecast(self, latitude: float, longitude: float) -> str:
        """Get a short weather forecast for a US lat/lon."""
        return await self.call_tool(
            "get_weather_forecast", {"latitude": latitude, "longitude": longitude}
        )

    async def get_weather_alerts(self, state: str) -> str:
        """Get active weather alerts for a 2-letter US state."""
        return await self.call_tool("get_weather_alerts", {"state": state})

    async def tail_logs(self, n: int = 100) -> str:
        """Return the last n lines of the server log."""
        return await self.call_tool("tail_logs", {"n": n})

    async def clear_logs(self) -> str:
        """Clear logs."""
        return await self.call_tool("clear_logs")

    async def list_tools(self) -> dict[str, Tool]:
        """List available tools and their descriptions."""
        if not self._session:
            print("Error: Not connected to server.")
            return {}

        response = await self._session.list_tools()
        tools_dict = {tool.name: tool for tool in response.tools}
        return tools_dict
