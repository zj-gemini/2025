import asyncio
import json
import sys

sys.path.append("../utils")
from gemini_api import get_response  # type: ignore
from search_mcp_client import SearchClient
from prompt_builder import build_llm_prompt
from response_parser import *
import fire


async def simple_test(client):
    # Example usage: San Jose, CA
    lat: float = 31.33
    lon: float = -121.88
    state: str = "CA"
    forecast = await client.get_weather_forecast(lat, lon)
    print("\n=== Forecast ===\n" + forecast)

    # Example usage: California alerts
    alerts = await client.get_weather_alerts(state)
    print("\n=== {} Alerts ===\n".format(state) + alerts)


async def call_search_agent(client, user_query: str):
    tools = await client.list_tools()
    print("\n=== Tools ===")
    for tool_name, tool_info in tools.items():
        print(f"\n[{tool_name}] {tool_info.description}")
        print(json.dumps(tool_info.inputSchema, indent=2))

    print("*" * 40)
    prompt = build_llm_prompt(user_query, tools)
    print("\n=== LLM Prompt ===\n" + prompt)
    response = get_response(prompt)
    print("\n=== LLM Response ===\n" + response)

    parsed_response = parse_llm_response(response)
    print("\n=== Parsed LLM Response ===\n")
    print("Thought:", parsed_response.thought)
    if parsed_response.final_response is not None:
        print("Final Response:", parsed_response.final_response)
    if parsed_response.tool_calls is not None:
        print("Tool Calls:", json.dumps(parsed_response.tool_calls, indent=2))
        # make tool calls using weather client
        for tool_call in parsed_response.tool_calls:
            tool_name = tool_call["tool"]
            tool_args = tool_call["arguments"]
            result = await client.call_tool(tool_name, tool_args)
            print("*" * 40)
            print("Result for ", tool_call)
            print("-" * 20)
            print(result)
            print("*" * 40)


async def run_search_agent(
    user_query: str = "What's the weather like in Austin, Texas?",
):
    client = SearchClient("search_mcp_server.py")
    await client.connect_to_server()

    logs = await client.clear_logs()
    print("\n=== Log Clear ===\n" + logs)

    # Simple client test without LLM
    # await simple_test(client)
    await call_search_agent(client, user_query)

    # Display server logs
    logs = await client.tail_logs()
    print("\n=== Server Logs ===\n" + logs)

    await client.close_connection()


if __name__ == "__main__":
    fire.Fire(run_search_agent)
