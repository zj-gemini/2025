import asyncio
from typing import Any, List, Dict
from code_prompt_builder import *
import sys

sys.path.append("../utils")
from gemini_api import get_response  # type: ignore
from code_response_parser import *
from code_client import CodeClient
import json
import subprocess
import fire


async def run_agent(folder_path: str, server_script: str = "code_server.py"):
    """Runs the code agent CLI."""
    client = CodeClient(server_script)
    await client.connect_to_server()

    tools = await client.list_tools()
    print("\n=== Tools ===")
    for tool_name, tool_info in tools.items():
        print(f"\n[{tool_name}] {tool_info.description}")
        print(json.dumps(tool_info.inputSchema, indent=2))

    logs = await client.clear_logs()
    print("\n=== Log Clear ===\n" + logs)

    print("*" * 40)
    print("Starting chat loop. Type 'exit' or 'quit' to end.")
    print("*" * 40)

    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        planner_prompt = build_planner_prompt(user_query, folder_path)
        print("\n=== planner prompt ===\n" + planner_prompt)
        response = get_response(planner_prompt)
        print("\n=== LLM Response ===\n" + response)

        planner_response = parse_planner_response(response)
        print("\n=== Parsed LLM Response ===\n")
        print("Thought:", planner_response.thought)
        if not planner_response.files_to_update:
            print("No feasible plan")
            break
        print(
            "Files to update:", json.dumps(planner_response.files_to_update, indent=2)
        )
        diffs = {}
        for file_and_plan in planner_response.files_to_update:
            diffs[file_and_plan["file_path"]] = update_file(
                file_and_plan["file_path"], file_and_plan["plan"]
            )

        print("Here's the summary of the changes:")
        for path, diff in diffs.items():
            print("File:", path)
            print(diff)
        print("*" * 40)

    # Display server logs
    logs = await client.tail_logs()
    print("\n=== Server Logs ===\n" + logs)

    await client.close_connection()


def update_file(file_path: str, plan: str) -> str:
    try:
        with open(file_path, "r") as f:
            content = f.read()
        code_update_prompt = build_code_update_prompt(file_path, content, plan)
        print("\n=== Code update prompt ===\n" + code_update_prompt)
        response = get_response(code_update_prompt)
        print("\n=== Generated source code ===\n" + response)

        # Temporarily stage changes to see a clean diff of just the LLM's update
        subprocess.run(["git", "add", file_path], check=False)

        with open(file_path, "w") as f:
            f.write(response)
        print(f"--- Successfully updated {file_path} ---")

        # Show the diff
        diff_process = subprocess.run(
            ["git", "diff", "--", file_path], capture_output=True, text=True
        )
        diff = diff_process.stdout
        # Reset the staging to not automatically commit the 'add'
        subprocess.run(["git", "reset", file_path], check=False)
        return diff
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    # if parsed_response.tool_calls:
    #     print("Tool Calls:", json.dumps(parsed_response.tool_calls, indent=2))
    #     for tool_call in parsed_response.tool_calls:
    #         tool_name = tool_call["tool"]
    #         tool_args = tool_call["arguments"]
    #         result = await client.call_tool(tool_name, tool_args)
    #         print("*" * 40)
    #         print(f"Result for {tool_name}({tool_args}):")
    #         print("-" * 20)
    #         print(result)
    #         print("*" * 40)
    # elif parsed_response.final_response:
    #     print("Final Response:", parsed_response.final_response)


if __name__ == "__main__":
    fire.Fire(run_agent)
