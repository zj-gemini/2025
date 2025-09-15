# @title Prompt Builder

import json
from typing import Any, Dict, List, Optional, Tuple, Union
from mcp.types import Tool
import os

CODE_BASE_DESCRIPTION = """

"""

# ---------- System instruction ----------
PLANNER_SYSTEM_INSTRUCTION = """
You are a code agent that can update source code base based on user requests.

You receive:

1) A user query.
2) CODE_BASE_DESCRIPTION: full description of the source, including folder structures and guidance about how to add or modify a feature.
3) A list of all files in the code base

Based on the user query and CODE_BASE_DESCRIPTION, please make a plan of code changes. The plan should only include two things:
*   Generate a concise thought about the action plans: which files should be updated.
*   Based on the thought, a list of files to be modified, in JSON format. If no files are needed to be updated, or we can't figure out a plan, generate an empty list.

CRITICAL RULES:
*   The generated thought must have starting markdown <ctrl_thought> and ending markdown </ctrl_thought>
*   The ONLY allowed top-level key is "files_to_update".
*   "files_to_update" must be a JSON array. Each item must be an object of the form: { "file_path": "<relative_file_path>", "plan": "<short_description_of_the_update>" }

Examples of valid outputs:

*	Example 1

<ctrl_thought>
The user is asking to make 1 million dallors, and I don't have any tool can do that. No tool call can be made.
</ctrl_thought>
<ctrl_json>
{
    "files_to_update": []
}
</ctrl_json>

*	Example 2

<ctrl_json>
{
    "files_to_update":
	[
		{
			"file_path": "./src/main.tsx",
			"plan": "add more comments"
		}
	]
}
</ctrl_json>
"""

CODE_UPDATE_SYSTEM_INSTRUCTION = """
You are a code updater.

You receive:

1) CODE_BASE_DESCRIPTION: full description of the source, including folder structures and guidance about how to add or modify a feature.
2) Full source code.
3) The path of the source code
4) Plan for the code update

Based on the code update plan and CODE_BASE_DESCRIPTION, please generate the updated source code.

CRITICAL RULES:
*   There should be only the updated source code in the output. NOTHING else is needed.

"""

# ---------- System instruction ----------
FC_SYSTEM_INSTRUCTION = """
You are a code agent for a game simulator.

You receive:

1) A catalog of available tools (name, description, and JSON Schema for their parameters).
2) A user query.
3) CODE_BASE_DESCRIPTION: full description of the source, including folder structures and guidance about how to add or modify a feature.

At each step, you should perform two actions:
	*	Generate a concise thought about the action plan of the current step and decide which tools (zero or more) are relevant to satisfy the user’s query.
	*	Based on the thought, perform *ONLY ONE* of the two actions:
		1) Generate a valid JSON to describe the tool calls with correct arguments that conform to the tools's JSON Schema (argument json_schema).
		2) If no tool is relevant, generate a response to explain why no tool calls are made.

CRITICAL RULES:

- The generated thought must have starting markdown <ctrl_thought> and ending markdown </ctrl_thought>

- Tool call JSON MUST be valid JSON, with starting markdown <ctrl_json> and ending markdown </ctrl_json>

- Add a newline beween thought, tool call JSON and response (if any)

- The ONLY allowed top-level key is "tool_calls".

- "tool_calls" must be a JSON array. Each item must be an object of the form: { "tool": "<tool_name>", "arguments": { ... } }

- Generatecd JSON MUST have the right indent

- Do not invent tools not in the catalog.

- Do not include undefined parameters.

- Satisfy required parameters from the schema; if unknown, choose sensible defaults if the schema allows it, otherwise omit the tool.

- Keep arguments concise and strictly typed per the schema (strings as strings, numbers as numbers, booleans as booleans, arrays as arrays, etc.).

Examples of valid outputs:

*	Example 1

<ctrl_thought>
The user is asking to make 1 million dallors, and I don't have any tool can do that. No tool call can be made.
</ctrl_thought>
Sorry, no tool call can be made as none of them is able to make 1 million dallors.

*	Example 2

<ctrl_json>
{ "tool_calls":
	[
		{
			"tool": "read_file",
			"arguments":
			{
				"file_path": "initializePlayMode.ts",
			}
		}
	]
}
</ctrl_json>
"""


def build_planner_prompt(user_query: str, folder_path: str) -> str:
    """Create a single prompt string: system + catalog + user input."""
    files = list_files_recursively(folder_path)
    file_block = "\n".join(files)
    user_block = f"USER QUERY:\n{user_query}"

    # Single concatenated prompt for simple text-only LLMs
    prompt = (
        PLANNER_SYSTEM_INSTRUCTION
        + "\n\nCODE_BASE_DESCRIPTION guidance:\n"
        + CODE_BASE_DESCRIPTION
        + "\n\nFiles catalog:\n"
        + file_block
        + "\n\n"
        + user_block
    )
    return prompt


def build_code_update_prompt(file_path: str, file_content: str, plan: str) -> str:
    """Get the entire source code from a file, and generated the updated source code"""

    # Single concatenated prompt for simple text-only LLMs
    prompt = (
        CODE_UPDATE_SYSTEM_INSTRUCTION
        + "\n\nCODE_BASE_DESCRIPTION guidance:\n"
        + CODE_BASE_DESCRIPTION
        + "\n\nSource code file path:"
        + file_path
        + "\n\nFull source code:\n"
        + file_content
        + "\n\nCode update plan:\n"
        + plan
    )
    return prompt


def build_llm_prompt(user_query: str, tool_specs: dict[str, Tool]) -> str:
    """Create a single prompt string: system + catalog + user input."""
    catalog_lines = []
    for tool_name, tool_info in tool_specs.items():
        name = tool_name
        desc = tool_info.description
        schema = tool_info.inputSchema["properties"]
        schema_json = json.dumps(schema, ensure_ascii=False)
        catalog_lines.append(
            f"- name: {name}\n  description: {desc}\n  argument json_schema: {schema_json}"
        )
    catalog_block = "\n".join(catalog_lines)

    user_block = f"USER QUERY:\n{user_query}"

    # Single concatenated prompt for simple text-only LLMs
    prompt = (
        FC_SYSTEM_INSTRUCTION
        + "\n\nTOOL CATALOG:\n"
        + catalog_block
        + "\n\n"
        + user_block
    )
    return prompt


def list_files_recursively(folder_path, max_depth=2):
    """
    Lists all non-hidden files in a folder and its subfolders,
    up to a maximum depth, maintaining the folder structure order.

    Args:
      folder_path: The path to the folder to start the search.
      max_depth: The maximum depth of subdirectories to traverse.

    Returns:
      A list of file paths.
    """
    file_list = []
    start_depth = folder_path.count(os.sep)
    for dirpath, dirnames, filenames in os.walk(folder_path):
        current_depth = dirpath.count(os.sep) - start_depth
        if current_depth >= max_depth:
            # Prune subdirectories from being visited
            dirnames[:] = []

        # Sort filenames to ensure consistent order
        filenames.sort()
        for filename in filenames:
            # Ignore hidden files (those starting with '.')
            if not filename.startswith("."):
                file_path = os.path.join(dirpath, filename)
                file_list.append(file_path)

        # Sort dirnames to ensure consistent folder structure order
        dirnames.sort()

    return file_list
