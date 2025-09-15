import json
from typing import Any, Dict, List, Optional, Tuple, Union
from mcp.types import Tool
import re
from dataclasses import dataclass


@dataclass
class PlannerResponse:
    thought: str
    files_to_update: List[str]


def parse_planner_response(response_text: str) -> PlannerResponse:
    """
    Parses the LLM response to extract the thought, and files to update.

    Args:
        response_text: The raw text response from the LLM.

    Returns:
        A tuple containing:
            - thought: The thought extracted from the response.
            - files_to_update: A list of file paths to update
    """
    thought_match = re.search(
        r"<ctrl_thought>(.*?)</ctrl_thought>", response_text, re.DOTALL
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    files_to_update = []
    json_match = re.search(r"<ctrl_json>(.*?)</ctrl_json>", response_text, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1).strip()
            data = json.loads(json_str)
            files_to_update = data.get("files_to_update", [])
        except json.JSONDecodeError:
            # If JSON is invalid, leave files_to_update as an empty list
            pass

    return PlannerResponse(thought=thought, files_to_update=files_to_update)
