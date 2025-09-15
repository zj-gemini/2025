import json
from typing import Any, Dict, List, Optional, Tuple, Union
from mcp.types import Tool
import re
from dataclasses import dataclass


@dataclass
class LLMResponse:
    thought: str
    tool_calls: Optional[List[Dict[str, Any]]]
    final_response: Optional[str]


def parse_llm_response(response_text: str) -> LLMResponse:
    """
    Parses the LLM response to extract the thought, tool calls, and final response.

    Args:
        response_text: The raw text response from the LLM.

    Returns:
        A tuple containing:
            - thought: The thought extracted from the response.
            - tool_calls: A list of tool calls if present, otherwise None.
            - final_response: The final response text if no tool calls are present, otherwise None.
    """
    thought_match = re.search(
        r"<ctrl_thought>(.*?)</ctrl_thought>", response_text, re.DOTALL
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    json_match = re.search(r"<ctrl_json>(.*?)</ctrl_json>", response_text, re.DOTALL)
    tool_calls = None
    final_response = None

    if json_match:
        try:
            json_str = json_match.group(1).strip()
            tool_calls_data = json.loads(json_str)
            tool_calls = tool_calls_data.get("tool_calls")
        except json.JSONDecodeError:
            # If JSON is invalid, treat the rest as final response
            final_response = response_text[json_match.end() :].strip()
    else:
        # If no JSON block, the rest of the text after the thought is the final response
        if thought_match:
            final_response = response_text[thought_match.end() :].strip()
        else:
            final_response = response_text.strip()

    # If both thought and tool_calls are found, the final_response should be None.
    # If only thought is found, final_response is the text after thought.
    # If only tool_calls are found, final_response is None.
    # If neither is found, final_response is the original text.
    if thought_match and json_match:
        final_response = None
    elif thought_match and not json_match:
        final_response = response_text[thought_match.end() :].strip()
    elif not thought_match and not json_match:
        final_response = response_text.strip()

    # Clean up the final response to remove any remaining markdown tags
    if final_response:
        final_response = re.sub(
            r"<ctrl_thought>.*?</ctrl_thought>", "", final_response, flags=re.DOTALL
        ).strip()
        final_response = re.sub(
            r"<ctrl_json>.*?</ctrl_json>", "", final_response, flags=re.DOTALL
        ).strip()

    return LLMResponse(
        thought=thought, tool_calls=tool_calls, final_response=final_response
    )


def test_response_parser():
    response = """
<ctrl_thought>
The user is asking about weather alerts in Texas. The `get_alerts` tool is suitable for this purpose, as it takes a 2-letter US state code as an argument. I should call this tool with 'TX' for Texas.
</ctrl_thought>
<ctrl_json>
{
        "tool_calls": [
                {
                        "tool": "get_alerts",
                        "arguments": {
                                "state": "TX"
                        }
                }
        ]
}
</ctrl_json>
"""

    parsed_response = parse_llm_response(response)

    print("Thought:", parsed_response.thought)
    if parsed_response.tool_calls is not None:
        print("Tool Calls:", json.dumps(parsed_response.tool_calls, indent=2))
    if parsed_response.final_response is not None:
        print("Final Response:", parsed_response.final_response)


# test_response_parser()
