import sys
import pathlib
import json
from typing import List, Dict, Any, Union
from dataclasses import dataclass
import dataclasses

# Add project root to path to allow running as a script from any location.
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent.conversation import Message
from src.api.gemini import get_response
from src.agent.prompt.planner_prompt import build_prompt as build_planner_prompt
from src.agent.deep_research import (
    run as run_deep_research,
    DeepResearchStatus,
    DeepResearchResponse,
)
from src.db import create_ticket


@dataclass
class PuntArgs:
    response: str


@dataclass
class AnswerWithCatalogArgs:
    response: str


@dataclass
class DeepResearchArgs:
    scientist_ids: List[str]


@dataclass
class SystemErrorArgs:
    error_message: str


@dataclass
class Action:
    type: str
    arguments: Union[PuntArgs, AnswerWithCatalogArgs, DeepResearchArgs, SystemErrorArgs]


@dataclass
class Plan:
    thought: str
    action: Action


def plan_next(messages: List[Message]) -> Plan:
    """
    Takes a list of messages, builds a planner prompt, and returns the planner's JSON response.
    """
    prompt = build_planner_prompt(messages)
    print("---------- LLM PROMPT ----------")
    print(prompt)
    print("--------------------------------\n")

    print("... Sending prompt to Gemini ...\n")
    response_text = get_response(prompt)
    print("---------- LLM RESPONSE ----------")
    print(response_text)
    print("--------------------------------\n")

    # Clean and parse the JSON response
    # The model is instructed to wrap the JSON in markdown tags.
    json_text = (
        response_text.strip().removeprefix("<ctrl_json>").removesuffix("</ctrl_json>")
    )
    try:
        response_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        error_message = f"Failed to parse planner response as JSON: {e}. Response text: '{response_text}'"
        print(error_message)
        return Plan(
            thought="System error when parsing the planner's response.",
            action=Action(
                type="SYSTEM_ERROR",
                arguments=SystemErrorArgs(error_message=error_message),
            ),
        )

    # Deserialize the JSON into dataclasses
    action_data = response_data["action"]
    action_type = action_data["type"]
    args_data = action_data["arguments"]

    arg_class_map = {
        "PUNT": PuntArgs,
        "ANSWER_WITH_CATAGLOG": AnswerWithCatalogArgs,
        "DEEP_RESEARCH": DeepResearchArgs,
        "SYSTEM_ERROR": SystemErrorArgs,
    }
    arg_class = arg_class_map.get(
        action_type, lambda **kwargs: kwargs
    )  # Fallback to dict if type is unknown

    return Plan(
        thought=response_data["thought"],
        action=Action(type=action_type, arguments=arg_class(**args_data)),
    )


def run(messages: List[Message]) -> str:
    """
    Generates a plan and executes the corresponding action.

    Args:
        messages: The conversation history.

    Returns:
        A string response to be sent to the user.
    """
    plan = plan_next(messages)
    print(f"--- Planner's Thought ---\n{plan.thought}\n-------------------------\n")
    print(f"--- Chosen Action: {plan.action.type} ---")

    action_type = plan.action.type
    action_args = plan.action.arguments

    if action_type in ("PUNT", "ANSWER_WITH_CATAGLOG"):
        print("Action execution: Responding directly based on the catalog.")
        return action_args.response
    elif action_type == "DEEP_RESEARCH":
        # Here you would trigger the deep research loop.
        print(
            f"Action execution: Starting deep research for scientist IDs: {action_args.scientist_ids}"
        )
        history_json = json.dumps([m.model_dump() for m in messages])
        user_question = messages[-1].text if messages else "Unknown question"
        research_result = run_deep_research(
            messages, action_args.scientist_ids, thought=plan.thought
        )
        if research_result.status == DeepResearchStatus.SUCCESS:
            return research_result.response
        else:
            print("Deep research failed to find an answer. Creating a ticket.")
            error_desc = f"Deep research did not return a SUCCESS status. Final status: {research_result.status.name}."
            create_ticket(
                user_question=user_question,
                user_contact="N/A",
                conversation_history=history_json,
                error_description=error_desc,
            )
            return research_result.response
    elif action_type == "SYSTEM_ERROR":
        print(f"Action execution: Handling system error: {action_args.error_message}")
        history_json = json.dumps([m.model_dump() for m in messages])
        user_question = messages[-1].text if messages else "Unknown question"
        print("System error occurred. Creating a ticket.")
        create_ticket(
            user_question=user_question,
            user_contact="N/A",
            conversation_history=history_json,
            error_description=action_args.error_message,
        )
        return "I'm sorry, I encountered a system error and couldn't process your request. A ticket has been created for our team to investigate the issue."


def main():
    """Dry run the prompt generation."""
    sample_messages = [
        Message(sender="user", text="Who was Einstein?"),
        Message(
            sender="bot",
            text="Albert Einstein was a German-born theoretical physicist...",
        ),
        Message(sender="user", text="When did he win Nobel Prize?"),
    ]
    response = run(sample_messages)
    print("---------- FINAL RESPONSE ----------")
    print(response)
    print("-----------------------------------")


if __name__ == "__main__":
    main()
