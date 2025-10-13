import sys
import pathlib
import json
from typing import List, Dict, Any, Union
from dataclasses import dataclass
import dataclasses
from enum import Enum, auto

# Add project root to path to allow running as a script from any location.
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent.conversation import Message
from src.api.gemini import get_response
from src.agent.prompt.deep_research_prompt import build_prompt
from knowledge.scientists import read_scientists_full


class DeepResearchStatus(Enum):
    """Enum to represent the status of the deep research."""

    SUCCESS = auto()
    FAILURE = auto()
    BEST_EFFORT = auto()


@dataclass
class DeepResearchResponse:
    response: str
    status: DeepResearchStatus


@dataclass
class AnswerArgs:
    answer: str


@dataclass
class NoteDownArgs:
    note: str


@dataclass
class IrrelevantArgs:
    pass


@dataclass
class SystemErrorArgs:
    error_message: str


@dataclass
class Action:
    type: str
    arguments: Union[AnswerArgs, NoteDownArgs, IrrelevantArgs, SystemErrorArgs]


@dataclass
class DeepResearchPlan:
    thought: str
    action: Action
    best_effort: str


def plan_next_step(
    messages: List[Message], notes: List[str], new_snippet: str, plan: str
) -> DeepResearchPlan:
    """
    Takes conversation, notes, and a new snippet to plan the next deep research step.
    """
    prompt = build_prompt(messages, notes, new_snippet, plan)
    print("--- DEEP RESEARCH PROMPT ---")
    print(prompt)
    print("----------------------------")
    response_text = get_response(prompt)
    print("--- DEEP RESEARCH RESPONSE ---")
    print(response_text)
    print("------------------------------")

    json_text = (
        response_text.strip().removeprefix("<ctrl_json>").removesuffix("</ctrl_json>")
    )
    try:
        response_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        error_message = f"Failed to parse deep research response as JSON: {e}. Response text: '{response_text}'"
        return DeepResearchPlan(
            thought="System error when parsing the deep research response.",
            action=Action(
                type="SYSTEM_ERROR",
                arguments=SystemErrorArgs(error_message=error_message),
            ),
            best_effort="I encountered a system error and cannot provide an answer.",
        )

    action_data = response_data["action"]
    action_type = action_data["type"]
    args_data = action_data["arguments"]

    arg_class_map = {
        "ANSWER": AnswerArgs,
        "NOTE_DOWN": NoteDownArgs,
        "IRRELEVANT": IrrelevantArgs,
        "SYSTEM_ERROR": SystemErrorArgs,
    }
    arg_class = arg_class_map.get(action_type, lambda **kwargs: kwargs)

    return DeepResearchPlan(
        thought=response_data["thought"],
        action=Action(type=action_type, arguments=arg_class(**args_data)),
        best_effort=response_data.get("best_effort", ""),
    )


def run(
    messages: List[Message], scientist_ids: List[str], thought: str
) -> DeepResearchResponse:
    """
    Performs deep research on specified scientists to answer the user's query.
    """
    print("\n--- STARTING DEEP RESEARCH ---")
    print(f"Initial thought: {thought}")
    print(f"Scientist IDs to research: {scientist_ids}")

    accumulated_notes = []
    last_best_effort = "I'm sorry, I couldn't find a definitive answer."
    scientists_db = {s.rank: s for s in read_scientists_full(scientist_ids)}
    print("Found:", scientists_db.keys())

    for scientist_id in scientist_ids:
        print(f"\n--- Researching Scientist ID: {scientist_id} ---")
        scientist = scientists_db[scientist_id]
        if not scientist or not scientist.full_bio:
            print(
                f"Scientist ID {scientist_id} not found or has no full bio. Skipping."
            )
            continue

        # Simple chunking: split by sentence and group into ~1000 char snippets.
        sentences = scientist.full_bio.split(". ")
        current_snippet = ""
        snippet_count = 0
        for sentence in sentences:
            if len(current_snippet) + len(sentence) > 5000:
                snippet_count += 1
                print(
                    f"\n... Processing snippet {snippet_count} for scientist {scientist_id} ..."
                )
                plan_obj = plan_next_step(
                    messages, accumulated_notes, current_snippet, plan=thought
                )
                action = plan_obj.action
                last_best_effort = plan_obj.best_effort
                print(f"Action taken: {action.type}")

                if action.type == "ANSWER":
                    print("Found answer. Concluding deep research.")
                    return DeepResearchResponse(
                        response=action.arguments.answer,
                        status=DeepResearchStatus.SUCCESS,
                    )
                elif action.type == "NOTE_DOWN":
                    note = action.arguments.note
                    accumulated_notes.append(note)
                    print(
                        f"Note added: '{note}'. Total notes: {len(accumulated_notes)}"
                    )
                elif action.type == "SYSTEM_ERROR":
                    print(
                        f"System error during deep research: {action.arguments.error_message}"
                    )
                    return DeepResearchResponse(
                        response="I encountered a system error during deep research.",
                        status=DeepResearchStatus.FAILURE,
                    )

                current_snippet = ""
            current_snippet += sentence + ". "

        # Process the last remaining snippet.
        if current_snippet:
            snippet_count += 1
            print(
                f"\n... Processing final snippet {snippet_count} for scientist {scientist_id} ..."
            )
            plan_obj = plan_next_step(
                messages, accumulated_notes, current_snippet, plan=thought
            )
            if plan_obj.best_effort:
                last_best_effort = plan_obj.best_effort
            print(f"Action taken: {plan_obj.action.type}")
            if plan_obj.action.type == "ANSWER":
                print("Found answer in final snippet. Concluding deep research.")
                return DeepResearchResponse(
                    response=plan_obj.action.arguments.answer,
                    status=DeepResearchStatus.SUCCESS,
                )

    # If no answer is found after all snippets, return the last best effort.
    print("\n--- DEEP RESEARCH COMPLETE ---")
    print("No definitive answer found after processing all snippets.")
    print(f"Returning last best effort: {last_best_effort}")
    return DeepResearchResponse(
        response=last_best_effort, status=DeepResearchStatus.BEST_EFFORT
    )
