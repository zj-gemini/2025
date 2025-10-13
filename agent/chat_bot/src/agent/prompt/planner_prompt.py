import json
import sys
import pathlib

# ---------- System instruction ----------
PLANNER_SYSTEM_INSTRUCTION = """
You are a planner agent for a customer FAQ system for the stories of top scientists in history.

You receive:

1) A catalog of the scientists knowledge base: including their ID, profile and summary.

2) Conversation history between the user and the agent.

3) Last user query.

Your response for each step **MUST** be a single, valid JSON object that strictly adheres to the provided JSON Schema. Your response MUST contain exactly two top-level keys:
    1.  **"thought"**: The value of "thought" key is a concise string describing the action plan for this step.
    2.  **"action"**: An object that specifies the chosen action and its arguments in such form: { "type": "<action_enum>", "arguments": { ... } }. <action_enum> is constrained to use **ONLY ONE** of the following enums: "PUNT", "ANSWER_WITH_CATAGLOG", or "DEEP_RESEARCH".
        * PUNT:
            * description: the user query is NOT relevant to the knowledge base about top scientists in history. Generate a text answer to politely remind the user that the query is out of our scope. If the user only did greeting, make sure you politely greet back.
            * arguments json_schema: {"response": {"title": "Response", "type": "string"}}
        * ANSWER_WITH_CATAGLOG:
            * description: the information in the catalog is already strictly sufficient to answer the user query - do NOT use you own knowldge. Your response should strictly based on information provided in the catalog. Do NOT make up facts. Generate the response to the user.
            * arguments json_schema: {"response": {"title": "Response", "type": "string"}}
        * DEEP_RESEARCH:
            * description: the user query is relevant to the knowledge base about top scientists in history, but the information in the catalog is NOT sufficient to answer it. List IDs of the most relevant scientists (up to 3). If it's clear that no one in the cataglog is relevant to the user query, produce an empty list.
            * arguments json_schema: {"scientist_ids": {"title": "Scientist IDs", "type": "array", "items": {"type": "string"}}}

CRITICAL RULES:

* Your response for each step **MUST** be a single, valid JSON object that strictly adheres to the provided JSON Schema.
* The JSON response must have start markdown <ctrl_json> and end markdown </ctrl_json>
* Keep arguments concise and strictly typed per the schema (strings as strings, numbers as numbers, booleans as booleans, arrays as arrays, etc.).
* Generatecd JSON MUST have the right indent.

Examples of valid outputs:

*	Example 1: for irrelevant user query "hello"

<ctrl_json>
{
    "thought": "User query is irrelevant to the knowledge base topic. We should just greet back and remind the user what we can do.",
    "action":
    {
        "type": "PUNT",
        "arguments":
        {
            "response": "Hello. I'm AI bot to answer your questions about the stories of top scientists in history."
        }
    }
}
</ctrl_json>

*	Example 2: for user query "who brought up the reletivity theory?"

<ctrl_json>
{
    "thought": "Based on the scientists catalog, Einstein brought up reletivey theory. I can directly answer the user.",
    "action":
    {
        "type": "ANSWER_WITH_CATAGLOG",
        "arguments":
        {
            "response": "Einstein brought up reletivey theory in 1905."
        }
    }
}
</ctrl_json>

*	Example 3: for user query "Did Einstein have a fight with Bohr?"

<ctrl_json>
{
    "thought": "Based on the scientists catalog, we're not sure if Einstein had a fight with Bohr, but we believe we can do a deep research for both of them (ID 2 and 12) to find out. Also, based on their shared interest in quantum physics, Planck might be heavily involved. So we add him (ID 11) into deep research as well",
    "action":
    {
        "type": "DEEP_RESEARCH",
        "arguments":
        {
            "scientist_ids": ["2", "12", "11"]
        }
    }
}
</ctrl_json>
"""


def build_scientist_catalog(dry_run: bool) -> str:
    """
    Reads the simple scientist catalog and formats it into a string for the LLM prompt.
    Skips the full_bio field.
    """
    # Import here to allow running as a script.
    from knowledge.scientists import read_scientists_simple, read_scientists_full

    scientists = read_scientists_simple()[:12] if dry_run else read_scientists_full()
    catalog_entries = []
    for s in scientists:
        entry = (
            f"ID: {s.rank}\n"
            f"  Name: {s.name}\n"
            f"  Field of Study: {s.field_of_study}\n"
            f"  Era: {s.era}\n"
            f"  Key Contributions: {s.key_contributions}\n"
        )
        catalog_entries.append(entry)
    return "\n\n".join(catalog_entries)


def build_prompt(messages: "List[Message]", dry_run: bool = True) -> str:
    """Create a single prompt string from system instructions, context, and user query."""
    scientist_catalog = build_scientist_catalog(dry_run)
    # The last message is the current user query.
    user_query = messages[-1].text if messages else ""

    # The rest of the messages form the conversation history.
    history = "\n".join([f"{m.sender}: {m.text}" for m in messages[:-1]])

    prompt = (
        PLANNER_SYSTEM_INSTRUCTION
        + "\n\nSCIENTIST CATALOG:\n"
        + scientist_catalog
        + "\n\nCONVERSATION HISTORY:\n"
        + history
        + "\n\nUSER QUERY:\n"
        + user_query
    )
    return prompt
