import os
import fire
from google import genai
from google.genai import types

# Create a genai.Client instance
client = genai.Client()


def get_gemini_response(
    prompt_text: str, model_name: str, thinking_budget: int = 0
) -> str:
    """Sends a prompt to the Gemini model using genai.Client.models.generate_content and returns the response."""
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        )
        response = client.models.generate_content(
            model=model_name, contents=prompt_text, config=config
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"


def main(
    prompt: str,
    model: str = "gemini-2.5-flash",
    thinking_budget: int = 0,
    list_models: bool = False,
) -> None:
    """Main function to handle command-line arguments using fire."""
    if list_models:
        for m in client.models.list():
            print(f"Model Name: {m.name}")
            print(f"  Description: {m.description}")
            print(f"  Input Tokens: {m.input_token_limit}")
            print(f"  Output Tokens: {m.output_token_limit}")
            print(f"  Actions: {m.supported_actions}")
            print("-" * 20)
        return

    print(
        f"Sending prompt to Gemini: '{prompt}' (model: {model}, thinking_budget: {thinking_budget})..."
    )
    response_text = get_gemini_response(prompt, model, thinking_budget)
    print("\n------------------ Gemini Response ------------------")
    print(response_text)
    print("-----------------------------------------------------")


if __name__ == "__main__":
    fire.Fire(main)
