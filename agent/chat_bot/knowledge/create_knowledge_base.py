import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
import pathlib

# Add project root to the Python path to allow importing from src
from scientists import Scientist, read_scientists_simple

script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.api.wiki import get_wiki_summary


def process_scientist_info(scientist: Scientist) -> Scientist:
    """
    Enriches the Scientist object with a summary and full bio from Wikipedia.
    """
    query = f"{scientist.name} in {scientist.era} for {scientist.field_of_study}"
    print(f"Fetching Wikipedia data for {query}...")
    response = get_wiki_summary(query)
    if response.success and response.page:
        scientist.summary = response.page.summary
        scientist.full_bio = response.page.content
        print(f"-> Successfully populated data for {scientist.name}.")
    else:
        print(f"-> Could not fetch data for {scientist.name}. Error: {response.error}")

    # Be a good internet citizen and don't spam the API
    time.sleep(0.5)

    return scientist


def create_json_files_from_csv(ids: list[str] | None = None):
    """
    Reads a CSV file containing data about scientists, and for each row,
    generates a JSON file named after the scientist's rank.

    The script expects 'cataglog.csv' to be in the same directory.
    It will create a 'scientists' subdirectory to store the output JSON files.
    Existing files will be overwritten.

    Args:
        ids (list[str] | None): A list of rank IDs (as strings) to process.
                                If None, all ranks are processed.
    """
    try:
        # Use pathlib for modern, object-oriented path manipulation
        output_dir = script_dir / "scientist_profiles"

        # 1. Ensure the output directory exists.
        #    exist_ok=True prevents an error if the directory already exists.
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory '{output_dir}' is ready.")

        # For efficient lookup, convert the list of ranks to a set.
        ranks_to_process = set(ids) if ids else None

        if ranks_to_process:
            print(
                f"Processing only specified ranks: {', '.join(sorted(ranks_to_process))}"
            )

        # 2. Read simple scientist infos
        all_scientists = read_scientists_simple()
        processed_count = 0

        # 3. Process each scientist object.
        for scientist_obj in all_scientists:
            # If a specific set of ranks is requested, skip others.
            if ranks_to_process and scientist_obj.rank.strip() not in ranks_to_process:
                continue

            processed_scientist = process_scientist_info(scientist_obj)

            json_file_path = output_dir / f"{processed_scientist.rank.strip()}.json"
            json_content = json.dumps(
                asdict(processed_scientist), indent=2, ensure_ascii=False
            )

            # Write the content to the file, overwriting if it exists.
            with open(json_file_path, "w", encoding="utf-8") as outfile:
                outfile.write(json_content)

            processed_count += 1
            print(f"Successfully created/updated {json_file_path.name}")

        print(f"\nProcessing complete. {processed_count} JSON files generated.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create JSON knowledge base files from a CSV of scientists."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="RANK_ID",
        help="A list of Rank IDs to process. If not provided, all ranks will be processed.",
    )
    args = parser.parse_args()
    create_json_files_from_csv(ids=args.only)
