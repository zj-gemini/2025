import csv
from dataclasses import dataclass
import json
import argparse
import pathlib


@dataclass
class Scientist:
    """Represents the information for a single scientist."""

    rank: str
    name: str
    field_of_study: str
    era: str
    key_contributions: str
    summary: str = "N/A"
    full_bio: str = "N/A"


def read_scientists_simple() -> list[Scientist]:
    """Returns a list of Scientist objects read from the catalog.csv. No summary and full_bio provided."""
    script_dir = pathlib.Path(__file__).parent.resolve()
    csv_file_path = script_dir / "cataglog.csv"
    scientists = []
    with open(csv_file_path, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if not row.get("Rank") or not row.get("Rank").strip():
                continue  # Skip rows without a rank

            normalized_row = {
                "rank": row.get("Rank", ""),
                "name": row.get("Scientist", ""),
                "field_of_study": row.get("Field of Study", ""),
                "era": row.get("Era/Century", ""),
                "key_contributions": row.get("Key Contribution(s)", ""),
            }
            scientists.append(Scientist(**normalized_row))
    return scientists


def read_scientists_full(ids: list[str] = []) -> list[Scientist]:
    """
    Reads all scientist JSON files from the 'scientists' directory
    and returns a list of Scientist objects with full information.

    Args:
        ids (list[str] | None): A list of rank IDs (as strings) to read.
                                If None, all scientists are read.
    """
    script_dir = pathlib.Path(__file__).parent.resolve()
    # The creation script uses 'scientists', so we read from there.
    scientists_dir = script_dir / "scientists"
    scientists = []

    if not scientists_dir.is_dir():
        print(  # DEBUG
            f"Warning: Directory '{scientists_dir}' does not exist. Cannot load scientists."
        )
        return []

    files_to_read = []
    if ids:
        # If specific IDs are requested, construct the file paths for them.
        files_to_read = sorted([scientists_dir / f"{id.strip()}.json" for id in ids])
        print(f"DEBUG: Reading specific scientist IDs: {ids}")  # DEBUG
    else:
        # Otherwise, glob all JSON files in the directory.
        files_to_read = sorted(scientists_dir.glob("*.json"))
        print(f"DEBUG: Reading all scientists from '{scientists_dir}'")  # DEBUG

    print(f"DEBUG: Found {len(files_to_read)} files to process.")  # DEBUG
    for json_file in files_to_read:
        try:
            print(f"DEBUG: Processing file: {json_file.name}")  # DEBUG
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                scientists.append(Scientist(**data))
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error processing file {json_file.name}: {e}. Skipping.")
        except FileNotFoundError:
            print(f"Warning: File {json_file.name} not found. Skipping.")
    return scientists


def main():
    """
    Main function to read and display scientist information based on IDs
    provided via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Read and display scientist information from the knowledge base."
    )
    parser.add_argument(
        "ids",
        nargs="+",
        metavar="ID",
        help="One or more scientist rank IDs to display.",
    )
    args = parser.parse_args()

    scientists_to_display = read_scientists_full(ids=args.ids)

    if not scientists_to_display:
        print("No information found for the provided IDs.")
        return

    for i, s in enumerate(scientists_to_display):
        if i > 0:
            print("\n" + "=" * 40 + "\n")
        print(f"Rank: {s.rank}")
        print(f"Name: {s.name}")
        print(f"Field of Study: {s.field_of_study}")
        print(f"Era: {s.era}")
        print(f"Key Contributions: {s.key_contributions}")
        print(f"Summary: {s.summary}")
        print(f"Full Bio (excerpt): {s.full_bio[:40]}...")


if __name__ == "__main__":
    # Example usage from project root: python -m knowledge.scientists 1 5 10
    main()
