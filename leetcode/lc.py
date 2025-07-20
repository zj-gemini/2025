from typing import List, Optional
import fire


def cli(
    str_arg: Optional[str] = None,
    str_list: Optional[str] = None,
    num_list: Optional[str] = None,
    num_arg: Optional[int] = None,
    file: Optional[str] = "./input.txt",
) -> None:
    """Processes string, numbers, and file inputs.

    This function acts as a command-line interface to process various
    inputs, including a single string, a comma-separated string of
    strings, a comma-separated string of numbers, a single number,
    and the path to a file.

    Args:
        str_arg: A single string input.
        str_list: A comma-separated string of strings.
        num_list: A comma-separated string of numbers.
        num_arg: A single integer input.
        file: Path to a file to read from. Defaults to "./input.txt".

    Raises:
        ValueError: If 'str_arg' is not a string or 'num_arg' is not an integer.
        FileNotFoundError: If the provided file path is not found.
    """

    if str_arg is not None:
        if not isinstance(str_arg, str):
            raise ValueError("The 'str_arg' argument must be a string.")
        print(f"Single string: {str_arg}")

    if str_list is not None:
        _validate_and_print_string_list(str_list)

    if num_list is not None:
        _validate_and_print_number_list(num_list)

    if num_arg is not None:
        if not isinstance(num_arg, int):
            raise ValueError("The 'num_arg' argument must be an integer.")
        print(f"Single number: {num_arg}")

    if file is not None:
        _print_file_content(file)
    else:
        print("No file path provided.")


def _validate_and_print_string_list(str_list: str) -> None:
    """Validates and prints a list of strings.

    Args:
        str_list: A comma-separated string of strings.
    """
    strings: List[str] = [s.strip() for s in str_list.split(",")]
    print(f"List of strings: {strings}")


def _validate_and_print_number_list(num_list: str) -> None:
    """Validates and prints a list of numbers.

    Args:
        num_list: A comma-separated string of numbers.

    Raises:
        ValueError: If any element in the input is not a valid integer.
    """
    try:
        numbers: List[int] = [int(n.strip()) for n in num_list.split(",")]
        print(f"List of numbers: {numbers}")
    except ValueError as e:
        raise ValueError(f"Invalid number provided in the list: {e}") from e


def _print_file_content(file: str) -> None:
    """Prints the content of a given file.

    Args:
        file: The path to the file.

    Raises:
        FileNotFoundError: If the provided file is not found.
    """
    try:
        with open(file, "r") as f:
            print("File contents:")
            print(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file}")


if __name__ == "__main__":
    fire.Fire(cli)
