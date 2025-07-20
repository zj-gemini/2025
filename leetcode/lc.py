from typing import List, Optional
import fire
import dataclasses


@dataclasses.dataclass
class InputData:
    str_arg: Optional[str] = None
    num_arg: Optional[int] = None
    str_list: Optional[str] = None
    num_list: Optional[str] = None
    file_content: Optional[str] = None  # Changed from 'file'


def read_input_data(
    str_arg: Optional[str] = None,
    num_arg: Optional[int] = None,
    str_list: Optional[str] = None,
    num_list: Optional[str] = None,
    file: Optional[str] = "./input.txt",
) -> InputData:
    """Reads and constructs InputData from arguments and file."""
    file_content = None
    if file is not None:
        try:
            with open(file, "r") as f:
                content = f.read()
                if content.strip() != "":
                    file_content = content
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found at {file}")

    return InputData(
        str_arg=str_arg,
        str_list=str_list,
        num_list=num_list,
        num_arg=num_arg,
        file_content=file_content,
    )


def cli(
    str_arg: Optional[str] = None,
    num_arg: Optional[int] = None,
    str_list: Optional[str] = None,
    num_list: Optional[str] = None,
    file: Optional[str] = "./input.txt",
) -> None:
    """Processes string, numbers, and file inputs."""

    input_data = read_input_data(
        str_arg=str_arg,
        num_arg=num_arg,
        str_list=str_list,
        num_list=num_list,
        file=file,
    )

    if input_data.str_arg is not None:
        if not isinstance(input_data.str_arg, str):
            raise ValueError("The 'str_arg' argument must be a string.")
        print(f"Single string: {input_data.str_arg}")

    if input_data.str_list is not None:
        _validate_and_print_string_list(input_data.str_list)

    if input_data.num_list is not None:
        _validate_and_print_number_list(input_data.num_list)

    if input_data.num_arg is not None:
        if not isinstance(input_data.num_arg, int):
            raise ValueError("The 'num_arg' argument must be an integer.")
        print(f"Single number: {input_data.num_arg}")

    if input_data.file_content is not None:
        print("File contents:")
        print(input_data.file_content)


def process_input_data(input_data: InputData) -> None:
    """Processes the input data and prints the results.

    Args:
        input_data: An instance of InputData containing the input data.
    """
    return


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


if __name__ == "__main__":
    fire.Fire(cli)
