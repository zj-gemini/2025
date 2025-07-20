import dataclasses
from typing import List


@dataclasses.dataclass
class InputData:
    str_arg: str | None = None
    num_arg: int | None = None
    strs: List[str] | None = None  # changed from str_list
    ints: List[int] | None = None  # changed from num_list
    file_content: str | None = None


def display_input_data(input_data: InputData) -> None:
    """Processes the input data and prints the results.

    Args:
        input_data: An instance of InputData containing the input data.
    """
    if input_data.str_arg is not None:
        print(f"Single string: {input_data.str_arg}")

    if input_data.strs is not None:
        print(f"List of strings: {input_data.strs}")

    if input_data.ints is not None:
        print(f"List of numbers: {input_data.ints}")

    if input_data.num_arg is not None:
        print(f"Single number: {input_data.num_arg}")

    if input_data.file_content is not None:
        print("File contents:")
        print(input_data.file_content)

    return
