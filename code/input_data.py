import dataclasses
from typing import List


@dataclasses.dataclass
class InputData:
    txt: str | None = None
    num: float | None = None
    strs: List[str] | None = None
    ints: List[int] | None = None
    floats: List[float] | None = None
    file_content: str | None = None


def display_input_data(input_data: InputData) -> None:
    """Processes the input data and prints the results.

    Args:
        input_data: An instance of InputData containing the input data.
    """
    if input_data.txt is not None:
        print(f"Single string: {input_data.txt}")

    if input_data.strs is not None:
        print(f"List of strings: {input_data.strs}")

    if input_data.ints is not None:
        print(f"List of integers: {input_data.ints}")

    if input_data.floats is not None:
        print(f"List of floats: {input_data.floats}")

    if input_data.num is not None:
        print(f"Single number: {input_data.num}")

    if input_data.file_content is not None:
        print("File content starts" + "-" * 20)
        print(input_data.file_content)
        print("File content ends" + "-" * 20)

    return
