import dataclasses
from typing_extensions import override
from abc import ABC, abstractmethod
from typing import Any, TypeVar, List, Generic
from input_data import InputData


# TypeVar for list element types
T = TypeVar("T", int, float, str)


class ListParser(ABC, Generic[T]):
    """Abstract base class for parsing lists of values."""

    @abstractmethod
    def _convert_item(self, item) -> T:
        pass

    def parse(self, input_value: Any | None) -> List[T] | None:
        if input_value is None:
            return None
        if isinstance(input_value, str):
            input_value = input_value.split(",")
        elif not isinstance(input_value, (list, tuple)):
            raise ValueError(
                f"Invalid type for input, expected 'str', 'list', or 'tuple', got '{type(input_value)}'"
            )
        return [self._convert_item(item) for item in input_value if item]


class IntListParser(ListParser[int]):
    def _convert_item(self, item) -> int:
        return int(item)


class FloatListParser(ListParser[float]):
    def _convert_item(self, item) -> float:
        return float(item)


class StrListParser(ListParser[str]):
    @override
    def _convert_item(self, item) -> str:
        return item.strip()  # No conversion needed, just strip whitespace


def read_input_data(
    str_arg,
    num_arg,
    str_list,
    num_list,
    file,
) -> InputData:
    """Reads and constructs InputData from arguments and file."""
    file_content = None
    if file is not None:
        if not isinstance(file, str):
            raise ValueError(
                f"Invalid type for 'file', expected 'str', got '{type(file)}'"
            )
        try:
            with open(file, "r") as f:
                content = f.read()
                if content.strip() != "":
                    file_content = content
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found at {file}")

    # Parse str_arg
    if str_arg is not None:
        if not isinstance(str_arg, str):
            raise ValueError(
                f"Invalid type for 'str_arg', expected 'str', got '{type(str_arg)}'"
            )

    # Parse num_arg
    if num_arg is not None:
        if isinstance(num_arg, str):
            try:
                num_arg = int(num_arg)
            except ValueError as e:
                raise ValueError(
                    f"Invalid value for 'num_arg', expected 'int', got '{num_arg}'. Error: {e}"
                ) from e
        elif not isinstance(num_arg, int):
            raise ValueError(
                f"Invalid type for 'num_arg', expected 'str' or 'int', got '{type(num_arg)}'"
            )

    return InputData(
        str_arg=str_arg,
        num_arg=num_arg,
        strs=StrListParser().parse(str_list),
        ints=IntListParser().parse(num_list),
        file_content=file_content,
    )
