import fire
import dataclasses
from typing import Any
import input_parser
from input_data import InputData, display_input_data


def cli(
    str_arg: str | None = None,
    num_arg: int | None = None,
    str_list: str | None = None,
    num_list: str | None = None,
    file: str | None = "./input.txt",
) -> None:
    """Processes string, numbers, and file inputs."""

    input_data = input_parser.read_input_data(
        str_arg=str_arg,
        num_arg=num_arg,
        str_list=str_list,
        num_list=num_list,
        file=file,
    )

    display_input_data(input_data)


if __name__ == "__main__":
    fire.Fire(cli)
