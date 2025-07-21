import fire
import dataclasses
from typing import Any
import input_parser
from input_data import InputData, display_input_data


def cli(
    txt: str | None = None,
    num: float | None = None,
    strs: str | None = None,
    ints: str | None = None,
    floats: str | None = None,
    file: str | None = "./input.txt",
) -> None:
    """Processes string, numbers, and file inputs."""

    input_data = input_parser.read_input_data(
        txt=txt,
        num=num,
        strs=strs,
        ints=ints,
        floats=floats,
        file=file,
    )

    display_input_data(input_data)


if __name__ == "__main__":
    fire.Fire(cli)
