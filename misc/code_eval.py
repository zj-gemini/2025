"""
This script evaluates coding problem submissions against their test cases.

Each problem has:
- A description of what needs to be implemented
- Function arguments and their types
- Test cases with inputs and expected outputs

Goal: you need to implement the `evaluate_submission` function that takes a problem and a submission and returns a score.
"""

import json
import logging
from pathlib import Path
from typing import Any
from rich import print
import pydantic
import inspect
import math
import types


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

PROBLEMS_PATH = Path("./data/code_eval_problems.json")
SUBMISSIONS_PATH = Path("./data/code_eval_submissions.json")


class TestCase(pydantic.BaseModel):
    input: Any
    expected_output: Any


class Problem(pydantic.BaseModel):
    problem_id: int
    title: str
    description: str
    function_args: dict[str, str]
    test_cases: list[TestCase]


class Submission(pydantic.BaseModel):
    submission_id: str
    problem_id: int
    code: str


def load_problems() -> list[Problem]:
    with open(PROBLEMS_PATH) as f:
        return [Problem.model_validate(p) for p in json.load(f)["problems"]]


def load_submissions() -> list[Submission]:
    with open(SUBMISSIONS_PATH) as f:
        return [Submission.model_validate(s) for s in json.load(f)["submissions"]]


def compare_outputs(a: Any, b: Any) -> bool:
    """Robust output comparison: handles floats, lists, dicts, etc."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(compare_outputs(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(compare_outputs(a[k], b[k]) for k in a)
    return a == b


def evaluate_submission(problem: Problem, submission: Submission) -> float:
    """Evaluate a solution against its test cases."""
    if not problem.test_cases:
        return 0.0
    score = 0
    for test_case in problem.test_cases:
        output = xbox(submission.code, test_case.input, problem.function_args)
        if output is not None and compare_outputs(output, test_case.expected_output):
            score += 1
    return score / len(problem.test_cases)


def xbox(code: str, input: Any, function_args: dict) -> Any:
    """Execute code with input specified in the argument, return the executed output"""
    if not function_args:
        return None

    local_vars = {}
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        logger.error(f"Syntax or runtime error in user code: {e}", exc_info=True)
        return None

    # Extract the first function object from local_vars
    func = next(
        (v for v in local_vars.values() if isinstance(v, types.FunctionType)), None
    )
    if not func:
        logger.error("No function defined in submission.")
        return None

    # Check function signature matches expected arguments
    sig = inspect.signature(func)
    expected_args = list(function_args.keys())
    func_args = list(sig.parameters.keys())
    if func_args != expected_args:
        logger.error(
            f"Function signature mismatch. Expected arguments: {expected_args}, "
            f"but got: {func_args}"
        )
        return None

    try:
        if isinstance(input, dict):
            return func(**input)
        else:
            return func(input)
    except Exception as e:
        logger.error(f"Error during function execution: {e}", exc_info=True)
        return None


def main() -> None:
    problems = {p.problem_id: p for p in load_problems()}
    submissions = load_submissions()

    for submission in submissions:
        problem = problems[submission.problem_id]
        print("*" * 20)
        logger.info(f"Evaluating (Problem {submission.problem_id})")
        print(problem.model_dump())
        print(submission.model_dump())
        result = evaluate_submission(problem, submission)
        logger.info(f"Problem {submission.problem_id} ({problem.title}):")
        logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
