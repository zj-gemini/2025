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
    # Those are the expected arguments for the function to implement
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


def evaluate_submission(problem: Problem, submission: Submission) -> Any:
    """Evaluate a solution against its test cases."""
    if not problem.test_cases:
        return 0
    score = 0
    for test_case in problem.test_cases:
        output = xbox(submission.code, test_case.input, problem.function_args)
        if output is not None and output == test_case.expected_output:
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

    func = local_vars.get("solution")
    if not func:
        logger.error("No function 'solution' defined in submission.")
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
