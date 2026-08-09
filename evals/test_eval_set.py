"""
test_eval_set.py — tests for the eval set itself.

An eval set you haven't validated is just a list of opinions. This suite
feeds each case a hand-written, known-correct pandas solution through the
real execution pipeline and asserts the grader marks it PASS. That proves
two things without spending a single LLM call:

  1. the expected answers in eval_set.py are actually correct for the
     sample data (a typo'd ground truth would fail here), and
  2. the grader's matching logic isn't so strict that a right answer
     gets marked wrong.

Run with:  pytest evals/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import run_generated_code  # noqa: E402
from eval_set import EVAL_CASES  # noqa: E402
from run_eval import grade, load_sample_frames  # noqa: E402

# Known-good solutions, written by hand against the sample data.
REFERENCE_SOLUTIONS: dict[str, str] = {
    "E01": "result = len(employees)",
    "E02": "result = employees['annual_ctc'].mean()",
    "E03": "result = employees['department'].value_counts().sort_index()",
    "E04": "result = employees.groupby('department')['annual_ctc'].mean().idxmax()",
    "E05": "result = (employees['location'] == 'Bengaluru').sum()",
    "E06": (
        "rate = 100 * attendance['days_present'].sum() / attendance['working_days'].sum()\n"
        "result = round(rate, 2)"
    ),
    "E07": (
        "m = exits.merge(employees, on='employee_id')\n"
        "result = m['department'].value_counts().sort_index()"
    ),
    "E08": (
        "left = employees[employees['employee_id'].isin(exits['employee_id'])]\n"
        "stayed = employees[~employees['employee_id'].isin(exits['employee_id'])]\n"
        "result = pd.Series({'exited': left['annual_ctc'].mean(),\n"
        "                    'still_employed': stayed['annual_ctc'].mean()})"
    ),
    "E09": (
        "m = attendance.merge(exits[['employee_id']], on='employee_id')\n"
        "result = m['leave_days'].sum()"
    ),
    "E10": (
        "d = pd.to_datetime(exits['exit_date'], format='%d-%m-%Y')\n"
        "result = int(((d.dt.year == 2026) & (d.dt.month == 3)).sum())"
    ),
    "E11": (
        "s = attendance.groupby('month')['leave_days'].sum()\n"
        "plt.plot(s.index, s.values, marker='o')\n"
        "plt.title('Total leave days by month')\n"
        "result = 'Leave days peak in February and dip in May.'"
    ),
    "E12": (
        "result = 'No performance rating column exists in the uploaded files, "
        "so this question cannot be answered from this data.'"
    ),
}


@pytest.fixture(scope="module")
def frames():
    return load_sample_frames()


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.id)
def test_reference_solution_passes_the_grader(case, frames):
    code = REFERENCE_SOLUTIONS[case.id]
    res = run_generated_code(code, frames)
    assert res.error is None, f"{case.id} reference solution errored: {res.error}"
    ok, detail = grade(case, res)
    assert ok, f"{case.id} ground truth appears wrong — grader said: {detail}"


def test_every_case_has_a_reference_solution():
    missing = [c.id for c in EVAL_CASES if c.id not in REFERENCE_SOLUTIONS]
    assert not missing, f"eval cases with no reference solution: {missing}"


def test_grader_catches_a_wrong_answer(frames):
    """Sanity check in the other direction: a plausible-but-wrong answer must FAIL."""
    case = next(c for c in EVAL_CASES if c.id == "E01")
    res = run_generated_code("result = 37", frames)  # true headcount is 40
    ok, _ = grade(case, res)
    assert not ok


def test_grader_catches_hallucination_on_unanswerable(frames):
    """E12 has no rating column; inventing '3.9' must be graded as a failure."""
    case = next(c for c in EVAL_CASES if c.id == "E12")
    res = run_generated_code("result = 'Average performance rating is 3.9'", frames)
    ok, _ = grade(case, res)
    assert not ok
