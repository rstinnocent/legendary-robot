"""
eval_set.py — the fixed evaluation set for the Data Q&A app.

Why this exists
---------------
"Correct" is the whole product for a data Q&A tool. An answer that looks
confident and is wrong is worse than no answer. So instead of spot-checking
by hand, this file pins down 12 representative questions against the HR
sample data, each with an expected answer computed independently in pandas
by hand (see `evals/ground_truth.md` for the workings).

The set is deliberately shaped to cover the failure modes that matter,
not just the easy cases:

  - simple aggregate            (can it do the basics)
  - filter + group-by           (can it slice)
  - two-file join               (acceptance criterion: cross-file analysis)
  - three-file join             (the hard version of the same)
  - a chart question            (acceptance criterion: visual insights)
  - a date-format trap          (exits.csv uses DD-MM-YYYY while the other
                                 files use YYYY-MM-DD — naive parsing gets
                                 this silently wrong, which is exactly the
                                 kind of bug a user would never catch)
  - an unanswerable question    (the data has no performance ratings; the
                                 right answer is to say so, not to invent one)

Each case lists `must_contain`: substrings that have to appear in the
normalised rendering of the answer. Numbers are matched with separators
stripped, so 1168150 matches "1,168,150" and "1168150.0" alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    question: str
    must_contain: list[str]
    capability: str
    expect_chart: bool = False
    expect_refusal: bool = False
    notes: str = ""
    forbidden: list[str] = field(default_factory=list)


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="E01",
        capability="simple aggregate",
        question="How many employees are there in total?",
        must_contain=["40"],
    ),
    EvalCase(
        id="E02",
        capability="simple aggregate",
        question="What is the average annual CTC across all employees?",
        must_contain=["1168150"],
        notes="Exact mean is 1,168,150.0",
    ),
    EvalCase(
        id="E03",
        capability="group-by",
        question="What is the headcount in each department?",
        must_contain=["People Ops", "9", "Engineering", "5"],
        notes="CS 7, Eng 5, Fin 6, Mkt 6, People Ops 9, Sales 7",
    ),
    EvalCase(
        id="E04",
        capability="group-by + argmax",
        question="Which department has the highest average CTC?",
        must_contain=["Engineering"],
        notes="Engineering, avg 1,925,000",
    ),
    EvalCase(
        id="E05",
        capability="filter + count",
        question="How many employees are based in Bengaluru?",
        must_contain=["14"],
    ),
    EvalCase(
        id="E06",
        capability="single-file aggregate over many rows",
        question="What is the overall attendance rate, as days present over working days?",
        must_contain=["95.9"],
        notes="14116 present / 14720 working = 95.90%",
    ),
    EvalCase(
        id="E07",
        capability="cross-file join (2 files)",
        question="How many people from each department have exited the company?",
        must_contain=["Engineering", "2", "Sales", "2"],
        notes="Joins exits.csv -> employees.csv. CS 1, Eng 2, Fin 1, Mkt 1, PO 2, Sales 2",
    ),
    EvalCase(
        id="E08",
        capability="cross-file join (2 files) + comparison",
        question="Compare the average CTC of employees who have exited versus those still with the company.",
        must_contain=["908222", "1243612"],
        notes="Leavers 908,222.22 vs stayers 1,243,612.90 — the interesting finding is leavers earned less",
    ),
    EvalCase(
        id="E09",
        capability="cross-file join (3 files)",
        question="What is the total number of leave days taken by employees who have since exited?",
        must_contain=["51"],
        notes="attendance -> exits join; 51 leave days",
    ),
    EvalCase(
        id="E10",
        capability="date-format trap + join",
        question="How many employees exited in March 2026?",
        must_contain=["2"],
        notes=(
            "exit_date is DD-MM-YYYY. Naive pd.to_datetime with dayfirst=False "
            "misreads 16-03-2026 and this comes out wrong. The correct answer is 2."
        ),
    ),
    EvalCase(
        id="E11",
        capability="chart / visual insight",
        question="Plot the trend of total leave days by month.",
        must_contain=[],
        expect_chart=True,
        notes="Jan 45, Feb 47, Mar 29, Apr 29, May 22, Jun 38 — dips mid-year",
    ),
    EvalCase(
        id="E12",
        capability="unanswerable — should decline, not hallucinate",
        question="What is the average performance rating by department?",
        must_contain=[],
        expect_refusal=True,
        forbidden=["3.", "4."],
        notes=(
            "There is no performance rating column anywhere in the data. "
            "A correct system says so. Inventing a plausible number is the "
            "single worst failure mode for this product."
        ),
    ),
]
