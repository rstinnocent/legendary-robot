"""
ui_helpers.py — pure display-logic helpers for the Insight UI (app.py).

Kept separate from app.py, and from Streamlit, on purpose: every function
here is plain Python over the existing engine's output types (ColumnProfile,
ChartResult, ExecutionResult), so this module is unit-testable the same way
core.py/profiler.py/findings.py already are. None of it calls an LLM or
computes a number that isn't already sitting in the object it's summarising
— a chart caption restates the chart's own table, it doesn't recompute
anything.
"""

from __future__ import annotations

import re

import pandas as pd

from chart_plan import (
    RECIPE_COUNT_BY_DIMENSION,
    RECIPE_CROSS_FILE_MEASURE,
    RECIPE_CROSSTAB,
    RECIPE_MEASURE_BY_DIMENSION,
    RECIPE_MEASURE_OVER_TIME,
)
from profiler import FrameProfile, JoinKey, ROLE_DATE, ROLE_DIMENSION, ROLE_IDENTIFIER, ROLE_MEASURE

# ---------------------------------------------------------------------------
# Schema chips
# ---------------------------------------------------------------------------

_ROLE_CHIP = {
    ROLE_IDENTIFIER: "id",
    ROLE_DIMENSION: "category",
    ROLE_MEASURE: "number",
    ROLE_DATE: "date",
}


def type_chip(role: str) -> str:
    """Short schema-preview label for a column role. `skip` and anything
    unrecognised render as free text — there's no more specific claim to make."""
    return _ROLE_CHIP.get(role, "text")


# ---------------------------------------------------------------------------
# Chart captions — one deterministic line per chart, from its own table
# ---------------------------------------------------------------------------

def chart_caption(spec, chart_result) -> str:
    """A one-line takeaway for a chart card, computed from the chart's own
    result table — never a new query. Empty string if there's nothing safe
    to say (an error, or a recipe this function doesn't have a line for)."""
    if chart_result.error or chart_result.table is None:
        return ""

    table = chart_result.table

    if spec.recipe in (RECIPE_COUNT_BY_DIMENSION, RECIPE_MEASURE_BY_DIMENSION, RECIPE_CROSS_FILE_MEASURE):
        if not isinstance(table, pd.Series) or table.empty:
            return ""
        top_idx, top_val = table.idxmax(), table.max()
        total = table.sum()
        if spec.recipe == RECIPE_COUNT_BY_DIMENSION and total:
            share = 100 * top_val / total
            return f"{top_idx} highest at {top_val:,.0f} ({share:.0f}%)"
        return f"{top_idx} highest at {top_val:,.0f}"

    if spec.recipe == RECIPE_MEASURE_OVER_TIME:
        if not isinstance(table, pd.Series) or len(table) < 2:
            return ""
        first, last = table.iloc[0], table.iloc[-1]
        if last > first:
            direction = "up"
        elif last < first:
            direction = "down"
        else:
            direction = "flat"
        return f"{direction}, from {first:,.0f} to {last:,.0f}"

    if spec.recipe == RECIPE_CROSSTAB:
        if not isinstance(table, pd.DataFrame) or table.empty:
            return ""
        stacked = table.stack()
        (row, col), val = stacked.idxmax(), stacked.max()
        return f"{row} × {col} highest at {val:,.0f}"

    return ""


# ---------------------------------------------------------------------------
# Answer-state classification — drives which of 3e's boxes to render
# ---------------------------------------------------------------------------

STATE_ERROR = "error"
STATE_CALM = "calm"
STATE_NORMAL = "normal"

# Same spirit as evals/run_eval.py's REFUSAL_MARKERS, kept separate on
# purpose: that list grades eval answers against a live model after the
# fact, this one drives which box renders in the running app. Duplicating
# a half-dozen short strings is cheaper than coupling app.py to evals/.
DECLINE_MARKERS = [
    "no ", "not ", "cannot", "can't", "doesn't", "does not", "unavailable",
    "n/a", "unable", "not available", "not found", "no such", "no column",
]


def classify_answer_state(result) -> str:
    """Which of the three 3e boxes an ExecutionResult should render as."""
    if result.error:
        return STATE_ERROR
    if isinstance(result.result, str):
        low = result.result.lower()
        if any(marker in low for marker in DECLINE_MARKERS):
            return STATE_CALM
    return STATE_NORMAL


# ---------------------------------------------------------------------------
# Suggested questions — deterministic, from the profile alone
# ---------------------------------------------------------------------------

def suggested_questions(
    profiles: dict[str, FrameProfile], join_keys: list[JoinKey], max_suggestions: int = 3
) -> list[str]:
    """A handful of questions the uploaded data can actually answer, built
    from column roles rather than invented — every suggestion references a
    real dimension/measure/date that exists in the data.

    A cross-file join question is generated first, ahead of single-file
    ones: cross-file analysis is the app's differentiator, so with only
    `max_suggestions` slots it shouldn't be the one that gets truncated.
    """
    suggestions: list[str] = []

    if join_keys:
        jk = join_keys[0]
        suggestions.append(
            f"How many {jk.left_frame} rows match {jk.right_frame} on {jk.left_column}?"
        )

    for profile in profiles.values():
        dims = profile.columns_with_role(ROLE_DIMENSION)
        measures = profile.columns_with_role(ROLE_MEASURE)
        if dims and measures:
            suggestions.append(f"What is the average {measures[0]} by {dims[0]}?")
            break

    for profile in profiles.values():
        dims = profile.columns_with_role(ROLE_DIMENSION)
        if dims:
            suggestions.append(f"What is the count by {dims[0]}?")
            break

    for profile in profiles.values():
        dates = profile.columns_with_role(ROLE_DATE)
        measures = profile.columns_with_role(ROLE_MEASURE)
        if dates and measures:
            suggestions.append(f"Show the trend of {measures[0]} by {dates[0]}")
            break

    # de-dupe while preserving order (small profiles can produce repeats)
    seen: set[str] = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:max_suggestions]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def dataframe_to_csv_bytes(df) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def safe_download_filename(question: str) -> str:
    """A filesystem-safe .csv filename derived from a question string."""
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    return f"{slug[:50] or 'answer'}.csv"
