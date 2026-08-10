"""
ui_helpers.py — pure display-logic helpers for the Crosswalk UI (app.py).

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
    RECIPE_HEADLINE_METRICS,
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


# A handful of business/HR terms that should stay uppercase rather than get
# title-cased ("Annual Ctc" is wrong, "Annual CTC" is right); everything not
# in this list gets ordinary title-casing instead of a guess.
_ACRONYMS = {"ctc", "id", "hr", "wfh", "roi", "kpi", "yoy", "mom", "qoq", "ytd", "hris"}
_MINOR_WORDS = {"of", "the", "and", "by", "in", "on", "at", "to", "for", "vs"}


def prettify_column_name(name: str) -> str:
    """'annual_ctc' -> 'Annual CTC', 'date_of_joining' -> 'Date of Joining'.

    Column names are real identifiers (snake_case, sometimes abbreviated)
    everywhere they're used to actually query the data — but everywhere
    they're shown to a business user in a sentence (a suggested question, a
    join-key explanation), the raw identifier reads like a database field,
    not a question a person would ask. This never touches what's sent to
    the Q&A engine, only what's displayed: the model gets the real schema
    regardless of how the question is worded.
    """
    words = name.replace("_", " ").split()
    if not words:
        return name
    pretty = []
    for i, word in enumerate(words):
        low = word.lower()
        if low in _ACRONYMS:
            pretty.append(low.upper())
        elif i > 0 and low in _MINOR_WORDS:
            pretty.append(low)
        else:
            pretty.append(word.capitalize())
    return " ".join(pretty)


def format_metric_value(value) -> str:
    """Comma-format a headline-metric value if it's numeric, else str() it.

    pd.api.types.is_number, not isinstance(value, (int, float)): a sum/mean
    straight out of pandas is numpy.int64/float64, and numpy.int64 is NOT a
    Python int (numpy.float64 is a Python float, inconsistently), so the
    isinstance form silently skipped comma formatting on any metric backed
    by an int64 column.
    """
    return f"{value:,.0f}" if pd.api.types.is_number(value) else str(value)


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """'1 file' / '3 files' / '292 rows' — the header tag renders a fresh
    count on every rerun (files added, files removed), so this can't be
    hand-written per call site the way a static label could."""
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {word}"


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


def _agg_word(agg: str) -> str:
    return {"mean": "average", "sum": "total", "count": "number of"}.get(agg, agg)


def question_for_spec(spec) -> str | None:
    """The plain-English question a chart card's 'ask about this' action
    feeds into the Q&A engine — a restatement of what the chart already
    shows, not a new claim, phrased the way a business user would actually
    ask it rather than as a database query. None for recipes with nothing
    sensible to ask (headline_metrics is a set of numbers, not a single
    question)."""
    if spec.recipe == RECIPE_COUNT_BY_DIMENSION:
        return f"What's the breakdown by {prettify_column_name(spec.dimension)}?"
    if spec.recipe == RECIPE_MEASURE_BY_DIMENSION:
        return (f"What's the {_agg_word(spec.agg)} {prettify_column_name(spec.measure)} "
                f"by {prettify_column_name(spec.dimension)}?")
    if spec.recipe == RECIPE_MEASURE_OVER_TIME:
        return f"How has {prettify_column_name(spec.measure)} changed over time?"
    if spec.recipe == RECIPE_CROSSTAB:
        return f"How does {prettify_column_name(spec.dimension)} break down by {prettify_column_name(spec.dimension2)}?"
    if spec.recipe == RECIPE_CROSS_FILE_MEASURE:
        measure_part = (f"{_agg_word(spec.agg)} {prettify_column_name(spec.measure)}"
                         if spec.measure else "number of matching records")
        return (f"What's the {measure_part} by {prettify_column_name(spec.dimension)}, "
                f"combining {prettify_column_name(spec.frame)} and {prettify_column_name(spec.frame2)}?")
    if spec.recipe == RECIPE_HEADLINE_METRICS:
        return None
    return None


def describe_join(jk) -> str:
    """'Attendance and Employees can be linked by Employee ID (40 matching
    records)' instead of the raw 'attendance.employee_id ->
    employees.employee_id' — a business user reads file/column names, not
    dot notation."""
    left = prettify_column_name(jk.left_frame)
    right = prettify_column_name(jk.right_frame)
    if jk.left_column == jk.right_column:
        key_desc = prettify_column_name(jk.left_column)
    else:
        key_desc = f"{prettify_column_name(jk.left_column)} / {prettify_column_name(jk.right_column)}"
    return f"{left} and {right} can be linked by {key_desc} ({pluralize(jk.overlap_count, 'matching record')})"


def chart_section_label(titles: list[str], max_named: int = 2) -> str:
    """Label for the collapsed charts expander — names the first few charts
    and ends with an explicit invitation ('click to view') rather than a
    neutral count. A plain "Charts" label wasn't enough of a click
    affordance on its own — a user reported missing the whole section,
    not realizing it was clickable — so the label itself has to carry
    that signal, keeping the section collapsed by default rather than
    always showing a chart up front."""
    if not titles:
        return "📊 Charts"
    shown = ", ".join(titles[:max_named])
    remainder = len(titles) - max_named
    tail = f" +{remainder} more" if remainder > 0 else ""
    return f"📊 {pluralize(len(titles), 'chart')}: {shown}{tail} — click to view"


def schema_section_label(profiles: dict[str, FrameProfile], join_keys: list[JoinKey],
                          max_named: int = 3) -> str:
    """Label for the collapsed file-schema expander: name the files and say
    whether Crosswalk found a way to link them, so someone can tell it's
    worth a click (or that it isn't, and they can go straight to asking a
    question) without opening a wall of per-column tables first."""
    names = list(profiles.keys())
    if not names:
        return "📋 Files"
    shown = ", ".join(names[:max_named])
    remainder = len(names) - max_named
    tail = f" +{remainder} more" if remainder > 0 else ""
    if len(names) <= 1:
        link_tail = ""
    elif not join_keys:
        link_tail = " · no shared columns found yet"
    elif len(join_keys) == 1:
        link_tail = f" · linked by {prettify_column_name(join_keys[0].left_column)}"
    else:
        link_tail = f" · {pluralize(len(join_keys), 'link')} found"
    return f"📋 {pluralize(len(names), 'file')}: {shown}{tail}{link_tail}"


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
    real dimension/measure/date that exists in the data, phrased the way a
    business user would ask it rather than as a database query (no raw
    column names, no "join").

    A cross-file question is generated first, ahead of single-file ones:
    cross-file analysis is the app's differentiator, so with only
    `max_suggestions` slots it shouldn't be the one that gets truncated.
    """
    suggestions: list[str] = []

    if join_keys:
        jk = join_keys[0]
        left = prettify_column_name(jk.left_frame)
        right = prettify_column_name(jk.right_frame)
        suggestions.append(f"How many {left} records have a matching entry in {right}?")

    for profile in profiles.values():
        dims = profile.columns_with_role(ROLE_DIMENSION)
        measures = profile.columns_with_role(ROLE_MEASURE)
        if dims and measures:
            suggestions.append(
                f"What's the average {prettify_column_name(measures[0])} "
                f"by {prettify_column_name(dims[0])}?"
            )
            break

    for profile in profiles.values():
        dims = profile.columns_with_role(ROLE_DIMENSION)
        if dims:
            suggestions.append(f"What's the breakdown by {prettify_column_name(dims[0])}?")
            break

    for profile in profiles.values():
        dates = profile.columns_with_role(ROLE_DATE)
        measures = profile.columns_with_role(ROLE_MEASURE)
        if dates and measures:
            suggestions.append(f"How has {prettify_column_name(measures[0])} changed over time?")
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
