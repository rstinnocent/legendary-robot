import pandas as pd
import pytest

from chart_plan import ChartSpec
from charts import ChartResult
from core import ExecutionResult
from profiler import detect_join_keys, profile_frames
from ui_helpers import (
    chart_caption,
    classify_answer_state,
    dataframe_to_csv_bytes,
    safe_download_filename,
    suggested_questions,
    type_chip,
)


# ---------------------------------------------------------------------------
# type_chip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,expected", [
    ("identifier", "id"),
    ("dimension", "category"),
    ("measure", "number"),
    ("date", "date"),
    ("skip", "text"),
    ("something_unknown", "text"),
])
def test_type_chip(role, expected):
    assert type_chip(role) == expected


# ---------------------------------------------------------------------------
# chart_caption
# ---------------------------------------------------------------------------

def test_caption_count_by_dimension_includes_share():
    spec = ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="department")
    table = pd.Series({"Sales": 3, "Engineering": 7}, name="count")
    result = ChartResult(title="t", table=table)
    caption = chart_caption(spec, result)
    assert "Engineering" in caption
    assert "70%" in caption


def test_caption_measure_by_dimension_no_share():
    spec = ChartSpec(recipe="measure_by_dimension", title="t", frame="employees",
                      dimension="department", measure="annual_ctc", agg="mean")
    table = pd.Series({"Sales": 500000, "Engineering": 1900000})
    result = ChartResult(title="t", table=table)
    caption = chart_caption(spec, result)
    assert "Engineering" in caption
    assert "%" not in caption


def test_caption_measure_over_time_direction_up():
    spec = ChartSpec(recipe="measure_over_time", title="t", frame="attendance",
                      date_col="month", measure="leave_days", agg="sum")
    table = pd.Series([10, 20, 30])
    result = ChartResult(title="t", table=table)
    assert chart_caption(spec, result) == "up, from 10 to 30"


def test_caption_measure_over_time_direction_down():
    spec = ChartSpec(recipe="measure_over_time", title="t", frame="attendance",
                      date_col="month", measure="leave_days", agg="sum")
    table = pd.Series([30, 20, 10])
    result = ChartResult(title="t", table=table)
    assert chart_caption(spec, result) == "down, from 30 to 10"


def test_caption_crosstab_finds_max_cell():
    spec = ChartSpec(recipe="crosstab", title="t", frame="employees",
                      dimension="department", dimension2="grade")
    table = pd.DataFrame({"G1": [1, 2], "G2": [5, 1]}, index=["Sales", "Engineering"])
    result = ChartResult(title="t", table=table)
    caption = chart_caption(spec, result)
    assert "Sales" in caption and "G2" in caption
    assert "5" in caption


def test_caption_empty_on_error():
    spec = ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="department")
    result = ChartResult(title="t", error="boom")
    assert chart_caption(spec, result) == ""


def test_caption_empty_when_no_table():
    spec = ChartSpec(recipe="headline_metrics", title="t", metrics=[])
    result = ChartResult(title="t", metrics=[{"label": "x", "value": 1}])
    assert chart_caption(spec, result) == ""


# ---------------------------------------------------------------------------
# classify_answer_state
# ---------------------------------------------------------------------------

def test_classify_error_state():
    res = ExecutionResult(error="KeyError: 'region'")
    assert classify_answer_state(res) == "error"


def test_classify_calm_decline_state():
    res = ExecutionResult(result="There is no performance rating column in this data.")
    assert classify_answer_state(res) == "calm"


def test_classify_normal_numeric_state():
    res = ExecutionResult(result=42)
    assert classify_answer_state(res) == "normal"


def test_classify_normal_short_string_state():
    res = ExecutionResult(result="Engineering")
    assert classify_answer_state(res) == "normal"


def test_classify_normal_dataframe_state():
    res = ExecutionResult(result=pd.DataFrame({"a": [1, 2]}))
    assert classify_answer_state(res) == "normal"


# ---------------------------------------------------------------------------
# suggested_questions — against the real sample-data-shaped profile
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_like_frames():
    employees = pd.DataFrame({
        "employee_id": range(1, 21),
        "department": ["Sales", "Engineering"] * 10,
        "annual_ctc": [500000 + i * 1000 for i in range(20)],
    })
    attendance = pd.DataFrame({
        "record_id": range(1, 21),
        "employee_id": list(range(1, 21)),
        "month": ["2026-01"] * 10 + ["2026-02"] * 10,
        "leave_days": [1, 2] * 10,
    })
    return {"employees": employees, "attendance": attendance}


def test_suggested_questions_reference_real_columns(sample_like_frames):
    profiles = profile_frames(sample_like_frames)
    join_keys = detect_join_keys(sample_like_frames, profiles)
    suggestions = suggested_questions(profiles, join_keys)
    assert 1 <= len(suggestions) <= 3
    for q in suggestions:
        assert isinstance(q, str) and q.strip()


def test_suggested_questions_include_join_question_when_join_exists(sample_like_frames):
    profiles = profile_frames(sample_like_frames)
    join_keys = detect_join_keys(sample_like_frames, profiles)
    assert join_keys  # sanity: this fixture does have a join key
    suggestions = suggested_questions(profiles, join_keys)
    assert any("match" in s.lower() for s in suggestions)


def test_suggested_questions_capped_at_max(sample_like_frames):
    profiles = profile_frames(sample_like_frames)
    join_keys = detect_join_keys(sample_like_frames, profiles)
    suggestions = suggested_questions(profiles, join_keys, max_suggestions=1)
    assert len(suggestions) == 1


def test_suggested_questions_empty_when_nothing_usable():
    frames = {"t": pd.DataFrame({"free_text": ["a very long unique string " + str(i) for i in range(10)]})}
    profiles = profile_frames(frames)
    assert suggested_questions(profiles, []) == []


# ---------------------------------------------------------------------------
# export helpers
# ---------------------------------------------------------------------------

def test_dataframe_to_csv_bytes_round_trips():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_bytes = dataframe_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    assert b"a,b" in csv_bytes


def test_safe_download_filename_slugifies():
    assert safe_download_filename("What's the headcount by dept?") == "what_s_the_headcount_by_dept.csv"


def test_safe_download_filename_handles_empty():
    assert safe_download_filename("???") == "answer.csv"
