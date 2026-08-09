import numpy as np
import pandas as pd
import pytest

from chart_plan import ChartSpec
from charts import ChartResult
from core import ExecutionResult
from profiler import JoinKey, detect_join_keys, profile_frames
from ui_helpers import (
    chart_caption,
    chart_section_label,
    classify_answer_state,
    dataframe_to_csv_bytes,
    describe_join,
    format_metric_value,
    pluralize,
    prettify_column_name,
    question_for_spec,
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
# format_metric_value
# ---------------------------------------------------------------------------

def test_format_metric_value_plain_int():
    assert format_metric_value(4672000) == "4,672,000"


def test_format_metric_value_plain_float():
    assert format_metric_value(1168150.0) == "1,168,150"


def test_format_metric_value_numpy_int64_gets_comma_formatted():
    """Regression: a headline metric backed by an int64 column (e.g. sum()
    of annual_ctc) is numpy.int64, which is NOT a Python int — the original
    isinstance(value, (int, float)) check silently fell through to str(),
    rendering totals without thousands separators."""
    total = pd.Series([700000, 800000, 1900000], dtype="int64").sum()
    assert isinstance(total, np.int64)
    assert format_metric_value(total) == "3,400,000"


def test_format_metric_value_numpy_float64():
    mean = pd.Series([700000, 800000, 1900000], dtype="int64").mean()
    assert format_metric_value(mean) == f"{mean:,.0f}"


def test_format_metric_value_non_numeric_passthrough():
    assert format_metric_value("Engineering") == "Engineering"


# ---------------------------------------------------------------------------
# pluralize
# ---------------------------------------------------------------------------

def test_pluralize_singular():
    assert pluralize(1, "file") == "1 file"


def test_pluralize_plural():
    assert pluralize(3, "file") == "3 files"


def test_pluralize_zero_is_plural():
    assert pluralize(0, "file") == "0 files"


def test_pluralize_default_plural_is_singular_plus_s():
    assert pluralize(1, "col") == "1 col"
    assert pluralize(2, "col") == "2 cols"


def test_pluralize_explicit_irregular_plural():
    assert pluralize(2, "box", "boxes") == "2 boxes"


def test_pluralize_large_count_gets_comma_formatted():
    assert pluralize(1234, "row") == "1,234 rows"


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
# question_for_spec — phrased for a business user, not a database query
# ---------------------------------------------------------------------------

def test_question_for_count_by_dimension():
    spec = ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="department")
    assert question_for_spec(spec) == "What's the breakdown by Department?"


def test_question_for_measure_by_dimension_prettifies_and_uses_average():
    spec = ChartSpec(recipe="measure_by_dimension", title="t", frame="employees",
                      dimension="department", measure="annual_ctc", agg="mean")
    assert question_for_spec(spec) == "What's the average Annual CTC by Department?"


def test_question_for_measure_by_dimension_sum_says_total():
    spec = ChartSpec(recipe="measure_by_dimension", title="t", frame="employees",
                      dimension="department", measure="annual_ctc", agg="sum")
    assert question_for_spec(spec) == "What's the total Annual CTC by Department?"


def test_question_for_measure_over_time():
    spec = ChartSpec(recipe="measure_over_time", title="t", frame="attendance",
                      date_col="month", measure="leave_days", agg="sum")
    assert question_for_spec(spec) == "How has Leave Days changed over time?"


def test_question_for_crosstab():
    spec = ChartSpec(recipe="crosstab", title="t", frame="employees",
                      dimension="department", dimension2="grade")
    assert question_for_spec(spec) == "How does Department break down by Grade?"


def test_question_for_cross_file_measure_with_measure_says_combining_not_joining():
    spec = ChartSpec(recipe="cross_file_measure", title="t", frame="exits", frame2="employees",
                      dimension="department", measure="annual_ctc", agg="mean")
    q = question_for_spec(spec)
    assert "Annual CTC" in q and "Exits" in q and "Employees" in q
    assert "combining" in q
    assert "joining" not in q


def test_question_for_cross_file_measure_without_measure():
    spec = ChartSpec(recipe="cross_file_measure", title="t", frame="exits", frame2="employees",
                      dimension="department", measure=None, agg="count")
    q = question_for_spec(spec)
    assert "number of matching records" in q


def test_question_for_headline_metrics_is_none():
    spec = ChartSpec(recipe="headline_metrics", title="t", metrics=[])
    assert question_for_spec(spec) is None


# ---------------------------------------------------------------------------
# prettify_column_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("annual_ctc", "Annual CTC"),
    ("employee_id", "Employee ID"),
    ("date_of_joining", "Date of Joining"),
    ("department", "Department"),
    ("working_days", "Working Days"),
    ("wfh_days", "WFH Days"),
    ("", ""),
])
def test_prettify_column_name(raw, expected):
    assert prettify_column_name(raw) == expected


# ---------------------------------------------------------------------------
# describe_join
# ---------------------------------------------------------------------------

def test_describe_join_same_column_name():
    jk = JoinKey(left_frame="attendance", left_column="employee_id",
                 right_frame="employees", right_column="employee_id",
                 overlap_count=40, overlap_ratio=1.0)
    assert describe_join(jk) == (
        "Attendance and Employees can be linked by Employee ID (40 matching records)"
    )


def test_describe_join_different_column_names():
    jk = JoinKey(left_frame="orders", left_column="cust_id",
                 right_frame="customers", right_column="id",
                 overlap_count=5, overlap_ratio=1.0)
    desc = describe_join(jk)
    assert "Cust ID" in desc and "ID" in desc
    assert "5 matching records" in desc


def test_describe_join_singular_match_count():
    jk = JoinKey(left_frame="a", left_column="k", right_frame="b", right_column="k",
                 overlap_count=1, overlap_ratio=1.0)
    assert "1 matching record" in describe_join(jk)
    assert "1 matching records" not in describe_join(jk)


# ---------------------------------------------------------------------------
# chart_section_label
# ---------------------------------------------------------------------------

def test_chart_section_label_names_titles_within_max():
    label = chart_section_label(["Department Distribution", "Grade Distribution"])
    assert label == "📊 2 charts: Department Distribution, Grade Distribution"


def test_chart_section_label_truncates_with_remainder_count():
    label = chart_section_label(
        ["Department Distribution", "Grade Distribution", "Attendance Over Time", "Exit Type by Reason"]
    )
    assert label == "📊 4 charts: Department Distribution, Grade Distribution +2 more"


def test_chart_section_label_empty():
    assert chart_section_label([]) == "📊 Charts"


def test_chart_section_label_respects_max_named():
    label = chart_section_label(["A", "B", "C"], max_named=1)
    assert label == "📊 3 charts: A +2 more"


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
