from pathlib import Path

import pandas as pd
import pytest

from core import load_files
from profiler import (
    ROLE_DATE,
    ROLE_DIMENSION,
    ROLE_IDENTIFIER,
    ROLE_MEASURE,
    ROLE_SKIP,
    detect_join_keys,
    profile_column,
    profile_frame,
    profile_frames,
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def test_id_suffix_numeric_column_is_identifier_regardless_of_cardinality():
    s = pd.Series([1, 2, 3, 1, 2], name="employee_id")
    cp = profile_column("employee_id", s)
    assert cp.role == ROLE_IDENTIFIER


def test_bare_id_column_name_is_identifier():
    s = pd.Series(range(20))
    cp = profile_column("id", s)
    assert cp.role == ROLE_IDENTIFIER


def test_near_unique_string_column_is_identifier():
    names = [f"Person {i}" for i in range(20)]
    cp = profile_column("name", pd.Series(names))
    assert cp.role == ROLE_IDENTIFIER


def test_near_unique_numeric_column_is_measure_not_identifier():
    """A continuous numeric column (e.g. salary) is ~100% unique by nature —
    that must not make it an identifier. Regression for the annual_ctc case."""
    ctc = pd.Series([700000 + i * 137 for i in range(40)], name="annual_ctc")
    cp = profile_column("annual_ctc", ctc)
    assert cp.role == ROLE_MEASURE


def test_low_cardinality_numeric_column_is_measure_not_dimension():
    """working_days only takes 3 distinct values (20/21/22) across 240 rows —
    inside the dimension cardinality window, but it's a measure you'd sum."""
    s = pd.Series([20, 21, 22] * 80, name="working_days")
    cp = profile_column("working_days", s)
    assert cp.role == ROLE_MEASURE


def test_low_cardinality_string_column_is_dimension():
    s = pd.Series(["Sales", "Engineering", "Sales", "Finance"] * 10)
    cp = profile_column("department", s)
    assert cp.role == ROLE_DIMENSION


def test_high_cardinality_non_unique_string_column_is_skipped():
    """~30 distinct free-text-ish values over 100 rows: too granular to be a
    useful dimension, not unique enough to be an identifier."""
    values = [f"note {i % 30}" for i in range(100)]
    cp = profile_column("notes", pd.Series(values))
    assert cp.role == ROLE_SKIP


def test_constant_column_is_skipped():
    cp = profile_column("country", pd.Series(["India"] * 10))
    assert cp.role == ROLE_SKIP


def test_mostly_null_column_is_skipped():
    values = [None] * 7 + ["x"] * 3
    cp = profile_column("comment", pd.Series(values))
    assert cp.role == ROLE_SKIP


def test_boolean_column_is_dimension():
    cp = profile_column("is_active", pd.Series([True, False, True, False, True]))
    assert cp.role == ROLE_DIMENSION


# ---------------------------------------------------------------------------
# Date detection — the DD-MM-YYYY vs YYYY-MM-DD disambiguation is the whole
# point of this section; see the module docstring on why exception-based
# dayfirst detection doesn't work reliably.
# ---------------------------------------------------------------------------

def test_iso_dates_are_detected():
    s = pd.Series(["2024-12-20", "2022-04-16", "2021-03-02"])
    cp = profile_column("date_of_joining", s)
    assert cp.role == ROLE_DATE
    assert cp.date_format == "%Y-%m-%d"


def test_ddmmyyyy_dates_are_detected_and_disambiguated_from_mmddyyyy():
    """16-03-2026 is invalid as MM-DD (no month 16), so this column can only
    be DD-MM — the exact date trap documented in sample_data/exits.csv."""
    s = pd.Series(["16-03-2026", "25-03-2026", "18-04-2026", "09-07-2026"])
    cp = profile_column("exit_date", s)
    assert cp.role == ROLE_DATE
    assert cp.date_format == "%d-%m-%Y"
    assert cp.min == pd.Timestamp("2026-03-16")


def test_year_month_dates_are_detected():
    s = pd.Series(["2026-01", "2026-02", "2026-03"])
    cp = profile_column("month", s)
    assert cp.role == ROLE_DATE
    assert cp.date_format == "%Y-%m"


def test_plain_numeric_id_strings_are_not_mistaken_for_dates():
    """pd.to_datetime alone would parse '1001' as the year 1001 — the format-
    candidate approach must not fall for that."""
    s = pd.Series([str(1000 + i) for i in range(10)], name="code")
    cp = profile_column("code", s)
    assert cp.role != ROLE_DATE


# ---------------------------------------------------------------------------
# Frame-level profiling
# ---------------------------------------------------------------------------

def test_profile_frame_covers_every_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    fp = profile_frame("t", df)
    assert set(fp.columns.keys()) == {"a", "b"}
    assert fp.n_rows == 3


def test_columns_with_role_filters_correctly():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "department": ["Sales", "Eng", "Sales", "Eng", "Sales"],
    })
    fp = profile_frame("t", df)
    assert fp.columns_with_role(ROLE_IDENTIFIER) == ["employee_id"]
    assert fp.columns_with_role(ROLE_DIMENSION) == ["department"]


# ---------------------------------------------------------------------------
# Join key detection
# ---------------------------------------------------------------------------

def test_detect_join_keys_finds_shared_entity_id():
    orders = pd.DataFrame({"order_id": [1, 2, 3], "customer_id": [10, 20, 10]})
    customers = pd.DataFrame({"customer_id": [10, 20, 30], "region": ["W", "E", "N"]})
    frames = {"orders": orders, "customers": customers}
    profiles = profile_frames(frames)
    keys = detect_join_keys(frames, profiles)
    assert len(keys) == 1
    assert keys[0].left_column == "customer_id" or keys[0].right_column == "customer_id"


def test_detect_join_keys_ignores_coincidental_row_counter_overlap():
    """Two unrelated row-counters (both starting at 1) must not be reported
    as a join key just because their value ranges happen to overlap."""
    a = pd.DataFrame({"record_id": list(range(1, 241)), "value": range(240)})
    b = pd.DataFrame({"exit_id": list(range(1, 10)), "reason": ["x"] * 9})
    frames = {"a": a, "b": b}
    profiles = profile_frames(frames)
    keys = detect_join_keys(frames, profiles)
    assert keys == []


def test_detect_join_keys_respects_minimum_overlap_count():
    a = pd.DataFrame({"employee_id": [1001, 1002, 1003]})
    b = pd.DataFrame({"employee_id": [1001, 9999, 9998]})
    frames = {"a": a, "b": b}
    profiles = profile_frames(frames)
    keys = detect_join_keys(frames, profiles, min_overlap_count=2)
    assert keys == []  # only 1 shared value


def test_detect_join_keys_finds_nothing_when_no_identifier_columns():
    a = pd.DataFrame({"department": ["Sales", "Eng"] * 5})
    b = pd.DataFrame({"department": ["Sales", "Eng"] * 5})
    frames = {"a": a, "b": b}
    profiles = profile_frames(frames)
    assert detect_join_keys(frames, profiles) == []


# ---------------------------------------------------------------------------
# Integration against the real sample data — locks in the behaviour the
# rest of the auto-analysis feature is built on.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_frames():
    files = {p.name: p.read_bytes() for p in sorted(SAMPLE_DIR.glob("*.csv"))}
    return load_files(files)


def test_sample_data_role_assignment(sample_frames):
    profiles = profile_frames(sample_frames)

    emp = profiles["employees"]
    assert emp.columns["employee_id"].role == ROLE_IDENTIFIER
    assert emp.columns["department"].role == ROLE_DIMENSION
    assert emp.columns["annual_ctc"].role == ROLE_MEASURE
    assert emp.columns["date_of_joining"].role == ROLE_DATE
    assert emp.columns["date_of_joining"].date_format == "%Y-%m-%d"

    att = profiles["attendance"]
    assert att.columns["working_days"].role == ROLE_MEASURE
    assert att.columns["days_present"].role == ROLE_MEASURE
    assert att.columns["month"].role == ROLE_DATE

    ex = profiles["exits"]
    assert ex.columns["exit_date"].role == ROLE_DATE
    assert ex.columns["exit_date"].date_format == "%d-%m-%Y"  # the date trap
    assert ex.columns["exit_type"].role == ROLE_DIMENSION


def test_sample_data_join_keys_include_employee_id_everywhere(sample_frames):
    profiles = profile_frames(sample_frames)
    keys = detect_join_keys(sample_frames, profiles)
    pairs = {(k.left_frame, k.left_column, k.right_frame, k.right_column) for k in keys}
    found_employee_id_pairs = [
        p for p in pairs if p[1] == "employee_id" and p[3] == "employee_id"
    ]
    assert len(found_employee_id_pairs) == 3  # employees-attendance, employees-exits, attendance-exits

    # the record_id/exit_id coincidental overlap must not appear
    assert not any(k.left_column == "record_id" or k.right_column == "record_id"
                   for k in keys if {k.left_frame, k.right_frame} == {"attendance", "exits"})
