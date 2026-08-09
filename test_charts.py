import pandas as pd
import pytest

from chart_plan import ChartSpec
from charts import execute_chart_plan
from profiler import profile_frames


@pytest.fixture
def frames():
    employees = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department": ["Sales", "Sales", "Engineering", "Engineering", "Sales", "Engineering"],
        "grade": ["G1", "G2", "G1", "G2", "G1", "G1"],
        "annual_ctc": [500000, 700000, 900000, 1100000, 600000, 950000],
    })
    exits = pd.DataFrame({
        "exit_id": [1, 2],
        "employee_id": [1, 3],
        "exit_type": ["Voluntary", "Involuntary"],
    })
    attendance = pd.DataFrame({
        "record_id": range(1, 7),
        "employee_id": [1, 1, 2, 2, 3, 3],
        "month": ["2026-01", "2026-02", "2026-01", "2026-02", "2026-01", "2026-02"],
        "leave_days": [1, 2, 0, 3, 1, 1],
    })
    return {"employees": employees, "exits": exits, "attendance": attendance}


@pytest.fixture
def profiles(frames):
    return profile_frames(frames)


def _run(spec, frames, profiles):
    return execute_chart_plan([spec], frames, profiles)[0]


def test_count_by_dimension(frames, profiles):
    spec = ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="department")
    r = _run(spec, frames, profiles)
    assert r.error is None
    assert r.figure is not None
    assert r.table["Sales"] == 3
    assert r.table["Engineering"] == 3


def test_measure_by_dimension_mean(frames, profiles):
    spec = ChartSpec(recipe="measure_by_dimension", title="t", frame="employees",
                      dimension="department", measure="annual_ctc", agg="mean")
    r = _run(spec, frames, profiles)
    assert r.error is None
    assert r.table["Sales"] == pytest.approx((500000 + 700000 + 600000) / 3)
    assert r.table["Engineering"] == pytest.approx((900000 + 1100000 + 950000) / 3)


def test_measure_by_dimension_sum(frames, profiles):
    spec = ChartSpec(recipe="measure_by_dimension", title="t", frame="employees",
                      dimension="department", measure="annual_ctc", agg="sum")
    r = _run(spec, frames, profiles)
    assert r.table["Sales"] == 500000 + 700000 + 600000


def test_measure_over_time_uses_profiler_date_format(frames, profiles):
    """Regression: must reuse the profiler's already-resolved date_format
    rather than re-parsing naively, or the DD-MM-YYYY trap comes back."""
    spec = ChartSpec(recipe="measure_over_time", title="t", frame="attendance",
                      date_col="month", measure="leave_days", agg="sum")
    r = _run(spec, frames, profiles)
    assert r.error is None
    assert len(r.table) == 2  # Jan, Feb
    assert r.table.iloc[0] == 1 + 0 + 1  # Jan: employees 1,2,3
    assert r.table.iloc[1] == 2 + 3 + 1  # Feb


def test_measure_over_time_handles_ddmmyyyy_dates_correctly():
    """Same trap as sample_data/exits.csv, at chart-execution level rather
    than just the profiler level."""
    df = pd.DataFrame({
        "exit_id": [1, 2, 3],
        "exit_date": ["16-03-2026", "25-03-2026", "18-04-2026"],
        "severance": [10000, 20000, 15000],
    })
    frames = {"exits": df}
    profiles = profile_frames(frames)
    spec = ChartSpec(recipe="measure_over_time", title="t", frame="exits",
                      date_col="exit_date", measure="severance", agg="sum")
    r = _run(spec, frames, profiles)
    assert r.error is None
    periods = [str(p) for p in r.table.index]
    assert "2026-03" in periods
    assert "2026-04" in periods
    assert r.table[r.table.index.astype(str) == "2026-03"].iloc[0] == 30000


def test_crosstab(frames, profiles):
    spec = ChartSpec(recipe="crosstab", title="t", frame="employees",
                      dimension="department", dimension2="grade")
    r = _run(spec, frames, profiles)
    assert r.error is None
    assert r.table.loc["Sales", "G1"] == 2
    assert r.table.loc["Engineering", "G2"] == 1


def test_headline_metrics_row_count_and_aggregates(frames, profiles):
    spec = ChartSpec(recipe="headline_metrics", title="t", metrics=[
        {"label": "Employees", "frame": "employees", "measure": None, "agg": "count"},
        {"label": "Total CTC", "frame": "employees", "measure": "annual_ctc", "agg": "sum"},
        {"label": "Avg CTC", "frame": "employees", "measure": "annual_ctc", "agg": "mean"},
    ])
    r = _run(spec, frames, profiles)
    assert r.error is None
    values = {m["label"]: m["value"] for m in r.metrics}
    assert values["Employees"] == 6
    assert values["Total CTC"] == sum([500000, 700000, 900000, 1100000, 600000, 950000])
    assert values["Avg CTC"] == pytest.approx(values["Total CTC"] / 6)


def test_cross_file_measure_count(frames, profiles):
    spec = ChartSpec(recipe="cross_file_measure", title="t", frame="exits", frame2="employees",
                      join_left="employee_id", join_right="employee_id",
                      dimension="department", measure=None, agg="count")
    r = _run(spec, frames, profiles)
    assert r.error is None
    assert r.table["Sales"] == 1  # employee 1
    assert r.table["Engineering"] == 1  # employee 3


def test_cross_file_measure_with_measure_and_agg(frames, profiles):
    spec = ChartSpec(recipe="cross_file_measure", title="t", frame="exits", frame2="employees",
                      join_left="employee_id", join_right="employee_id",
                      dimension="department", measure="annual_ctc", agg="mean")
    r = _run(spec, frames, profiles)
    assert r.error is None
    assert r.table["Sales"] == 500000  # employee 1's CTC
    assert r.table["Engineering"] == 900000  # employee 3's CTC


def test_unknown_recipe_produces_error_not_crash(frames, profiles):
    spec = ChartSpec(recipe="not_a_real_recipe", title="t", frame="employees")
    r = _run(spec, frames, profiles)
    assert r.error is not None
    assert r.figure is None


def test_one_bad_spec_does_not_take_down_the_others(frames, profiles):
    good = ChartSpec(recipe="count_by_dimension", title="good", frame="employees", dimension="department")
    bad = ChartSpec(recipe="measure_by_dimension", title="bad", frame="employees",
                     dimension="department", measure="does_not_exist", agg="sum")
    results = execute_chart_plan([good, bad], frames, profiles)
    assert results[0].error is None
    assert results[0].figure is not None
    assert results[1].error is not None
