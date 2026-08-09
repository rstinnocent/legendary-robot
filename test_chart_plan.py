import pandas as pd
import pytest

from chart_plan import (
    ChartPlanError,
    ChartSpec,
    MAX_VALIDATED_SPECS,
    build_chart_plan_prompt,
    parse_chart_plan,
    validate_plan,
)
from profiler import detect_join_keys, profile_frames


@pytest.fixture
def frames():
    employees = pd.DataFrame({
        "employee_id": range(1, 21),
        "department": ["Sales", "Engineering"] * 10,
        "annual_ctc": [500000 + i * 10000 for i in range(20)],
    })
    exits = pd.DataFrame({
        "exit_id": range(1, 6),
        "employee_id": [1, 3, 5, 7, 9],
        "exit_type": ["Voluntary", "Involuntary"] * 2 + ["Voluntary"],
    })
    return {"employees": employees, "exits": exits}


@pytest.fixture
def profiles(frames):
    return profile_frames(frames)


@pytest.fixture
def join_keys(frames, profiles):
    return detect_join_keys(frames, profiles)


# ---------------------------------------------------------------------------
# parse_chart_plan
# ---------------------------------------------------------------------------

def test_parse_chart_plan_reads_json_array():
    raw = '[{"recipe": "count_by_dimension", "title": "t", "frame": "employees", "dimension": "department"}]'
    specs = parse_chart_plan(raw)
    assert len(specs) == 1
    assert specs[0].recipe == "count_by_dimension"
    assert specs[0].dimension == "department"


def test_parse_chart_plan_strips_markdown_fences():
    raw = '```json\n[{"recipe": "headline_metrics", "title": "t", "metrics": []}]\n```'
    specs = parse_chart_plan(raw)
    assert len(specs) == 1


def test_parse_chart_plan_ignores_unknown_fields():
    raw = '[{"recipe": "count_by_dimension", "title": "t", "frame": "employees", "dimension": "department", "made_up_field": 123}]'
    specs = parse_chart_plan(raw)
    assert len(specs) == 1  # doesn't blow up on the unexpected key


def test_parse_chart_plan_skips_non_dict_items():
    raw = '[{"recipe": "count_by_dimension", "frame": "employees", "dimension": "department", "title": "t"}, "not a spec", 42]'
    specs = parse_chart_plan(raw)
    assert len(specs) == 1


def test_parse_chart_plan_rejects_invalid_json():
    with pytest.raises(ChartPlanError):
        parse_chart_plan("not json at all {{{")


def test_parse_chart_plan_rejects_non_array_json():
    with pytest.raises(ChartPlanError):
        parse_chart_plan('{"recipe": "count_by_dimension"}')


# ---------------------------------------------------------------------------
# validate_plan — per-recipe structural checks
# ---------------------------------------------------------------------------

def test_valid_count_by_dimension_survives(profiles, join_keys):
    specs = [ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="department")]
    out = validate_plan(specs, profiles, join_keys)
    assert any(s.recipe == "count_by_dimension" for s in out)


def test_nonexistent_column_is_dropped(profiles, join_keys):
    specs = [ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="not_a_real_column")]
    out = validate_plan(specs, profiles, join_keys)
    assert not any(s.recipe == "count_by_dimension" and s.dimension == "not_a_real_column" for s in out)


def test_identifier_used_as_dimension_is_dropped(profiles, join_keys):
    specs = [ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="employee_id")]
    out = validate_plan(specs, profiles, join_keys)
    assert out == [] or all(s.dimension != "employee_id" for s in out if s.recipe == "count_by_dimension")


def test_measure_by_dimension_requires_real_measure_and_dimension(profiles, join_keys):
    good = ChartSpec(recipe="measure_by_dimension", title="t", frame="employees",
                      dimension="department", measure="annual_ctc", agg="mean")
    bad_measure = ChartSpec(recipe="measure_by_dimension", title="t", frame="employees",
                             dimension="department", measure="department", agg="mean")  # dimension, not measure
    out = validate_plan([good, bad_measure], profiles, join_keys)
    recipes = [(s.recipe, s.measure) for s in out]
    assert ("measure_by_dimension", "annual_ctc") in recipes
    assert ("measure_by_dimension", "department") not in recipes


def test_crosstab_rejects_same_column_twice(profiles, join_keys):
    specs = [ChartSpec(recipe="crosstab", title="t", frame="employees",
                        dimension="department", dimension2="department")]
    out = validate_plan(specs, profiles, join_keys)
    assert not any(s.recipe == "crosstab" for s in out)


def test_headline_metrics_drops_bad_entries_keeps_good_ones(profiles, join_keys):
    spec = ChartSpec(recipe="headline_metrics", title="t", metrics=[
        {"label": "Employees", "frame": "employees", "measure": None, "agg": "count"},
        {"label": "Avg CTC", "frame": "employees", "measure": "annual_ctc", "agg": "mean"},
        {"label": "Bad", "frame": "employees", "measure": "department", "agg": "mean"},  # not a measure
        {"label": "BadFrame", "frame": "nope", "measure": None, "agg": "count"},
    ])
    out = validate_plan([spec], profiles, join_keys)
    hm = [s for s in out if s.recipe == "headline_metrics"]
    assert len(hm) == 1
    assert len(hm[0].metrics) == 2


def test_headline_metrics_dropped_entirely_if_fewer_than_two_valid_metrics(profiles, join_keys):
    spec = ChartSpec(recipe="headline_metrics", title="t", metrics=[
        {"label": "Employees", "frame": "employees", "measure": None, "agg": "count"},
    ])
    out = validate_plan([spec], profiles, join_keys)
    assert not any(s.recipe == "headline_metrics" for s in out)


def test_cross_file_measure_requires_identifier_join_columns(profiles, join_keys):
    good = ChartSpec(recipe="cross_file_measure", title="t", frame="exits", frame2="employees",
                      join_left="employee_id", join_right="employee_id",
                      dimension="department", measure=None, agg="count")
    bad = ChartSpec(recipe="cross_file_measure", title="t", frame="exits", frame2="employees",
                     join_left="exit_type", join_right="department",  # neither is an identifier
                     dimension="department", measure=None, agg="count")
    out = validate_plan([good, bad], profiles, join_keys)
    cfm = [s for s in out if s.recipe == "cross_file_measure"]
    assert any(s.join_left == "employee_id" for s in cfm)
    assert not any(s.join_left == "exit_type" for s in cfm)


def test_unknown_recipe_is_dropped(profiles, join_keys):
    specs = [ChartSpec(recipe="make_up_a_pie_chart", title="t", frame="employees")]
    out = validate_plan(specs, profiles, join_keys)
    assert not any(s.recipe == "make_up_a_pie_chart" for s in out)


# ---------------------------------------------------------------------------
# Forcing a cross-file chart + capping
# ---------------------------------------------------------------------------

def test_cross_file_chart_is_injected_when_join_exists_but_model_omitted_one(profiles, join_keys):
    assert join_keys  # sanity: this fixture does have a join key
    specs = [ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="department")]
    out = validate_plan(specs, profiles, join_keys)
    assert any(s.recipe == "cross_file_measure" for s in out)


def test_no_cross_file_chart_injected_when_no_join_keys(profiles):
    specs = [ChartSpec(recipe="count_by_dimension", title="t", frame="employees", dimension="department")]
    out = validate_plan(specs, profiles, [])
    assert not any(s.recipe == "cross_file_measure" for s in out)


def test_validate_plan_caps_at_max_validated_specs(profiles, join_keys):
    specs = [
        ChartSpec(recipe="count_by_dimension", title=f"t{i}", frame="employees", dimension="department")
        for i in range(10)
    ]
    out = validate_plan(specs, profiles, join_keys)
    assert len(out) <= MAX_VALIDATED_SPECS


# ---------------------------------------------------------------------------
# Prompt construction — light sanity check, not a snapshot test
# ---------------------------------------------------------------------------

def test_prompt_mentions_every_frame_and_join_key(profiles, join_keys):
    prompt = build_chart_plan_prompt(profiles, join_keys)
    assert "employees" in prompt
    assert "exits" in prompt
    assert "annual_ctc" in prompt
    for jk in join_keys:
        assert jk.left_column in prompt


def test_prompt_says_no_join_keys_when_none_detected(profiles):
    prompt = build_chart_plan_prompt(profiles, [])
    assert "none detected" in prompt
