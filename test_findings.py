import json
from pathlib import Path

import pandas as pd
import pytest

from core import LLMBackend, load_files
from findings import (
    Finding,
    FindingsPhrasingError,
    _join_difference_candidates,
    _largest_gap_candidates,
    _outlier_group_candidates,
    _trend_candidates,
    build_phrasing_prompt,
    compute_findings,
    parse_phrasings,
    phrase_findings,
)
from profiler import detect_join_keys, profile_frames

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"


# ---------------------------------------------------------------------------
# Candidate generators — synthetic data with known answers
# ---------------------------------------------------------------------------

def test_largest_gap_finds_correct_extremes():
    df = pd.DataFrame({
        "department": ["A", "A", "B", "B", "C", "C"],
        "salary": [100, 110, 500, 510, 300, 310],
    })
    frames = {"t": df}
    profiles = profile_frames(frames)
    candidates = _largest_gap_candidates(frames, profiles)
    assert len(candidates) == 1
    n = candidates[0].numbers
    assert n["top_group"] == "B"
    assert n["bottom_group"] == "A"
    assert n["gap"] == pytest.approx(400)


def test_largest_gap_skips_single_group_dimension():
    df = pd.DataFrame({"department": ["A"] * 10, "salary": range(10)})
    frames = {"t": df}
    profiles = profile_frames(frames)
    # department has only 1 distinct value -> not even classified as a dimension
    assert _largest_gap_candidates(frames, profiles) == []


def test_trend_detects_increasing_series():
    df = pd.DataFrame({
        "month": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "amount": [10, 20, 30, 40],
    })
    frames = {"t": df}
    profiles = profile_frames(frames)
    candidates = _trend_candidates(frames, profiles)
    assert len(candidates) == 1
    assert candidates[0].numbers["correlation"] == pytest.approx(1.0, abs=0.01)
    assert "increasing" in candidates[0].description


def test_trend_detects_decreasing_series():
    df = pd.DataFrame({
        "month": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "amount": [40, 30, 20, 10],
    })
    frames = {"t": df}
    profiles = profile_frames(frames)
    candidates = _trend_candidates(frames, profiles)
    assert candidates[0].numbers["correlation"] == pytest.approx(-1.0, abs=0.01)
    assert "decreasing" in candidates[0].description


def test_outlier_group_flags_group_beyond_threshold():
    # 4 tightly-clustered groups (~50) plus one far outlier (~500): with only
    # 3 groups the sample std of the group means is too noisy to clear 1.5
    # SD even for an extreme outlier, so this needs enough groups for the
    # z-score to be a meaningful measure of "far from the pack."
    df = pd.DataFrame({
        "team": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5,
        "score": (
            [50, 51, 49, 50, 50] + [52, 48, 51, 49, 50]
            + [49, 50, 51, 50, 50] + [51, 49, 50, 50, 50]
            + [500, 501, 499, 500, 500]
        ),
    })
    frames = {"t": df}
    profiles = profile_frames(frames)
    candidates = _outlier_group_candidates(frames, profiles)
    assert len(candidates) == 1
    assert candidates[0].numbers["group"] == "E"
    assert abs(candidates[0].numbers["z_score"]) >= 1.5


def test_outlier_group_finds_nothing_when_groups_are_similar():
    df = pd.DataFrame({
        "team": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "score": [50, 51, 49, 50, 50] + [51, 49, 50, 51, 49] + [50, 50, 51, 49, 50],
    })
    frames = {"t": df}
    profiles = profile_frames(frames)
    assert _outlier_group_candidates(frames, profiles) == []


def test_join_difference_computes_correct_means():
    base = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "salary": [100, 200, 300, 400],
    })
    subset = pd.DataFrame({"exit_id": [1, 2], "employee_id": [1, 2]})
    frames = {"employees": base, "exits": subset}
    profiles = profile_frames(frames)
    join_keys = detect_join_keys(frames, profiles)
    candidates = _join_difference_candidates(frames, profiles, join_keys)
    assert len(candidates) == 1
    n = candidates[0].numbers
    assert n["joined_mean"] == pytest.approx(150)   # employees 1,2 -> 100,200
    assert n["unjoined_mean"] == pytest.approx(350)  # employees 3,4 -> 300,400


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_compute_findings_returns_top_n_sorted_by_effect_size():
    df = pd.DataFrame({
        "department": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "salary": [100] * 5 + [110] * 5 + [1000] * 5,  # C is a huge outlier
    })
    frames = {"t": df}
    profiles = profile_frames(frames)
    top = compute_findings(frames, profiles, [], top_n=2)
    assert len(top) <= 2
    if len(top) == 2:
        assert top[0].effect_size >= top[1].effect_size


def test_compute_findings_empty_when_no_signal():
    df = pd.DataFrame({"x": [1]})
    frames = {"t": df}
    profiles = profile_frames(frames)
    assert compute_findings(frames, profiles, []) == []


def test_sample_data_join_difference_matches_hand_computed_eval_ground_truth():
    """Cross-check against evals/eval_set.py E08: leavers 908,222.22 vs
    stayers 1,243,612.90 average CTC."""
    files = {p.name: p.read_bytes() for p in sorted(SAMPLE_DIR.glob("*.csv"))}
    frames = load_files(files)
    profiles = profile_frames(frames)
    join_keys = detect_join_keys(frames, profiles)
    candidates = _join_difference_candidates(frames, profiles, join_keys)
    ctc_finding = next(f for f in candidates if f.numbers["measure"] == "annual_ctc"
                        and f.numbers["base_frame"] == "employees")
    assert ctc_finding.numbers["joined_mean"] == pytest.approx(908222.22, abs=0.5)
    assert ctc_finding.numbers["unjoined_mean"] == pytest.approx(1243612.90, abs=0.5)


# ---------------------------------------------------------------------------
# LLM phrasing — the model rewrites, it must never be trusted with numbers
# ---------------------------------------------------------------------------

class FakeBackend(LLMBackend):
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def _sample_findings():
    """Two hand-built findings — deliberately not derived from compute_findings,
    so the count here is fixed and doesn't depend on how many candidates a
    synthetic dataframe happens to produce."""
    return [
        Finding(kind="largest_gap", description="A beats B by 400.", effect_size=3.0,
                numbers={"gap": 400}),
        Finding(kind="trend", description="Leave days are falling.", effect_size=1.5,
                numbers={"correlation": -0.8}),
    ]


def test_parse_phrasings_valid():
    result = parse_phrasings('["one.", "two."]', 2)
    assert result == ["one.", "two."]


def test_parse_phrasings_wrong_count_raises():
    with pytest.raises(FindingsPhrasingError):
        parse_phrasings('["only one."]', 2)


def test_parse_phrasings_invalid_json_raises():
    with pytest.raises(FindingsPhrasingError):
        parse_phrasings("not json", 1)


def test_phrase_findings_uses_model_output_when_valid():
    findings = _sample_findings()
    custom = [f"Custom phrasing {i}" for i in range(len(findings))]
    backend = FakeBackend(json.dumps(custom))
    result = phrase_findings(findings, backend)
    assert result == custom


def test_phrase_findings_falls_back_to_description_on_garbage_response():
    findings = _sample_findings()
    backend = FakeBackend("complete garbage, not json")
    result = phrase_findings(findings, backend)
    assert result == [f.description for f in findings]


def test_phrase_findings_falls_back_on_wrong_length_response():
    findings = _sample_findings()
    backend = FakeBackend('["only one string"]')
    result = phrase_findings(findings, backend)
    assert result == [f.description for f in findings]


def test_phrase_findings_empty_input_returns_empty():
    backend = FakeBackend("[]")
    assert phrase_findings([], backend) == []


def test_build_phrasing_prompt_includes_every_finding_description():
    findings = _sample_findings()
    prompt = build_phrasing_prompt(findings)
    for f in findings:
        assert f.description in prompt
