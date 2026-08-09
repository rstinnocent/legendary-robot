"""
chart_plan.py — ask an LLM which charts are worth showing, never what they say.

One model call, one JSON response: a list of chart *specs* drawn from a fixed
recipe set (see RECIPES below). The model's only job is judgement — which
columns are interesting together — not arithmetic. Every spec is validated
against the real schema in `validate_plan` before anything runs, and the
actual numbers are always produced by `charts.py`'s deterministic recipe
functions, never by the model. This mirrors the reasoning in core.py for the
Q&A path (the engine does the arithmetic) but goes one step further: here the
model doesn't even write code, just fills in a template.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core import LLMBackend
from profiler import FrameProfile, JoinKey, ROLE_DATE, ROLE_DIMENSION, ROLE_IDENTIFIER, ROLE_MEASURE

RECIPE_COUNT_BY_DIMENSION = "count_by_dimension"
RECIPE_MEASURE_BY_DIMENSION = "measure_by_dimension"
RECIPE_MEASURE_OVER_TIME = "measure_over_time"
RECIPE_CROSSTAB = "crosstab"
RECIPE_HEADLINE_METRICS = "headline_metrics"
RECIPE_CROSS_FILE_MEASURE = "cross_file_measure"

RECIPES = {
    RECIPE_COUNT_BY_DIMENSION,
    RECIPE_MEASURE_BY_DIMENSION,
    RECIPE_MEASURE_OVER_TIME,
    RECIPE_CROSSTAB,
    RECIPE_HEADLINE_METRICS,
    RECIPE_CROSS_FILE_MEASURE,
}

VALID_AGGS = {"sum", "mean", "count"}

MIN_SPECS = 5
MAX_SPECS = 7          # requested from the model
MAX_VALIDATED_SPECS = 6  # hard cap after validation, per the brief


@dataclass
class ChartSpec:
    recipe: str
    title: str
    frame: str | None = None
    dimension: str | None = None
    dimension2: str | None = None
    measure: str | None = None
    date_col: str | None = None
    agg: str = "count"
    frame2: str | None = None
    join_left: str | None = None
    join_right: str | None = None
    metrics: list[dict[str, Any]] = field(default_factory=list)


class ChartPlanError(Exception):
    """Raised when the model's response can't be parsed as a chart plan at all."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _describe_frame_for_plan(profile: FrameProfile) -> str:
    lines = [f"`{profile.name}` — {profile.n_rows} rows"]
    for col in profile.columns.values():
        if col.role in (ROLE_DIMENSION, ROLE_MEASURE, ROLE_DATE):
            detail = f"    - {col.name} ({col.role}"
            if col.role == ROLE_DIMENSION:
                detail += f", {col.distinct_count} distinct values: {col.sample_values}"
            elif col.role == ROLE_MEASURE:
                detail += f", range {col.min}-{col.max}"
            detail += ")"
            lines.append(detail)
    return "\n".join(lines)


def build_chart_plan_prompt(
    profiles: dict[str, FrameProfile], join_keys: list[JoinKey]
) -> str:
    frames_block = "\n\n".join(_describe_frame_for_plan(p) for p in profiles.values())
    if join_keys:
        joins_block = "\n".join(
            f"  - `{jk.left_frame}.{jk.left_column}` <-> `{jk.right_frame}.{jk.right_column}` "
            f"({jk.overlap_count} shared values)"
            for jk in join_keys
        )
    else:
        joins_block = "  (none detected)"

    return f"""You are a data analyst choosing which charts to show on an \
auto-generated overview screen, before any question has been asked. You do \
NOT compute any numbers yourself — you only pick which charts are worth \
showing, from a fixed set of recipes. A separate deterministic step will do \
the actual math.

Available data:

{frames_block}

Detected cross-file join keys (columns that share real values across files):
{joins_block}

Pick {MIN_SPECS} to {MAX_SPECS} charts from this fixed recipe set. Return \
ONLY a JSON array, no markdown fences, no explanation. Each element is one \
of these shapes:

{{"recipe": "count_by_dimension", "title": "...", "frame": "<frame>", "dimension": "<column>"}}

{{"recipe": "measure_by_dimension", "title": "...", "frame": "<frame>", "dimension": "<column>", "measure": "<column>", "agg": "sum"|"mean"}}

{{"recipe": "measure_over_time", "title": "...", "frame": "<frame>", "date_col": "<column>", "measure": "<column>", "agg": "sum"|"mean"}}

{{"recipe": "crosstab", "title": "...", "frame": "<frame>", "dimension": "<column>", "dimension2": "<column>"}}

{{"recipe": "headline_metrics", "title": "...", "metrics": [{{"label": "...", "frame": "<frame>", "measure": "<column or null for row count>", "agg": "sum"|"mean"|"count"}}]}}

{{"recipe": "cross_file_measure", "title": "...", "frame": "<frame>", "frame2": "<other frame>", "join_left": "<column in frame>", "join_right": "<column in frame2>", "dimension": "<column, from either frame>", "measure": "<column or null for row count>", "agg": "sum"|"mean"|"count"}}

Rules:
- Only reference frame and column names exactly as listed above.
- Only use `dimension` columns for `dimension`/`dimension2`, and only `measure` \
columns for `measure`, and only `date` columns for `date_col`.
- If any join keys are listed above, include at least one `cross_file_measure` \
chart using one of them.
- Include exactly one `headline_metrics` spec with 3-5 metrics summarising the \
data at a glance (counts, totals, averages).
- Prefer charts that would actually be useful to an HR/People Ops analyst \
looking at this data for the first time.
"""


def parse_chart_plan(raw_text: str) -> list[ChartSpec]:
    text = _strip_code_fences(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChartPlanError(f"Model response was not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ChartPlanError("Model response was not a JSON array.")

    specs: list[ChartSpec] = []
    for item in data:
        if not isinstance(item, dict) or "recipe" not in item:
            continue
        known_fields = {f for f in ChartSpec.__dataclass_fields__}
        kwargs = {k: v for k, v in item.items() if k in known_fields}
        specs.append(ChartSpec(**kwargs))
    return specs


def generate_chart_plan(
    profiles: dict[str, FrameProfile], join_keys: list[JoinKey], backend: LLMBackend
) -> list[ChartSpec]:
    prompt = build_chart_plan_prompt(profiles, join_keys)
    raw = backend.generate(prompt)
    return parse_chart_plan(raw)


# ---------------------------------------------------------------------------
# Validation — the model picks, this decides what's actually safe to run.
# ---------------------------------------------------------------------------

def _role(profiles: dict[str, FrameProfile], frame: str, col: str | None) -> str | None:
    if col is None or frame not in profiles:
        return None
    cp = profiles[frame].columns.get(col)
    return cp.role if cp else None


def _is_valid_metric(profiles: dict[str, FrameProfile], metric: dict[str, Any]) -> bool:
    frame = metric.get("frame")
    if frame not in profiles:
        return False
    measure = metric.get("measure")
    agg = metric.get("agg", "count")
    if agg not in VALID_AGGS:
        return False
    if measure is None:
        return agg == "count"
    return _role(profiles, frame, measure) == ROLE_MEASURE


def _validate_one(spec: ChartSpec, profiles: dict[str, FrameProfile]) -> bool:
    """True if `spec` references real columns with the roles each recipe needs."""
    r = spec.recipe

    if r == RECIPE_COUNT_BY_DIMENSION:
        return _role(profiles, spec.frame, spec.dimension) == ROLE_DIMENSION

    if r == RECIPE_MEASURE_BY_DIMENSION:
        return (
            _role(profiles, spec.frame, spec.dimension) == ROLE_DIMENSION
            and _role(profiles, spec.frame, spec.measure) == ROLE_MEASURE
            and spec.agg in ("sum", "mean")
        )

    if r == RECIPE_MEASURE_OVER_TIME:
        return (
            _role(profiles, spec.frame, spec.date_col) == ROLE_DATE
            and _role(profiles, spec.frame, spec.measure) == ROLE_MEASURE
            and spec.agg in ("sum", "mean")
        )

    if r == RECIPE_CROSSTAB:
        return (
            spec.dimension != spec.dimension2
            and _role(profiles, spec.frame, spec.dimension) == ROLE_DIMENSION
            and _role(profiles, spec.frame, spec.dimension2) == ROLE_DIMENSION
        )

    if r == RECIPE_HEADLINE_METRICS:
        valid_metrics = [m for m in spec.metrics if isinstance(m, dict) and _is_valid_metric(profiles, m)]
        spec.metrics = valid_metrics  # drop bad entries in place rather than the whole spec
        return len(valid_metrics) >= 2

    if r == RECIPE_CROSS_FILE_MEASURE:
        if spec.frame not in profiles or spec.frame2 not in profiles or spec.frame == spec.frame2:
            return False
        if _role(profiles, spec.frame, spec.join_left) != ROLE_IDENTIFIER:
            return False
        if _role(profiles, spec.frame2, spec.join_right) != ROLE_IDENTIFIER:
            return False
        dim_role = _role(profiles, spec.frame, spec.dimension) or _role(profiles, spec.frame2, spec.dimension)
        if dim_role != ROLE_DIMENSION:
            return False
        if spec.agg not in VALID_AGGS:
            return False
        if spec.measure is None:
            return spec.agg == "count"
        measure_role = _role(profiles, spec.frame, spec.measure) or _role(profiles, spec.frame2, spec.measure)
        return measure_role == ROLE_MEASURE

    return False


def _find_column_with_role(profile: FrameProfile, role: str) -> str | None:
    cols = profile.columns_with_role(role)
    return cols[0] if cols else None


def _synthesize_cross_file_spec(
    profiles: dict[str, FrameProfile], join_keys: list[JoinKey]
) -> ChartSpec | None:
    """Build one cross_file_measure spec deterministically when the model's
    plan didn't include a valid one, so a join key never goes unused."""
    for jk in join_keys:
        left, right = profiles[jk.left_frame], profiles[jk.right_frame]
        dimension = _find_column_with_role(right, ROLE_DIMENSION) or _find_column_with_role(left, ROLE_DIMENSION)
        if not dimension:
            continue
        measure = _find_column_with_role(right, ROLE_MEASURE) or _find_column_with_role(left, ROLE_MEASURE)
        return ChartSpec(
            recipe=RECIPE_CROSS_FILE_MEASURE,
            title=f"{jk.left_frame} x {jk.right_frame} by {dimension}",
            frame=jk.left_frame, frame2=jk.right_frame,
            join_left=jk.left_column, join_right=jk.right_column,
            dimension=dimension,
            measure=measure, agg="sum" if measure else "count",
        )
    return None


def validate_plan(
    specs: list[ChartSpec], profiles: dict[str, FrameProfile], join_keys: list[JoinKey]
) -> list[ChartSpec]:
    valid = [s for s in specs if _validate_one(s, profiles)]

    has_cross_file = any(s.recipe == RECIPE_CROSS_FILE_MEASURE for s in valid)
    if join_keys and not has_cross_file:
        synthesized = _synthesize_cross_file_spec(profiles, join_keys)
        if synthesized:
            valid.append(synthesized)

    return valid[:MAX_VALIDATED_SPECS]
