"""
charts.py — deterministic execution of a validated chart plan.

No generated code and no LLM anywhere in this module. Each recipe named in
chart_plan.py maps to one small parameterised function here that does exactly
what its name says with plain pandas/matplotlib — the model chose *which*
chart, this module is the only thing that touches the numbers.

Date columns are re-parsed here using the `date_format` the profiler already
worked out (`profiler._try_parse_dates`), rather than re-guessing with a bare
`pd.to_datetime`. Re-guessing would reintroduce exactly the DD-MM-YYYY vs
MM-DD-YYYY ambiguity the profiler was built to resolve once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chart_plan import (
    ChartSpec,
    RECIPE_COUNT_BY_DIMENSION,
    RECIPE_CROSS_FILE_MEASURE,
    RECIPE_CROSSTAB,
    RECIPE_HEADLINE_METRICS,
    RECIPE_MEASURE_BY_DIMENSION,
    RECIPE_MEASURE_OVER_TIME,
)
from profiler import FrameProfile


@dataclass
class ChartResult:
    title: str
    figure: Any = None
    table: pd.DataFrame | pd.Series | None = None
    metrics: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _parsed_date_column(df: pd.DataFrame, col: str, profile: FrameProfile) -> pd.Series:
    fmt = profile.columns[col].date_format
    return pd.to_datetime(df[col].astype(str), format=fmt, errors="coerce")


def _bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    series.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def _count_by_dimension(spec: ChartSpec, frames: dict[str, pd.DataFrame],
                         profiles: dict[str, FrameProfile]) -> ChartResult:
    counts = frames[spec.frame][spec.dimension].value_counts().sort_values(ascending=False)
    fig = _bar_chart(counts, spec.title, spec.dimension, "count")
    return ChartResult(title=spec.title, figure=fig, table=counts)


def _measure_by_dimension(spec: ChartSpec, frames: dict[str, pd.DataFrame],
                           profiles: dict[str, FrameProfile]) -> ChartResult:
    grouped = frames[spec.frame].groupby(spec.dimension)[spec.measure].agg(spec.agg)
    grouped = grouped.sort_values(ascending=False)
    fig = _bar_chart(grouped, spec.title, spec.dimension, f"{spec.agg}({spec.measure})")
    return ChartResult(title=spec.title, figure=fig, table=grouped)


def _measure_over_time(spec: ChartSpec, frames: dict[str, pd.DataFrame],
                        profiles: dict[str, FrameProfile]) -> ChartResult:
    df = frames[spec.frame]
    parsed = _parsed_date_column(df, spec.date_col, profiles[spec.frame])
    period = parsed.dt.to_period("M")
    grouped = df.assign(_period=period).groupby("_period")[spec.measure].agg(spec.agg).sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grouped.index.astype(str), grouped.values, marker="o", color="#4C72B0")
    ax.set_title(spec.title)
    ax.set_xlabel(spec.date_col)
    ax.set_ylabel(f"{spec.agg}({spec.measure})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return ChartResult(title=spec.title, figure=fig, table=grouped)


def _crosstab(spec: ChartSpec, frames: dict[str, pd.DataFrame],
              profiles: dict[str, FrameProfile]) -> ChartResult:
    df = frames[spec.frame]
    ct = pd.crosstab(df[spec.dimension], df[spec.dimension2])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)
    ax.set_title(spec.title)
    fig.colorbar(im, ax=ax)
    threshold = ct.values.max() / 2 if ct.values.size else 0
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            value = ct.values[i, j]
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > threshold else "black")
    fig.tight_layout()
    return ChartResult(title=spec.title, figure=fig, table=ct)


def _compute_metric_value(metric: dict[str, Any], frames: dict[str, pd.DataFrame]) -> Any:
    df = frames[metric["frame"]]
    measure = metric.get("measure")
    agg = metric.get("agg", "count")
    if measure is None:
        return len(df)
    series = df[measure]
    if agg == "sum":
        return series.sum()
    if agg == "mean":
        return series.mean()
    return series.count()  # agg == "count"


def _headline_metrics(spec: ChartSpec, frames: dict[str, pd.DataFrame],
                       profiles: dict[str, FrameProfile]) -> ChartResult:
    values = [
        {"label": m["label"], "value": _compute_metric_value(m, frames)}
        for m in spec.metrics
    ]
    return ChartResult(title=spec.title, metrics=values)


def _resolve_merged_column(name: str, merged: pd.DataFrame) -> str:
    """After a merge with suffixes=('', '_right'), a column that existed in
    both frames under the same name (but wasn't the join key) survives on the
    left as `name` and on the right as `name_right`. Prefer the plain name;
    only known not to matter for the sample data, where the one shared name
    across files is the join key itself."""
    return name if name in merged.columns else f"{name}_right"


def _cross_file_measure(spec: ChartSpec, frames: dict[str, pd.DataFrame],
                         profiles: dict[str, FrameProfile]) -> ChartResult:
    left, right = frames[spec.frame], frames[spec.frame2]
    if spec.join_left == spec.join_right:
        merged = left.merge(right, on=spec.join_left, how="inner", suffixes=("", "_right"))
    else:
        merged = left.merge(right, left_on=spec.join_left, right_on=spec.join_right,
                             how="inner", suffixes=("", "_right"))

    dim_col = _resolve_merged_column(spec.dimension, merged)
    if spec.measure is None:
        grouped = merged.groupby(dim_col).size()
        ylabel = "count"
    else:
        measure_col = _resolve_merged_column(spec.measure, merged)
        grouped = merged.groupby(dim_col)[measure_col].agg(spec.agg)
        ylabel = f"{spec.agg}({spec.measure})"
    grouped = grouped.sort_values(ascending=False)

    fig = _bar_chart(grouped, spec.title, spec.dimension, ylabel)
    return ChartResult(title=spec.title, figure=fig, table=grouped)


_RECIPE_EXECUTORS = {
    RECIPE_COUNT_BY_DIMENSION: _count_by_dimension,
    RECIPE_MEASURE_BY_DIMENSION: _measure_by_dimension,
    RECIPE_MEASURE_OVER_TIME: _measure_over_time,
    RECIPE_CROSSTAB: _crosstab,
    RECIPE_HEADLINE_METRICS: _headline_metrics,
    RECIPE_CROSS_FILE_MEASURE: _cross_file_measure,
}


def execute_chart_plan(
    specs: list[ChartSpec], frames: dict[str, pd.DataFrame], profiles: dict[str, FrameProfile]
) -> list[ChartResult]:
    """Run every validated spec. A single bad spec (an edge case validation
    missed — an empty group after some upstream filtering, say) is contained
    to its own ChartResult.error rather than taking the whole overview down.
    """
    plt.close("all")
    results = []
    for spec in specs:
        executor = _RECIPE_EXECUTORS.get(spec.recipe)
        if executor is None:
            results.append(ChartResult(title=spec.title, error=f"Unknown recipe: {spec.recipe}"))
            continue
        try:
            results.append(executor(spec, frames, profiles))
        except Exception as exc:  # noqa: BLE001 — surface any runtime error, don't crash the screen
            results.append(ChartResult(title=spec.title, error=f"{type(exc).__name__}: {exc}"))
    return results
