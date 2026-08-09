"""
findings.py — "what stands out" for the auto-analysis overview.

Same split as the rest of this feature: pandas computes, the LLM only
phrases. Four candidate generators look for the shapes of finding that
actually matter to a People Ops analyst — the biggest gap between groups,
a time trend, one group that's an outlier, and a joined-vs-not-joined
difference (e.g. leavers vs stayers) — each producing a `Finding` with the
numbers already baked into a deterministic, correct `description`. The top
three by effect size go to the model for a single rewrite pass into more
natural language; if that call fails or returns something malformed, the
deterministic descriptions are shown as-is. Either way, no number ever
passes through the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core import LLMBackend
from profiler import FrameProfile, JoinKey, ROLE_DATE, ROLE_DIMENSION, ROLE_MEASURE

MIN_GROUPS_FOR_GAP = 2
MIN_GROUPS_FOR_OUTLIER = 3
MIN_PERIODS_FOR_TREND = 3
OUTLIER_Z_THRESHOLD = 1.5
TOP_N_FINDINGS = 3


@dataclass
class Finding:
    kind: str
    description: str
    effect_size: float
    numbers: dict[str, Any] = field(default_factory=dict)


def _parsed_dates(df: pd.DataFrame, col: str, profile: FrameProfile) -> pd.Series:
    fmt = profile.columns[col].date_format
    return pd.to_datetime(df[col].astype(str), format=fmt, errors="coerce")


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------

def _largest_gap_candidates(
    frames: dict[str, pd.DataFrame], profiles: dict[str, FrameProfile]
) -> list[Finding]:
    candidates = []
    for fname, fp in profiles.items():
        df = frames[fname]
        for dim in fp.columns_with_role(ROLE_DIMENSION):
            for measure in fp.columns_with_role(ROLE_MEASURE):
                grouped = df.groupby(dim)[measure].mean()
                if len(grouped) < MIN_GROUPS_FOR_GAP:
                    continue
                overall_std = df[measure].std()
                if not overall_std or pd.isna(overall_std):
                    continue
                top_group, top_val = grouped.idxmax(), grouped.max()
                bottom_group, bottom_val = grouped.idxmin(), grouped.min()
                gap = top_val - bottom_val
                if gap == 0:
                    continue
                candidates.append(Finding(
                    kind="largest_gap",
                    description=(
                        f"In `{fname}`, {top_group} has the highest average {measure} "
                        f"({top_val:,.0f}), and {bottom_group} the lowest ({bottom_val:,.0f}) "
                        f"— a gap of {gap:,.0f}."
                    ),
                    effect_size=gap / overall_std,
                    numbers={
                        "frame": fname, "dimension": dim, "measure": measure,
                        "top_group": str(top_group), "top_value": float(top_val),
                        "bottom_group": str(bottom_group), "bottom_value": float(bottom_val),
                        "gap": float(gap),
                    },
                ))
    return candidates


def _trend_candidates(
    frames: dict[str, pd.DataFrame], profiles: dict[str, FrameProfile]
) -> list[Finding]:
    candidates = []
    for fname, fp in profiles.items():
        df = frames[fname]
        for date_col in fp.columns_with_role(ROLE_DATE):
            parsed = _parsed_dates(df, date_col, fp)
            period = parsed.dt.to_period("M")
            for measure in fp.columns_with_role(ROLE_MEASURE):
                grouped = df.assign(_period=period).groupby("_period")[measure].sum().sort_index()
                if len(grouped) < MIN_PERIODS_FOR_TREND:
                    continue
                x = pd.Series(range(len(grouped)), dtype=float)
                corr = x.corr(pd.Series(grouped.values, dtype=float))
                if pd.isna(corr) or corr == 0:
                    continue
                direction = "increasing" if corr > 0 else "decreasing"
                first_val, last_val = grouped.iloc[0], grouped.iloc[-1]
                candidates.append(Finding(
                    kind="trend",
                    description=(
                        f"In `{fname}`, total {measure} by month is {direction} — from "
                        f"{first_val:,.0f} in {grouped.index[0]} to {last_val:,.0f} in "
                        f"{grouped.index[-1]} (correlation {corr:.2f})."
                    ),
                    effect_size=abs(corr) * 3,  # rescaled onto roughly the same order as a z-score-style gap
                    numbers={
                        "frame": fname, "date_col": date_col, "measure": measure,
                        "first_period": str(grouped.index[0]), "first_value": float(first_val),
                        "last_period": str(grouped.index[-1]), "last_value": float(last_val),
                        "correlation": float(corr),
                        "series": {str(k): float(v) for k, v in grouped.items()},
                    },
                ))
    return candidates


def _outlier_group_candidates(
    frames: dict[str, pd.DataFrame], profiles: dict[str, FrameProfile]
) -> list[Finding]:
    candidates = []
    for fname, fp in profiles.items():
        df = frames[fname]
        for dim in fp.columns_with_role(ROLE_DIMENSION):
            for measure in fp.columns_with_role(ROLE_MEASURE):
                grouped = df.groupby(dim)[measure].mean()
                if len(grouped) < MIN_GROUPS_FOR_OUTLIER:
                    continue
                std = grouped.std()
                if not std or pd.isna(std):
                    continue
                z = (grouped - grouped.mean()) / std
                idx = z.abs().idxmax()
                z_val = z.loc[idx]
                if abs(z_val) < OUTLIER_Z_THRESHOLD:
                    continue
                candidates.append(Finding(
                    kind="outlier_group",
                    description=(
                        f"In `{fname}`, {idx} stands out on average {measure} "
                        f"({grouped.loc[idx]:,.0f}), about {abs(z_val):.1f} standard deviations "
                        f"{'above' if z_val > 0 else 'below'} the typical {dim} group."
                    ),
                    effect_size=abs(z_val),
                    numbers={
                        "frame": fname, "dimension": dim, "measure": measure,
                        "group": str(idx), "value": float(grouped.loc[idx]), "z_score": float(z_val),
                        "group_means": {str(k): float(v) for k, v in grouped.items()},
                    },
                ))
    return candidates


def _join_difference_candidates(
    frames: dict[str, pd.DataFrame], profiles: dict[str, FrameProfile], join_keys: list[JoinKey]
) -> list[Finding]:
    """Compares a measure in the frame with more rows ('base') between rows
    whose key also appears in the other frame ('subset') and rows that don't
    — e.g. exited vs still-employed CTC, when `exits` is the subset side."""
    candidates = []
    for jk in join_keys:
        left_n = frames[jk.left_frame][jk.left_column].nunique()
        right_n = frames[jk.right_frame][jk.right_column].nunique()
        if left_n <= right_n:
            subset_frame, subset_col = jk.left_frame, jk.left_column
            base_frame, base_col = jk.right_frame, jk.right_column
        else:
            subset_frame, subset_col = jk.right_frame, jk.right_column
            base_frame, base_col = jk.left_frame, jk.left_column

        subset_keys = set(frames[subset_frame][subset_col].dropna().unique())
        base_df = frames[base_frame]
        is_joined = base_df[base_col].isin(subset_keys)
        if is_joined.sum() == 0 or (~is_joined).sum() == 0:
            continue

        for measure in profiles[base_frame].columns_with_role(ROLE_MEASURE):
            overall_std = base_df[measure].std()
            if not overall_std or pd.isna(overall_std):
                continue
            joined_mean = base_df.loc[is_joined, measure].mean()
            unjoined_mean = base_df.loc[~is_joined, measure].mean()
            diff = joined_mean - unjoined_mean
            if diff == 0:
                continue
            candidates.append(Finding(
                kind="join_difference",
                description=(
                    f"In `{base_frame}`, rows also present in `{subset_frame}` (matched on "
                    f"{base_col}) average {measure} of {joined_mean:,.0f}, vs {unjoined_mean:,.0f} "
                    f"for rows that aren't — a difference of {diff:,.0f}."
                ),
                effect_size=abs(diff) / overall_std,
                numbers={
                    "base_frame": base_frame, "subset_frame": subset_frame, "measure": measure,
                    "base_col": base_col, "subset_col": subset_col,
                    "joined_mean": float(joined_mean), "unjoined_mean": float(unjoined_mean),
                    "diff": float(diff),
                },
            ))
    return candidates


def compute_findings(
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, FrameProfile],
    join_keys: list[JoinKey],
    top_n: int = TOP_N_FINDINGS,
) -> list[Finding]:
    candidates = (
        _largest_gap_candidates(frames, profiles)
        + _trend_candidates(frames, profiles)
        + _outlier_group_candidates(frames, profiles)
        + _join_difference_candidates(frames, profiles, join_keys)
    )
    candidates.sort(key=lambda f: f.effect_size, reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# LLM phrasing — rewrite only, never compute
# ---------------------------------------------------------------------------

class FindingsPhrasingError(Exception):
    pass


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def build_phrasing_prompt(findings: list[Finding]) -> str:
    listed = "\n".join(f"{i + 1}. {f.description}" for i, f in enumerate(findings))
    return f"""Rewrite each of these {len(findings)} already-computed findings as one \
short, plain-English sentence for a People Ops / HR analyst. Every number in \
each finding has already been computed correctly — restate it, don't \
recompute it, round it differently, or add any number that isn't already \
there.

Findings:
{listed}

Return ONLY a JSON array of exactly {len(findings)} strings, in the same \
order, no markdown fences, no explanation.
"""


def parse_phrasings(raw_text: str, expected_count: int) -> list[str]:
    text = _strip_code_fences(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FindingsPhrasingError(f"Not valid JSON: {exc}") from exc
    if not isinstance(data, list) or len(data) != expected_count:
        raise FindingsPhrasingError(
            f"Expected a JSON array of {expected_count} strings, got {data!r}"
        )
    if not all(isinstance(s, str) for s in data):
        raise FindingsPhrasingError("Not all elements were strings.")
    return data


def phrase_findings(findings: list[Finding], backend: LLMBackend) -> list[str]:
    """Best-effort natural-language rewrite. Falls back to the deterministic
    `description` (already correct, just plainer) if the model's response
    can't be parsed — a bad phrasing call must never block the panel."""
    if not findings:
        return []
    prompt = build_phrasing_prompt(findings)
    try:
        raw = backend.generate(prompt)
        return parse_phrasings(raw, len(findings))
    except Exception:
        return [f.description for f in findings]
