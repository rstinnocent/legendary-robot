"""
eval_auto_analysis.py — live-model checks for the auto-analysis feature.

Unlike test_chart_plan.py / test_findings.py (deterministic, synthetic
inputs, no API key), this exercises the real pipeline against a live model
and the real sample_data, because only a live run can show whether the
*actual* model output — not a hand-crafted fixture — holds up. It checks
exactly the three properties called for when this feature was scoped:

  1. every chart spec that actually gets executed references real columns
  2. at least one cross-file chart appears, given sample_data has join keys
  3. each of the top-3 "what stands out" findings recomputes, independently
     and from scratch, to the same numbers already in Finding.numbers —
     and the model's phrased sentence still contains those numbers

(1) and (2) are guaranteed by validate_plan's own logic (see
test_chart_plan.py), so this is a regression check that the guarantee
actually holds against live output, not a test of validate_plan itself.
(3) is the one property no unit test can cover: it depends on whether the
phrasing model call, running for real, altered a number it was told not to.

Usage
-----
    export GROQ_API_KEY=...
    python evals/eval_auto_analysis.py
    python evals/eval_auto_analysis.py --runs 3

Writes evals/AUTO_ANALYSIS_RESULTS.md.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chart_plan import generate_chart_plan, validate_plan, RECIPE_CROSS_FILE_MEASURE  # noqa: E402
from charts import execute_chart_plan  # noqa: E402
from core import get_backend, load_files  # noqa: E402
from findings import Finding, compute_findings, phrase_findings  # noqa: E402
from profiler import detect_join_keys, profile_frames  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def load_sample_frames() -> dict[str, pd.DataFrame]:
    files = {p.name: p.read_bytes() for p in sorted(SAMPLE_DIR.glob("*.csv"))}
    if not files:
        raise SystemExit(f"No sample CSVs found in {SAMPLE_DIR}")
    return load_files(files)


def normalise(text: str) -> str:
    text = str(text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Check 1 — every executed spec references real columns
# ---------------------------------------------------------------------------

def _column_exists(profiles, frame: str | None, col: str) -> bool:
    fp = profiles.get(frame)
    return fp is not None and col in fp.columns


def check_specs_reference_real_columns(specs, profiles) -> tuple[bool, str]:
    bad = []
    for spec in specs:
        for field_name, col in [("dimension", spec.dimension), ("dimension2", spec.dimension2),
                                 ("measure", spec.measure), ("date_col", spec.date_col)]:
            if col is None:
                continue
            if not (_column_exists(profiles, spec.frame, col) or _column_exists(profiles, spec.frame2, col)):
                bad.append(f"{spec.recipe}.{field_name}={col!r} not found in {spec.frame}/{spec.frame2}")
        for m in spec.metrics:
            frame = m.get("frame")
            measure = m.get("measure")
            if frame not in profiles:
                bad.append(f"headline_metrics metric frame={frame!r} unknown")
            elif measure is not None and measure not in profiles[frame].columns:
                bad.append(f"headline_metrics metric measure={measure!r} not in {frame}")
    if bad:
        return False, "; ".join(bad)
    return True, f"all {len(specs)} specs reference real columns"


# ---------------------------------------------------------------------------
# Check 2 — a cross-file chart is present when a join key exists
# ---------------------------------------------------------------------------

def check_cross_file_chart_present(specs, join_keys) -> tuple[bool, str]:
    if not join_keys:
        return True, "no join keys in this dataset — n/a"
    has_one = any(s.recipe == RECIPE_CROSS_FILE_MEASURE for s in specs)
    if has_one:
        return True, "cross_file_measure present"
    return False, "join keys exist but no cross_file_measure spec was executed"


# ---------------------------------------------------------------------------
# Check 3 — findings recompute independently, and survive phrasing intact
# ---------------------------------------------------------------------------

def _recompute(finding: Finding, frames: dict[str, pd.DataFrame], profiles) -> dict:
    """Redo each finding's arithmetic from scratch, straight from the raw
    frames — deliberately not calling back into findings.py's own generator
    functions, so a bug shared between the generator and this check
    wouldn't silently cancel out."""
    n = finding.numbers
    if finding.kind == "largest_gap":
        grouped = frames[n["frame"]].groupby(n["dimension"])[n["measure"]].mean()
        return {"top_value": float(grouped.max()), "bottom_value": float(grouped.min()),
                "gap": float(grouped.max() - grouped.min())}

    if finding.kind == "trend":
        df = frames[n["frame"]]
        fmt = profiles[n["frame"]].columns[n["date_col"]].date_format
        parsed = pd.to_datetime(df[n["date_col"]].astype(str), format=fmt, errors="coerce")
        grouped = df.assign(_p=parsed.dt.to_period("M")).groupby("_p")[n["measure"]].sum().sort_index()
        return {"first_value": float(grouped.iloc[0]), "last_value": float(grouped.iloc[-1])}

    if finding.kind == "outlier_group":
        grouped = frames[n["frame"]].groupby(n["dimension"])[n["measure"]].mean()
        z = (grouped[n["group"]] - grouped.mean()) / grouped.std()
        return {"value": float(grouped[n["group"]]), "z_score": float(z)}

    if finding.kind == "join_difference":
        base_df = frames[n["base_frame"]]
        subset_keys = set(frames[n["subset_frame"]][n["subset_col"]].dropna().unique())
        is_joined = base_df[n["base_col"]].isin(subset_keys)
        return {
            "joined_mean": float(base_df.loc[is_joined, n["measure"]].mean()),
            "unjoined_mean": float(base_df.loc[~is_joined, n["measure"]].mean()),
        }

    raise ValueError(f"Unknown finding kind: {finding.kind}")


def check_findings_recompute(findings, phrasings, frames, profiles) -> tuple[bool, str]:
    if not findings:
        return True, "no findings surfaced — n/a"
    problems = []
    for finding, phrasing in zip(findings, phrasings):
        recomputed = _recompute(finding, frames, profiles)
        for key, expected in recomputed.items():
            actual = finding.numbers.get(key)
            if actual is None or abs(actual - expected) > max(0.5, abs(expected) * 1e-6):
                problems.append(f"[{finding.kind}] {key}: stored={actual} recomputed={expected}")

        # every number that appears in the finding's own numbers dict should
        # still be findable in the model's phrased sentence — this is the
        # check that actually exercises the live phrasing call.
        phrasing_norm = normalise(phrasing)
        for key, val in finding.numbers.items():
            if not isinstance(val, (int, float)):
                continue
            rendered = f"{val:,.0f}"
            if normalise(rendered) not in phrasing_norm and f"{val:.2f}" not in phrasing_norm:
                problems.append(
                    f"[{finding.kind}] number {key}={rendered} missing from phrasing: {phrasing[:100]!r}"
                )

    if problems:
        return False, "; ".join(problems)
    return True, f"all {len(findings)} findings recompute correctly and survive phrasing"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_once(backend) -> list[tuple[str, bool, str]]:
    frames = load_sample_frames()
    profiles = profile_frames(frames)
    join_keys = detect_join_keys(frames, profiles)

    raw_plan = generate_chart_plan(profiles, join_keys, backend)
    validated = validate_plan(raw_plan, profiles, join_keys)
    execute_chart_plan(validated, frames, profiles)  # exercised for side effects / crashes only

    findings = compute_findings(frames, profiles, join_keys)
    phrasings = phrase_findings(findings, backend)

    results = []
    for name, fn, args in [
        ("specs_reference_real_columns", check_specs_reference_real_columns, (validated, profiles)),
        ("cross_file_chart_present", check_cross_file_chart_present, (validated, join_keys)),
        ("findings_recompute", check_findings_recompute, (findings, phrasings, frames, profiles)),
    ]:
        ok, detail = fn(*args)
        results.append((name, ok, detail))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="groq", choices=["groq", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    kwargs = {"model": args.model} if args.model else {}
    backend = get_backend(args.backend, **kwargs)
    model_name = getattr(backend, "model", args.backend)

    all_rows = []
    total_pass = 0
    total_checks = 0
    for run in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"--- run {run}/{args.runs} ---")
        t0 = time.time()
        try:
            results = run_once(backend)
        except Exception as exc:  # noqa: BLE001 — a pipeline failure is a check failure
            results = [
                ("specs_reference_real_columns", False, f"pipeline error: {exc}"),
                ("cross_file_chart_present", False, f"pipeline error: {exc}"),
                ("findings_recompute", False, f"pipeline error: {exc}"),
            ]
        elapsed = time.time() - t0
        for name, ok, detail in results:
            total_checks += 1
            total_pass += int(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} {elapsed:5.1f}s  {'' if ok else detail}")
            if run == 1:
                all_rows.append({"name": name, "passed": ok, "detail": detail})

    pct = 100 * total_pass / total_checks if total_checks else 0
    print(f"\n{total_pass}/{total_checks} checks passed ({pct:.0f}%)  model={model_name}")

    if not args.no_write:
        out = Path(__file__).resolve().parent / "AUTO_ANALYSIS_RESULTS.md"
        lines = [
            "# Auto-analysis eval results",
            "",
            f"- Model: `{model_name}` (backend: `{args.backend}`)",
            f"- Runs: {args.runs}",
            f"- **Score: {total_pass}/{total_checks} ({pct:.0f}%)**",
            "",
            "| Check | Result | Detail |",
            "|---|---|---|",
        ]
        for r in all_rows:
            mark = "pass" if r["passed"] else "**fail**"
            note = "" if r["passed"] else r["detail"].replace("|", "\\|")[:200]
            lines.append(f"| {r['name']} | {mark} | {note} |")
        lines += ["", "Regenerate with `python evals/eval_auto_analysis.py`.", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if total_pass == total_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
