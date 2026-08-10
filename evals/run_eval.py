"""
run_eval.py — run the fixed eval set against a real LLM backend and report accuracy.

Usage
-----
    export GROQ_API_KEY=...            # free tier, no card
    python evals/run_eval.py                       # default: groq / llama-3.3-70b
    python evals/run_eval.py --backend ollama --model llama3.2
    python evals/run_eval.py --runs 3              # repeat to see variance

Writes a markdown summary to evals/RESULTS.md.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import answer_question, get_backend, load_files  # noqa: E402
from eval_set import EVAL_CASES, EvalCase  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

REFUSAL_MARKERS = [
    "no ", "not ", "cannot", "can't", "unavailable", "missing", "does not exist",
    "doesn't exist", "does not include", "doesn't include", "n/a", "unable",
    "no column", "not present", "not found",
]


def load_sample_frames() -> dict[str, pd.DataFrame]:
    files = {p.name: p.read_bytes() for p in sorted(SAMPLE_DIR.glob("*.csv"))}
    if not files:
        raise SystemExit(f"No sample CSVs found in {SAMPLE_DIR}")
    return load_files(files)


def normalise(text: str) -> str:
    """Strip thousands separators and collapse whitespace so numeric matching is robust."""
    text = str(text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)   # 1,168,150 -> 1168150
    text = re.sub(r"\s+", " ", text)
    return text


def render_answer(res) -> str:
    """Flatten whatever came back (scalar / Series / DataFrame / str) into one string.

    Floats are forced to fixed-point notation: pandas renders large means as
    `9.082222e+05` by default, which would fail a substring match against a
    perfectly correct answer.
    """
    r = res.result
    fixed = lambda v: f"{v:.2f}"  # noqa: E731
    if isinstance(r, (pd.DataFrame, pd.Series)):
        try:
            return r.to_string(float_format=fixed)
        except TypeError:  # non-numeric frames don't accept float_format
            return r.to_string()
    if isinstance(r, float):
        return f"{r:.4f}".rstrip("0").rstrip(".")
    return "" if r is None else str(r)


def grade(case: EvalCase, res) -> tuple[bool, str]:
    if res.error:
        return False, f"execution error: {res.error}"

    rendered = normalise(render_answer(res))
    low = rendered.lower()

    if case.expect_refusal:
        if any(m in low for m in REFUSAL_MARKERS):
            return True, "correctly declined"
        for bad in case.forbidden:
            if bad in rendered:
                return False, f"hallucinated a value ({bad!r} present) instead of declining"
        return False, f"did not decline; answered: {rendered[:120]!r}"

    if case.expect_chart and res.figure is None:
        return False, "no chart was produced"

    missing = [t for t in case.must_contain if normalise(t) not in rendered]
    if missing:
        return False, f"missing expected value(s) {missing}; got: {rendered[:160]!r}"

    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="groq", choices=["groq", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=1, help="repeat the whole set N times")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    backend = get_backend(args.backend, **kwargs)
    model_name = getattr(backend, "model", args.backend)

    frames = load_sample_frames()
    print(f"Loaded {len(frames)} files: {', '.join(frames)}\n")

    tallies: dict[str, int] = {c.id: 0 for c in EVAL_CASES}
    rows: list[dict] = []

    for run in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"--- run {run}/{args.runs} ---")
        for case in EVAL_CASES:
            t0 = time.time()
            try:
                res = answer_question(case.question, frames, backend)
                ok, detail = grade(case, res)
            except Exception as exc:  # noqa: BLE001 — a backend failure is an eval failure
                ok, detail, res = False, f"backend error: {exc}", None
            elapsed = time.time() - t0
            tallies[case.id] += int(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {case.id}  {case.capability:<38} "
                  f"{elapsed:5.1f}s  {'' if ok else detail}")
            if run == 1:
                rows.append({
                    "id": case.id, "capability": case.capability,
                    "question": case.question, "passed": ok, "detail": detail,
                })

    total = len(EVAL_CASES) * args.runs
    passed = sum(tallies.values())
    pct = 100 * passed / total
    print(f"\n{passed}/{total} passed ({pct:.0f}%)  model={model_name}")

    if not args.no_write:
        out = Path(__file__).resolve().parent / "RESULTS.md"
        lines = [
            "# Eval results",
            "",
            f"- Model: `{model_name}` (backend: `{args.backend}`)",
            f"- Dataset: `sample_data/` — employees, attendance, exits ({len(frames)} files)",
            f"- Runs: {args.runs}",
            f"- **Score: {passed}/{total} ({pct:.0f}%)**",
            "",
            "| ID | Capability | Result | Notes |",
            "|---|---|---|---|",
        ]
        for r in rows:
            mark = "pass" if r["passed"] else "**fail**"
            note = "" if r["passed"] else r["detail"].replace("|", "\\|")[:120]
            lines.append(f"| {r['id']} | {r['capability']} | {mark} | {note} |")
        lines += ["", "Regenerate with `python evals/run_eval.py`.", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
