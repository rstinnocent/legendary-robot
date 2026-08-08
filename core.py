"""
core.py — backend logic for the AI-powered Data Q&A app.

Kept separate from the Streamlit UI (app.py) on purpose: this module has
no dependency on Streamlit, so it can be unit tested directly (see
test_core.py) and reused from a CLI or notebook if needed.

Approach: rather than a black-box library, questions are answered by
asking an LLM to write short pandas/matplotlib code against the uploaded
dataframes, then executing that code in a restricted namespace. This
keeps the whole pipeline inspectable — the generated code is shown to
the user alongside every answer.
"""

from __future__ import annotations

import builtins
import io
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend — no GUI needed to render charts
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def sanitize_name(filename: str) -> str:
    """Turn 'Sales Q1.csv' into a valid Python identifier: 'sales_q1'."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"\W+", "_", name).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"df_{name}"
    return name


def load_file(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    """Load a single CSV or Excel file (given as raw bytes) into a DataFrame."""
    buf = io.BytesIO(raw_bytes)
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return pd.read_csv(buf)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(buf)
    raise ValueError(f"Unsupported file type: {filename}")


def load_files(files: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    """files: {filename: raw_bytes} -> {sanitized_var_name: DataFrame}."""
    frames: dict[str, pd.DataFrame] = {}
    for filename, raw in files.items():
        var_name = sanitize_name(filename)
        base, i = var_name, 1
        while var_name in frames:
            i += 1
            var_name = f"{base}_{i}"
        frames[var_name] = load_file(filename, raw)
    return frames


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def describe_frame(name: str, df: pd.DataFrame, sample_rows: int = 3) -> str:
    dtypes = "\n".join(f"    - {c}: {t}" for c, t in df.dtypes.astype(str).items())
    sample = df.head(sample_rows).to_csv(index=False)
    return (
        f"DataFrame `{name}` — {len(df)} rows, {len(df.columns)} columns\n"
        f"  columns:\n{dtypes}\n"
        f"  sample rows:\n{sample}"
    )


def build_prompt(question: str, frames: dict[str, pd.DataFrame]) -> str:
    schema_block = "\n\n".join(describe_frame(n, d) for n, d in frames.items())
    frame_list = ", ".join(frames.keys())
    return f"""You are a data analyst. These pandas DataFrames are already loaded \
in the execution environment: {frame_list}.

{schema_block}

Write Python code to answer this question:
"{question}"

Rules:
- Use only pandas (pd), numpy (np), and matplotlib.pyplot (plt) — already imported.
- Do not redefine or reload the DataFrames; use them as given.
- To combine data across files, merge/join on shared columns you can see in the schemas above.
- If the answer is a number, string, or table, assign it to a variable named `result`.
- If a chart is the clearest answer, build it with matplotlib and leave the figure \
open (no plt.show()); still assign a short text summary to `result`.
- Do not use file I/O, network calls, `import os`, `import sys`, `eval`, `exec`, \
or any dunder attributes.
- Return ONLY the Python code. No explanation, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Restricted execution of LLM-generated code
# ---------------------------------------------------------------------------

BLOCKED_PATTERNS = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bopen\s*\(", r"__\w+__", r"\beval\s*\(", r"\bexec\s*\(",
    r"\bcompile\s*\(", r"\binput\s*\(", r"\bgetattr\s*\(", r"\bsetattr\s*\(",
]

_ALLOWED_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "print", "range", "round",
    "set", "sorted", "str", "sum", "tuple", "zip",
]
SAFE_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_NAMES}


class UnsafeCodeError(Exception):
    """Raised when LLM-generated code trips the (deliberately blunt) safety filter."""


def check_code_safety(code: str) -> None:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            raise UnsafeCodeError(f"Generated code contains a disallowed pattern: {pattern}")


def strip_code_fences(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


@dataclass
class ExecutionResult:
    result: Any = None
    figure: Any = None
    code: str = ""
    error: str | None = None


def run_generated_code(code: str, frames: dict[str, pd.DataFrame]) -> ExecutionResult:
    code = strip_code_fences(code)
    try:
        check_code_safety(code)
    except UnsafeCodeError as exc:
        return ExecutionResult(code=code, error=str(exc))

    plt.close("all")
    local_ns: dict[str, Any] = dict(frames)
    global_ns = {"pd": pd, "np": np, "plt": plt, "__builtins__": SAFE_BUILTINS}

    try:
        exec(code, global_ns, local_ns)  # noqa: S102 — restricted namespace above
    except Exception as exc:  # noqa: BLE001 — surface any runtime error to the UI
        return ExecutionResult(code=code, error=f"{type(exc).__name__}: {exc}")

    fig = plt.gcf() if plt.get_fignums() else None
    return ExecutionResult(result=local_ns.get("result"), figure=fig, code=code)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class LLMBackend:
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class GroqBackend(LLMBackend):
    """Groq's free tier — no credit card required, open-weight models only."""

    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq  # lazy import: only needed if this backend is used

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No Groq API key found. Get a free one (no card needed) at "
                "console.groq.com/keys and set GROQ_API_KEY, or paste it in the sidebar."
            )
        self.client = Groq(api_key=key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content


class OllamaBackend(LLMBackend):
    """A locally running Ollama instance — zero external dependency."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt: str) -> str:
        import requests

        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]


def get_backend(name: str, **kwargs) -> LLMBackend:
    if name == "groq":
        return GroqBackend(**kwargs)
    if name == "ollama":
        return OllamaBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def answer_question(
    question: str, frames: dict[str, pd.DataFrame], backend: LLMBackend
) -> ExecutionResult:
    prompt = build_prompt(question, frames)
    code = backend.generate(prompt)
    return run_generated_code(code, frames)
