"""
app.py — Streamlit UI for Crosswalk (the AI-powered cross-file data Q&A app).

Structure follows the converged wireframe set (turn 3 of the "Wireframe
screens scoping" design canvas): empty state -> files loaded -> overview
-> Q&A, with two inline states (declined / errored) inside Q&A cards. The
backend logic (core.py, profiler.py, chart_plan.py, charts.py, findings.py)
is unchanged from before this redesign — this file only rearranges how
their outputs are presented.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from chart_plan import ChartPlanError, RECIPE_HEADLINE_METRICS, generate_chart_plan, validate_plan
from charts import execute_chart_plan
from core import answer_question, get_backend, load_files, sanitize_name
from findings import compute_findings, phrase_findings
from profiler import detect_join_keys, profile_frames
from ui_helpers import (
    chart_caption,
    classify_answer_state,
    dataframe_to_csv_bytes,
    describe_join,
    format_metric_value,
    more_charts_label,
    pluralize,
    question_for_spec,
    safe_download_filename,
    schema_section_label,
    suggested_questions,
    type_chip,
)

APP_TAGLINE = "Ask questions about your data in plain English — Crosswalk automatically connects the dots across multiple files."

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"

st.set_page_config(page_title="Crosswalk", page_icon="✎", layout="wide")

st.markdown(
    """
<style>
.in-hdr{display:flex;justify-content:space-between;align-items:center;
  border-bottom:2px solid #1a1a1a;padding-bottom:10px;margin-bottom:18px}
.in-logo{font-size:22px;font-weight:700}
.in-subtitle{font-size:13px;font-weight:400;color:rgba(0,0,0,.55);margin-top:2px}
.in-tag{font-size:12px;color:rgba(0,0,0,.55);border:1px dashed rgba(0,0,0,.4);
  padding:3px 9px;border-radius:10px;white-space:nowrap;margin-left:12px}
.in-join{font-size:12.5px;color:#1F4E79;border:1px dashed #1F4E79;
  padding:6px 9px;margin:6px 0;border-radius:4px}
.in-caption{font-size:12px;color:rgba(0,0,0,.55);margin-top:4px}
.in-empty{border:2px dashed rgba(0,0,0,.35);border-radius:6px;padding:48px 20px;
  text-align:center;color:rgba(0,0,0,.55)}
.in-empty .sub{font-size:12.5px;margin-top:8px;color:rgba(0,0,0,.4)}
</style>
""",
    unsafe_allow_html=True,
)


def _sample_data_bytes() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(SAMPLE_DIR.glob("*.csv"))}


def _default_api_key() -> str:
    """Pre-fill the sidebar's API key field from whichever secret store the
    host actually uses — st.secrets (Streamlit Community Cloud's Secrets
    manager, TOML-based) or a plain environment variable (most other
    hosts) — so a deployed app can be configured without the key ever
    living in the repo. st.secrets raises if no secrets.toml exists at all
    (the normal case for local dev), hence the try/except rather than a
    plain lookup."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def _build_backend():
    return (
        get_backend("groq", api_key=st.session_state.get("api_key"), model=st.session_state.get("model"))
        if st.session_state.get("backend_choice", "").startswith("Groq")
        else get_backend("ollama", model=st.session_state.get("model"))
    )


def _ingest_files(file_bytes: dict[str, bytes]) -> None:
    """Merge newly-provided files into the existing dataset rather than
    replacing it outright. Files arrive from three different places — the
    empty-state drop zone, "try sample data", and "Add more files" in the
    sidebar — each its own widget/call with no knowledge of files loaded
    through the others; replacing st.session_state.frames wholesale here
    (the original approach) silently dropped everything already loaded the
    moment someone used "Add more files" to add a fourth file to three
    already-loaded ones.

    "Already loaded" is judged against st.session_state.frames itself
    (via each incoming file's sanitized name), not a separate "seen
    filenames" set: a Streamlit rerun fires on every widget interaction,
    and the sidebar uploader keeps reporting the same files on every one of
    them, so *something* has to stop that from re-triggering a reset on
    every rerun — but checking the real frames dict, rather than a set that
    only ever grows, means removing a file and re-dropping the same name
    later correctly loads it again instead of being silently ignored as
    "already seen".

    Known limitation: re-uploading a *different* file under a name already
    loaded (e.g. a corrected employees.csv, without removing the old one
    first) is treated as nothing new and silently ignored — good enough for
    "add more files", not a general re-upload/replace-in-place feature.
    """
    if not file_bytes:
        return
    current_frames = st.session_state.get("frames", {})
    new_bytes = {name: data for name, data in file_bytes.items()
                 if sanitize_name(name) not in current_frames}
    if not new_bytes:
        return
    try:
        new_frames = load_files(new_bytes)
    except Exception as exc:
        st.error(f"Couldn't read one of the files: {exc}")
        return
    st.session_state.frames = {**current_frames, **new_frames}
    st.session_state.analysis = None
    st.session_state.qa_history = []


def _remove_file(name: str) -> None:
    st.session_state.frames.pop(name, None)
    st.session_state.analysis = None
    st.session_state.qa_history = []


def _header(tag: str) -> None:
    st.markdown(
        '<div class="in-hdr">'
        '<div><div class="in-logo">✎ Crosswalk</div>'
        f'<div class="in-subtitle">{html.escape(APP_TAGLINE)}</div></div>'
        f'<div class="in-tag">{html.escape(tag)}</div></div>',
        unsafe_allow_html=True,
    )


@st.dialog("How Crosswalk works")
def _show_welcome_dialog() -> None:
    st.markdown(
        "1. **Upload your files** — CSV or Excel. Files that share a common "
        "column, like an employee ID, get linked automatically.\n"
        "2. **Click Analyze** for an instant overview — headline numbers, "
        "charts, and what stands out — or skip straight to asking a "
        "question.\n"
        "3. **Ask anything in plain English.** Crosswalk writes and runs "
        "the analysis for you, and always shows its work so you can check "
        "it."
    )
    if st.button("Got it — let's go", type="primary", width="stretch"):
        st.rerun()


def _describe_backend_error(exc: Exception) -> str:
    """Groq's free tier caps total tokens per day, shared across every model
    — easy to hit during a demo/eval session. That error is common enough,
    and its raw form technical enough, that it earns a specific, actionable
    message instead of the generic fallback."""
    try:
        from groq import RateLimitError
    except ImportError:  # pragma: no cover - groq is a hard dependency
        RateLimitError = ()
    if isinstance(exc, RateLimitError) or "rate_limit_exceeded" in str(exc):
        return ("Groq's free daily limit has been reached for this model. Try "
                 "switching to a different model in the sidebar under **AI "
                 "model**, or wait a few minutes and try again.")
    return f"The AI provider call failed: {exc}"


def _ask(question: str, frames: dict) -> None:
    """Run one question through the Q&A engine and prepend it to history."""
    try:
        backend = _build_backend()
    except Exception as exc:
        st.error(f"Couldn't connect to the AI provider: {exc}")
        return
    with st.spinner("Thinking..."):
        try:
            result = answer_question(question, frames, backend)
        except Exception as exc:  # network / API errors from the backend itself
            st.error(_describe_backend_error(exc))
            return
    st.session_state.qa_history.insert(0, {"question": question, "result": result, "pinned": False})


def _render_chart_card(idx: int, spec, result) -> None:
    """One chart card's contents — title, chart/table, caption, and the
    details/ask actions. Shared between the always-visible first chart and
    the grid of remaining ones, indexed by position (not spec.title): the
    LLM's chart plan isn't validated for title uniqueness, and two cards
    sharing a title would collide on a title-based widget key."""
    with st.container(border=True):
        st.markdown(f"**{spec.title}**")
        if result.error:
            st.warning(result.error)
        elif result.figure is not None:
            st.pyplot(result.figure)
        elif result.table is not None:
            st.dataframe(result.table)
        caption = chart_caption(spec, result)
        if caption:
            st.markdown(f'<div class="in-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        q = question_for_spec(spec)
        with b1:
            with st.popover("details", key=f"recipe_{idx}"):
                if q:
                    st.caption(f"This chart answers: *{q}*")
                st.code(f"{spec.recipe}({spec.__dict__})", language="python")
        with b2:
            if q and st.button("ask", key=f"ask_{idx}"):
                st.session_state.pending_question = q
                st.rerun()


# ---------------------------------------------------------------- state ---
for key, default in [
    ("frames", {}), ("analysis", None), ("qa_history", []),
    ("pending_question", None), ("welcome_seen", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Settings")
    backend_choice = st.selectbox(
        "AI provider", ["Groq (free, hosted)", "Ollama (local)"],
        help="Powers the plain-English Q&A below. Both options are free and require no paid account.",
    )
    st.session_state.backend_choice = backend_choice

    if backend_choice.startswith("Groq"):
        _preconfigured_key = _default_api_key()
        if _preconfigured_key:
            # Configured via a deployment secret — nothing for a user to
            # enter or see here, just a quiet confirmation it's connected.
            st.session_state.api_key = _preconfigured_key
            st.caption("✓ Connected")
        else:
            st.session_state.api_key = st.text_input(
                "Groq API key", type="password",
                help="Free, no credit card required — get one at console.groq.com/keys. "
                     "Set as a deployment secret and this field won't show at all.",
            )
        st.session_state.model = st.selectbox(
            "AI model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
            help="llama-3.3-70b-versatile is the most accurate; llama-3.1-8b-instant is faster.",
        )
    else:
        st.session_state.api_key = None
        st.session_state.model = st.text_input("Ollama model", value="llama3.2")
        st.caption("Requires Ollama running locally (`ollama serve`).")

    st.divider()

    if st.session_state.frames:
        # Ingest before rendering the file list below, not after: a file
        # dropped into "Add more files" needs to show up in the count and
        # per-file rows in this same rerun, not the next one.
        uploaded = st.file_uploader(
            "Add more files", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="uploader"
        )
        _ingest_files({f.name: f.getvalue() for f in uploaded} if uploaded else {})
        frames_now = st.session_state.frames
        st.subheader(f"Files ({len(frames_now)})")
        for name, df in list(frames_now.items()):
            row_col, remove_col = st.columns([5, 1])
            row_col.caption(f"**{name}** — {pluralize(len(df), 'row')} · {pluralize(len(df.columns), 'col')}")
            if remove_col.button("✕", key=f"remove_{name}", help=f"Remove {name}"):
                _remove_file(name)
                st.rerun()
    else:
        uploaded = None

frames = st.session_state.frames

# ---------------------------------------------------------- empty state ---
if not frames:
    _header("no files yet")
    if not st.session_state.welcome_seen:
        # Set before opening, not inside the button handler: this way the
        # dialog is guaranteed to auto-open only once per session no matter
        # how it's dismissed (the "Got it" button, the dialog's own close
        # X, or clicking outside it) — only the manual link below can
        # reopen it after that.
        st.session_state.welcome_seen = True
        _show_welcome_dialog()
    if st.button("❔ How does this work?"):
        _show_welcome_dialog()
    st.markdown(
        '<div class="in-empty">Drop your CSV or Excel files here'
        '<div class="sub">upload the files you\'d normally join together — '
        "HRIS, attendance, payroll, exits</div></div>",
        unsafe_allow_html=True,
    )
    st.file_uploader(
        "Upload data files", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="uploader_empty"
    )
    empty_uploaded = st.session_state.get("uploader_empty")
    if empty_uploaded:
        _ingest_files({f.name: f.getvalue() for f in empty_uploaded})
        st.rerun()
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("or try it with sample HR data →", width="stretch"):
            _ingest_files(_sample_data_bytes())
            st.rerun()
    st.stop()

# ------------------------------------------------------- files loaded -----
total_rows = sum(len(df) for df in frames.values())
_header(f"{pluralize(len(frames), 'file')} · {pluralize(total_rows, 'row')}")

profiles = profile_frames(frames)
join_keys = detect_join_keys(frames, profiles)

_, mid, _ = st.columns([1, 1, 1])
with mid:
    analyze_clicked = st.button(
        "Analyze", type="primary", width="stretch",
        help="Get an instant summary: key numbers, charts, and what stands out in your data.",
    )

if not st.session_state.analysis:
    st.caption("Click **Analyze** above for an instant overview — key numbers, charts, and "
               "what stands out — or skip straight to **asking a question** near the bottom "
               "of the page.")

if analyze_clicked:
    try:
        backend = _build_backend()
    except Exception as exc:
        st.error(f"Couldn't connect to the AI provider: {exc}")
        st.stop()

    with st.spinner("Analyzing your data and building the overview..."):
        try:
            raw_plan = generate_chart_plan(profiles, join_keys, backend)
        except ChartPlanError as exc:
            st.error(f"Couldn't build an overview from the AI's response — try clicking "
                      f"Analyze again. ({exc})")
            st.stop()
        except Exception as exc:
            st.error(_describe_backend_error(exc))
            st.stop()

        validated = validate_plan(raw_plan, profiles, join_keys)
        chart_results = execute_chart_plan(validated, frames, profiles)
        findings = compute_findings(frames, profiles, join_keys)
        try:
            phrasings = phrase_findings(findings, backend)
        except Exception as exc:
            st.error(_describe_backend_error(exc))
            st.stop()

    st.session_state.analysis = {
        "specs": validated, "chart_results": chart_results,
        "findings": findings, "phrasings": phrasings,
    }

analysis = st.session_state.analysis

# ---------------------------------------------------- 3b: schema preview --
if not analysis:
    with st.expander(schema_section_label(profiles, join_keys), expanded=False):
        for name, profile in profiles.items():
            st.markdown(f"**{name}**")
            schema_df = pd.DataFrame(
                {"Column": list(profile.columns.keys()),
                 "Type": [type_chip(c.role) for c in profile.columns.values()]}
            )
            st.dataframe(schema_df, hide_index=True)
        for jk in join_keys:
            st.markdown(
                f'<div class="in-join">🔗 {html.escape(describe_join(jk))}</div>',
                unsafe_allow_html=True,
            )
        if not join_keys and len(profiles) > 1:
            st.caption("We didn't find a common column to link these files on yet — "
                        "you can still ask questions about each one individually.")

# ------------------------------------------------------------ 3c: overview
else:
    specs = analysis["specs"]
    chart_results = analysis["chart_results"]

    headline = next((r for r, s in zip(chart_results, specs) if s.recipe == RECIPE_HEADLINE_METRICS), None)
    chart_items = [(s, r) for s, r in zip(specs, chart_results) if s.recipe != RECIPE_HEADLINE_METRICS]

    if headline and headline.metrics:
        cols = st.columns(len(headline.metrics))
        for col, m in zip(cols, headline.metrics):
            col.metric(m["label"], format_metric_value(m["value"]))

    if analysis["findings"]:
        with st.container(border=True):
            st.markdown("**✎ What stands out**")
            for text in analysis["phrasings"]:
                c1, c2 = st.columns([5, 1])
                c1.markdown(text)
                c2.markdown('<a href="#charts" style="font-size:12px">Show me →</a>', unsafe_allow_html=True)

    st.markdown('<a name="charts"></a>', unsafe_allow_html=True)
    if chart_items:
        # The first chart is always visible, at the same card size as the
        # rest (st.columns(3), only the first column used) — a real chart
        # sitting right here is a much stronger "there's more to see" signal
        # than a label on a fully collapsed section, which one user missed
        # entirely, not realizing it was clickable.
        first_col, _, _ = st.columns(3)
        with first_col:
            _render_chart_card(0, *chart_items[0])

        # Indexed from 1, not 0: the first chart above already claimed
        # index 0 for its widget keys, and these must stay unique.
        remaining = list(enumerate(chart_items[1:], start=1))
        if remaining:
            label = more_charts_label([spec.title for _, (spec, _) in remaining])
            with st.expander(label, expanded=False):
                for row_start in range(0, len(remaining), 3):
                    row = remaining[row_start:row_start + 3]
                    # Always 3 columns, even on a trailing row with fewer
                    # than 3 charts: st.columns(len(row)) would stretch a
                    # lone leftover chart to the full row width instead of
                    # staying card-sized.
                    cols = st.columns(3)
                    for col, (idx, (spec, result)) in zip(cols, row):
                        with col:
                            _render_chart_card(idx, spec, result)

st.divider()

# ------------------------------------------------------------------ Q&A ---
st.subheader("Ask a question")
st.caption("Type a question below in plain English — Crosswalk writes and runs the "
           "analysis for you, and always shows its work so you can check it.")

pending = st.session_state.pending_question
if pending:
    st.session_state.pending_question = None
    _ask(pending, frames)

chat_question = st.chat_input("Ask anything about your data…")
if chat_question:
    _ask(chat_question, frames)

history = st.session_state.qa_history
history_sorted = sorted(history, key=lambda h: not h["pinned"])

for i, entry in enumerate(history_sorted):
    question, result = entry["question"], entry["result"]
    state = classify_answer_state(result)

    with st.container(border=True):
        pin_col, q_col = st.columns([0.06, 0.94])
        with pin_col:
            pin_label = "📌" if entry["pinned"] else "📍"
            if st.button(pin_label, key=f"pin_{i}_{question}", help="Pin to top"):
                entry["pinned"] = not entry["pinned"]
                st.rerun()
        with q_col:
            # html.escape + a styled div, not st.markdown(f"**{question}**"):
            # a typed question containing markdown syntax (stray * or _)
            # would otherwise render mangled instead of as literal text.
            st.markdown(f'<div style="font-weight:700">{html.escape(question)}</div>',
                        unsafe_allow_html=True)

        if state == "error":
            st.error("Something went wrong while answering that — the analysis "
                      "hit an unexpected snag. Try rephrasing the question, or retry as-is.")
            with st.expander("Technical details"):
                st.code(result.error, language="text")
                st.code(result.code, language="python")
            if st.button("Retry", key=f"retry_{i}_{question}"):
                st.session_state.pending_question = question
                st.rerun()

        elif state == "calm":
            st.info(result.result)
            suggestions_here = suggested_questions(profiles, join_keys)
            if suggestions_here:
                st.caption("Try one of these instead:")
                chip_cols = st.columns(len(suggestions_here))
                for chip_col, sugg in zip(chip_cols, suggestions_here):
                    if chip_col.button(sugg, key=f"calmsugg_{i}_{sugg}"):
                        st.session_state.pending_question = sugg
                        st.rerun()

        else:
            if result.figure is not None:
                st.pyplot(result.figure)
            if isinstance(result.result, (pd.DataFrame, pd.Series)):
                st.dataframe(result.result)
            elif result.result is not None:
                # Same reasoning as the question header above: the model's
                # own text answer could itself contain markdown syntax, and
                # st.markdown would render that instead of showing it as-is.
                st.markdown(
                    f'<div style="font-size:26px;font-weight:700;margin:6px 0">'
                    f'{html.escape(str(result.result))}</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("How this was calculated"):
                st.code(result.code, language="python")

            action_cols = st.columns([1, 1, 6])
            if isinstance(result.result, (pd.DataFrame, pd.Series)):
                as_df = result.result.to_frame() if isinstance(result.result, pd.Series) else result.result
                action_cols[0].download_button(
                    "Export CSV", data=dataframe_to_csv_bytes(as_df),
                    file_name=safe_download_filename(question), mime="text/csv",
                    key=f"export_{i}_{question}",
                )

if not history:
    suggestions = suggested_questions(profiles, join_keys)
    if suggestions:
        st.caption("Try asking:")
        cols = st.columns(len(suggestions))
        for col, sugg in zip(cols, suggestions):
            if col.button(sugg, key=f"sugg_{sugg}"):
                st.session_state.pending_question = sugg
                st.rerun()
