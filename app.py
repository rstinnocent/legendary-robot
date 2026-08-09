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
from core import answer_question, get_backend, load_files
from findings import compute_findings, phrase_findings
from profiler import detect_join_keys, profile_frames
from ui_helpers import (
    chart_caption,
    chart_section_label,
    classify_answer_state,
    dataframe_to_csv_bytes,
    format_metric_value,
    question_for_spec,
    safe_download_filename,
    suggested_questions,
    type_chip,
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"

st.set_page_config(page_title="Crosswalk", page_icon="✎", layout="wide")

st.markdown(
    """
<style>
.in-hdr{display:flex;justify-content:space-between;align-items:center;
  border-bottom:2px solid #1a1a1a;padding-bottom:10px;margin-bottom:18px}
.in-logo{font-size:22px;font-weight:700}
.in-tag{font-size:12px;color:rgba(0,0,0,.55);border:1px dashed rgba(0,0,0,.4);
  padding:3px 9px;border-radius:10px}
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

    Also only reload when `file_bytes` actually contains a new filename: a
    Streamlit rerun fires on every widget interaction, and the sidebar
    uploader keeps reporting the same files on every one of them — without
    this guard, every rerun (asking a question, anything) would look like a
    fresh upload and wipe the overview and Q&A history.

    Known limitation: re-uploading a *different* file under a name already
    seen (e.g. a corrected employees.csv) is treated as nothing new and
    silently ignored, rather than replacing the earlier version — good
    enough for "add more files", not a general re-upload/replace feature.
    """
    if not file_bytes:
        return
    already_ingested = st.session_state.get("_ingested_filenames", set())
    if set(file_bytes.keys()) <= already_ingested:
        return
    try:
        new_frames = load_files(file_bytes)
    except Exception as exc:
        st.error(f"Couldn't read one of the files: {exc}")
        return
    st.session_state.frames = {**st.session_state.get("frames", {}), **new_frames}
    st.session_state._ingested_filenames = already_ingested | set(file_bytes.keys())
    st.session_state.analysis = None
    st.session_state.qa_history = []


def _header(tag: str) -> None:
    st.markdown(
        f'<div class="in-hdr"><div class="in-logo">✎ Crosswalk</div>'
        f'<div class="in-tag">{html.escape(tag)}</div></div>',
        unsafe_allow_html=True,
    )


def _ask(question: str, frames: dict) -> None:
    """Run one question through the Q&A engine and prepend it to history."""
    try:
        backend = _build_backend()
    except Exception as exc:
        st.error(f"Couldn't set up the LLM backend: {exc}")
        return
    with st.spinner("Thinking..."):
        try:
            result = answer_question(question, frames, backend)
        except Exception as exc:  # network / API errors from the backend itself
            st.error(f"The LLM backend call failed: {exc}")
            return
    st.session_state.qa_history.insert(0, {"question": question, "result": result, "pinned": False})


# ---------------------------------------------------------------- state ---
for key, default in [
    ("frames", {}), ("analysis", None), ("qa_history", []),
    ("_ingested_filenames", set()), ("pending_question", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Settings")
    backend_choice = st.selectbox("LLM backend", ["Groq (free, hosted)", "Ollama (local)"])
    st.session_state.backend_choice = backend_choice

    if backend_choice.startswith("Groq"):
        st.session_state.api_key = st.text_input(
            "Groq API key", type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Free, no credit card required — get one at console.groq.com/keys",
        )
        st.session_state.model = st.selectbox(
            "Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
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
        for name, df in frames_now.items():
            st.caption(f"**{name}** — {len(df):,} rows · {len(df.columns)} cols")
    else:
        uploaded = None

frames = st.session_state.frames

# ---------------------------------------------------------- empty state ---
if not frames:
    _header("no files yet")
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
_header(f"{len(frames)} files · {total_rows:,} rows")

profiles = profile_frames(frames)
join_keys = detect_join_keys(frames, profiles)

_, mid, _ = st.columns([1, 1, 1])
with mid:
    analyze_clicked = st.button("Analyze", type="primary", width="stretch")

if analyze_clicked:
    try:
        backend = _build_backend()
    except Exception as exc:
        st.error(f"Couldn't set up the LLM backend: {exc}")
        st.stop()

    with st.spinner("Profiling data and building the overview..."):
        try:
            raw_plan = generate_chart_plan(profiles, join_keys, backend)
        except ChartPlanError as exc:
            st.error(f"The model's chart plan couldn't be parsed: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"The LLM backend call failed: {exc}")
            st.stop()

        validated = validate_plan(raw_plan, profiles, join_keys)
        chart_results = execute_chart_plan(validated, frames, profiles)
        findings = compute_findings(frames, profiles, join_keys)
        phrasings = phrase_findings(findings, backend)

    st.session_state.analysis = {
        "specs": validated, "chart_results": chart_results,
        "findings": findings, "phrasings": phrasings,
    }

analysis = st.session_state.analysis

# ---------------------------------------------------- 3b: schema preview --
if not analysis:
    for name, profile in profiles.items():
        st.markdown(f"**{name}**")
        schema_df = pd.DataFrame(
            {"Column": list(profile.columns.keys()),
             "Type": [type_chip(c.role) for c in profile.columns.values()]}
        )
        st.dataframe(schema_df, hide_index=True)
    for jk in join_keys:
        st.markdown(
            f'<div class="in-join">🔗 {html.escape(jk.left_frame)}.{html.escape(jk.left_column)} '
            f"→ {html.escape(jk.right_frame)}.{html.escape(jk.right_column)} "
            f"({jk.overlap_count} shared values)</div>",
            unsafe_allow_html=True,
        )
    if not join_keys:
        st.caption("No shared identifiers detected across files yet.")
    st.caption("Click **Analyze** above for an overview.")

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
        label = chart_section_label([spec.title for spec, _ in chart_items])
        with st.expander(label, expanded=False):
            hero_spec, hero_result = chart_items[0]
            # Constrained to ~2/3 width rather than the full page: st.pyplot
            # preserves the figure's own aspect ratio when stretched, so a
            # full-width container blows the height up proportionally too.
            hero_col, _ = st.columns([2, 1])
            with hero_col, st.container(border=True):
                st.markdown(f"**{hero_spec.title}**")
                if hero_result.error:
                    st.warning(hero_result.error)
                elif hero_result.figure is not None:
                    st.pyplot(hero_result.figure)
                elif hero_result.table is not None:
                    st.dataframe(hero_result.table)
                caption = chart_caption(hero_spec, hero_result)
                if caption:
                    st.markdown(f'<div class="in-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
                act1, act2 = st.columns(2)
                with act1:
                    with st.popover("view recipe"):
                        st.code(f"{hero_spec.recipe}({hero_spec.__dict__})", language="python")
                with act2:
                    q = question_for_spec(hero_spec)
                    if q and st.button("ask about this", key="ask_hero"):
                        st.session_state.pending_question = q
                        st.rerun()

            # Indexed by position, not spec.title: the LLM's chart plan isn't
            # validated for title uniqueness, and two cards sharing a title
            # would collide on a title-based widget key and crash the render.
            rest = list(enumerate(chart_items[1:]))
            for row_start in range(0, len(rest), 3):
                row = rest[row_start:row_start + 3]
                cols = st.columns(len(row))
                for col, (idx, (spec, result)) in zip(cols, row):
                    with col:
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
                                st.markdown(f'<div class="in-caption">{html.escape(caption)}</div>',
                                            unsafe_allow_html=True)
                            b1, b2 = st.columns(2)
                            with b1:
                                with st.popover("recipe", key=f"recipe_{idx}"):
                                    st.code(f"{spec.recipe}({spec.__dict__})", language="python")
                            with b2:
                                q = question_for_spec(spec)
                                if q and st.button("ask", key=f"ask_{idx}"):
                                    st.session_state.pending_question = q
                                    st.rerun()

st.divider()

# ------------------------------------------------------------------ Q&A ---
st.subheader("Ask a question")

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
            st.error("⚠ Something went wrong running this query")
            st.code(result.error, language="text")
            with st.expander("View generated code"):
                st.code(result.code, language="python")
            if st.button("Retry", key=f"retry_{i}_{question}"):
                st.session_state.pending_question = question
                st.rerun()

        elif state == "calm":
            st.info(result.result)
            suggestions_here = suggested_questions(profiles, join_keys)
            if suggestions_here:
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
