"""
app.py — Streamlit UI for the AI-Powered Data Q&A app.

Run with:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from chart_plan import ChartPlanError, generate_chart_plan, validate_plan
from charts import execute_chart_plan
from core import answer_question, get_backend, load_files
from findings import compute_findings, phrase_findings
from profiler import detect_join_keys, profile_frames

st.set_page_config(page_title="Data Q&A", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Data Q&A")
st.caption("Upload CSV/Excel files and ask questions about them in plain English.")

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Settings")
    backend_choice = st.selectbox("LLM backend", ["Groq (free, hosted)", "Ollama (local)"])

    if backend_choice.startswith("Groq"):
        api_key = st.text_input(
            "Groq API key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Free, no credit card required — get one at console.groq.com/keys",
        )
        model = st.selectbox(
            "Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
        )
    else:
        api_key = None
        model = st.text_input("Ollama model", value="llama3.2")
        st.caption("Requires Ollama running locally (`ollama serve`).")

    st.divider()
    uploaded = st.file_uploader(
        "Upload data files", type=["csv", "xlsx", "xls"], accept_multiple_files=True
    )
    st.divider()
    st.caption(
        "Sample files in `sample_data/` (employees, attendance, exits) — "
        "upload all three for a cross-file demo."
    )


def _build_backend():
    return (
        get_backend("groq", api_key=api_key, model=model)
        if backend_choice.startswith("Groq")
        else get_backend("ollama", model=model)
    )


# ------------------------------------------------------------ load data ---
if "frames" not in st.session_state:
    st.session_state.frames = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if uploaded:
    file_bytes = {f.name: f.getvalue() for f in uploaded}
    # Only reload (and drop any existing overview) when the actual set of
    # uploaded files changes — Streamlit reruns this whole script on every
    # widget interaction, and `uploaded` stays truthy across all of them, so
    # reloading unconditionally here would wipe out the Analyze results on
    # the very next click of anything else on the page.
    signature = tuple(sorted(file_bytes.keys()))
    if signature != st.session_state.get("_frames_signature"):
        try:
            st.session_state.frames = load_files(file_bytes)
            st.session_state._frames_signature = signature
            st.session_state.analysis = None
        except Exception as exc:
            st.error(f"Couldn't read one of the files: {exc}")

frames = st.session_state.frames

if not frames:
    st.info("Upload one or more CSV/Excel files in the sidebar to get started.")
    st.stop()

# ------------------------------------------------------ preview & schema --
with st.expander("📁 Loaded files", expanded=False):
    for name, df in frames.items():
        st.markdown(f"**`{name}`** — {len(df)} rows × {len(df.columns)} columns")
        st.dataframe(df.head(5), use_container_width=True)

# --------------------------------------------------------- auto-analysis --
st.subheader("Overview")
st.caption(
    "Profiles your data locally, asks the model which charts are worth "
    "showing, then runs those charts deterministically — the model never "
    "computes a number itself."
)

if st.button("Analyze", type="secondary"):
    try:
        backend = _build_backend()
    except Exception as exc:
        st.error(f"Couldn't set up the LLM backend: {exc}")
        st.stop()

    with st.spinner("Profiling data and building the overview..."):
        profiles = profile_frames(frames)
        join_keys = detect_join_keys(frames, profiles)
        try:
            raw_plan = generate_chart_plan(profiles, join_keys, backend)
        except ChartPlanError as exc:
            st.error(f"The model's chart plan couldn't be parsed: {exc}")
            st.stop()
        except Exception as exc:  # backend/network errors
            st.error(f"The LLM backend call failed: {exc}")
            st.stop()

        validated = validate_plan(raw_plan, profiles, join_keys)
        chart_results = execute_chart_plan(validated, frames, profiles)
        findings = compute_findings(frames, profiles, join_keys)
        phrasings = phrase_findings(findings, backend)

    st.session_state.analysis = {
        "chart_results": chart_results,
        "findings": findings,
        "phrasings": phrasings,
    }

analysis = st.session_state.analysis
if analysis:
    chart_results = analysis["chart_results"]
    headline = next((r for r in chart_results if r.metrics), None)
    other_charts = [r for r in chart_results if not r.metrics]

    if headline:
        cols = st.columns(len(headline.metrics))
        for col, m in zip(cols, headline.metrics):
            value = m["value"]
            display = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
            col.metric(m["label"], display)

    if other_charts:
        chart_cols = st.columns(2)
        for i, r in enumerate(other_charts):
            with chart_cols[i % 2]:
                st.markdown(f"**{r.title}**")
                if r.error:
                    st.warning(r.error)
                elif r.figure is not None:
                    st.pyplot(r.figure)

    if analysis["findings"]:
        st.markdown("#### What stands out")
        for text in analysis["phrasings"]:
            st.markdown(f"- {text}")

st.divider()

# ---------------------------------------------------------------- Q&A -----
st.subheader("Ask a question")
question = st.text_input(
    "Try: \"What's total revenue by region?\" or \"Plot monthly sales trend\"",
    key="question_input",
)
ask = st.button("Ask", type="primary")

if ask and question.strip():
    try:
        backend = _build_backend()
    except Exception as exc:
        st.error(f"Couldn't set up the LLM backend: {exc}")
        st.stop()

    with st.spinner("Thinking..."):
        try:
            result = answer_question(question, frames, backend)
        except Exception as exc:  # network / API errors from the backend itself
            st.error(f"The LLM backend call failed: {exc}")
            st.stop()
    st.session_state.history.insert(0, (question, result))

for q, res in st.session_state.history:
    st.markdown(f"**Q: {q}**")
    if res.error:
        # Wrapped in backticks: st.error renders markdown, so an error mentioning
        # `__import__` would otherwise display as a bolded "import" and hide the
        # actual name. Cost us a debugging session once already.
        st.error(f"`{res.error}`")
    else:
        if res.figure is not None:
            st.pyplot(res.figure)
        if isinstance(res.result, (pd.DataFrame, pd.Series)):
            st.dataframe(res.result, use_container_width=True)
        elif res.result is not None:
            st.write(res.result)
    with st.expander("Generated code"):
        st.code(res.code, language="python")
    st.divider()
