"""
app.py — Streamlit UI for the AI-Powered Data Q&A app.

Run with:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from core import answer_question, get_backend, load_files

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
        "Sample files in `sample_data/` (orders, customers, returns) — "
        "try uploading all three for a cross-file demo."
    )

# ------------------------------------------------------------ load data ---
if "frames" not in st.session_state:
    st.session_state.frames = {}
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded:
    file_bytes = {f.name: f.getvalue() for f in uploaded}
    try:
        st.session_state.frames = load_files(file_bytes)
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

# ---------------------------------------------------------------- Q&A -----
st.subheader("Ask a question")
question = st.text_input(
    "Try: \"What's total revenue by region?\" or \"Plot monthly sales trend\"",
    key="question_input",
)
ask = st.button("Ask", type="primary")

if ask and question.strip():
    try:
        backend = (
            get_backend("groq", api_key=api_key, model=model)
            if backend_choice.startswith("Groq")
            else get_backend("ollama", model=model)
        )
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
        st.error(res.error)
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
