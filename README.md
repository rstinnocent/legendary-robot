# AI-Powered Data Q&A

Upload one or more CSV/Excel files, ask questions about them in plain English,
get back an answer — a number, a table, or a chart — plus the exact pandas
code that produced it.

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Free Groq API key, no credit card: https://console.groq.com/keys
export GROQ_API_KEY=your_key_here

streamlit run app.py
```

Then open the local URL Streamlit prints, upload files from `sample_data/`
(or your own), and ask something like:

- "What's total revenue by region?"
- "Which product sold the most units in March?"
- "Plot monthly revenue trend"
- "Which customers have returned an order?" *(joins across all three sample files)*

No Groq key handy, or want zero external dependency? Switch the sidebar to
**Ollama (local)** — run `ollama serve` and `ollama pull llama3.2` first.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) | Fastest path to a usable multi-file upload + chat UI within the time box |
| LLM (default) | [Groq](https://groq.com) free tier — Llama 3.3 70B / GPT-OSS 120B | Genuinely free, **no credit card required**, serves only open-weight models, and is fast enough that the UI doesn't feel laggy |
| LLM (fallback) | [Ollama](https://ollama.com) running locally | Zero external dependency at all, for a fully offline compliant run |
| Analysis engine | Custom "text-to-pandas-code" agent (`core.py`) | The LLM writes short pandas/matplotlib code against the uploaded dataframes, which is then executed in a restricted namespace. Chosen over a library like PandasAI so every answer is inspectable — the generated code is shown in the UI next to the result, and the whole pipeline is ~250 lines I can walk through line by line. |
| Testing | pytest, LLM mocked via a `FakeBackend` | Lets the data-loading, join, chart, and safety-filter logic be verified without hitting a live API |

## How this maps to the acceptance criteria

- **Multi-file upload** — `st.file_uploader(..., accept_multiple_files=True)`;
  each file becomes a named pandas DataFrame (`orders`, `customers`, ...).
- **Cross-file analysis** — every uploaded file's schema (columns, dtypes,
  sample rows) is included in the LLM prompt, so it can `merge()`/`groupby()`
  across files when a question calls for it (see the returns-join example above).
- **Visual insights** — the LLM is free to build a matplotlib chart instead of
  a scalar/table when that's the clearer answer; the app renders whichever one
  comes back.

## Tooling & constraint compliance

- No paid or metered API is used. Groq's free tier requires no credit card
  and only serves open-weight models (Llama, GPT-OSS, etc.) — no GPT/Claude/Gemini
  on it. Ollama is a fully local, offline fallback for the same reason.
- AI coding assistants (Claude, Cursor) were used to build and debug this repo,
  per the brief — the runtime constraint applies only to the app's own Q&A engine.

## Project structure

```
app.py            # Streamlit UI
core.py           # data loading, prompt building, safe code execution, LLM backends
test_core.py       # pytest suite (10 tests, run with `pytest`)
sample_data/       # orders.csv, customers.csv, returns.csv — for the demo
requirements.txt
.env.example
```

## Known limitations

See `WRITEUP.md` for what I'd build next — in short: sandboxing the generated
code more rigorously than a regex filter, handling larger files (currently
loads everything into memory), and a proper eval set of test questions instead
of ad hoc manual testing.
