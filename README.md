# AI-Powered Data Q&A

Upload one or more CSV/Excel files, get an automatic overview — headline
metrics, a handful of charts, and the findings that actually stand out —
then ask follow-up questions in plain English and get back an answer: a
number, a table, or a chart, plus the exact pandas code that produced it.

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Free Groq API key, no credit card: https://console.groq.com/keys
export GROQ_API_KEY=your_key_here        # or paste it into the app sidebar

streamlit run app.py
```

Then open the local URL Streamlit prints, upload all three files from
`sample_data/` (or your own), and click **Analyze** for an automatic
overview — headline metrics, five to six charts the model judged worth
showing for this specific data, and a "what stands out" panel — before
asking anything at all. Then try a question like:

- "What is the headcount in each department?"
- "Which department has the highest average CTC?"
- "How many people from each department have exited?" *(joins exits → employees)*
- "Compare the average CTC of employees who have exited versus those still here"
- "Plot the trend of total leave days by month"
- "What is the average performance rating by department?" *(there is no such
  column — the app should say so rather than invent a number)*

No Groq key handy, or want zero external dependency? Switch the sidebar to
**Ollama (local)** — run `ollama serve` and `ollama pull llama3.2` first.

## Sample data

Three files that mirror what an HR/People Ops analyst actually exports, and
that only become useful when joined:

| File | Rows | Contents |
|---|---|---|
| `employees.csv` | 40 | employee_id, department, grade, location, manager_id, date_of_joining, annual_ctc, gender |
| `attendance.csv` | 240 | monthly working days, days present, leave days, WFH days (Jan–Jun 2026) |
| `exits.csv` | 9 | exit date, voluntary/involuntary, reason |

`exits.csv` deliberately uses `DD-MM-YYYY` while the other two use
`YYYY-MM-DD` — real exports are inconsistent like this, and naive date parsing
gets it silently wrong. There is an eval case for exactly this.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) | Fastest path to a usable multi-file upload + chat UI within the time box |
| LLM (default) | [Groq](https://groq.com) free tier — Llama 3.3 70B / GPT-OSS 120B | Genuinely free, **no credit card required**, serves only open-weight models, fast enough that the UI doesn't feel laggy |
| LLM (fallback) | [Ollama](https://ollama.com) running locally | Zero external dependency at all, for a fully offline compliant run |
| Analysis engine | Custom "text-to-pandas-code" agent (`core.py`) | The LLM writes short pandas/matplotlib code against the uploaded dataframes, executed in a restricted namespace. Chosen over PandasAI so every answer is inspectable — the generated code is shown in the UI next to the result, and the whole pipeline is ~250 lines I can walk through line by line |
| Auto-analysis engine | Local pandas profiler + constrained chart-plan (`profiler.py`, `chart_plan.py`, `charts.py`, `findings.py`) | The overview screen shown before any question is asked. The model picks from a fixed menu of 6 chart recipes and only *phrases* pre-computed findings — it never produces a number itself. See below |
| Correctness | 12-case Q&A eval set + 3-check auto-analysis eval (`evals/`), all against hand-computed or independently-recomputed answers | "Correct" is the product. See below |
| Testing | pytest — 111 tests, LLM mocked via `FakeBackend` | Verifies loading, joins, charts, the safety filter, the profiler's column-role logic, chart-plan validation, and both eval sets — all without hitting a live API |

## How this maps to the acceptance criteria

- **Multi-file upload** — `st.file_uploader(..., accept_multiple_files=True)`;
  each file becomes a named pandas DataFrame (`employees`, `attendance`, ...).
- **Cross-file analysis** — every uploaded file's schema (columns, dtypes,
  sample rows) goes into the Q&A prompt together, so the model can `merge()`
  and `groupby()` across files (two- and three-file joins are both covered by
  evals); separately, the auto-analysis profiler detects join keys by actual
  value overlap and the chart plan is required to use one when available.
- **Visual insights** — in Q&A, the model builds a matplotlib chart instead of
  a scalar/table when that's the clearer answer; on the overview screen, the
  chart plan always includes several charts plus headline metrics and a
  "what stands out" panel, no question required.

## Auto-analysis — how "what to show" stays honest

Click **Analyze** and the app profiles the data locally (dtype, distinct
count, null rate, a role per column — identifier / dimension / measure /
date / skip — and cross-file join keys by actual value overlap, not name
matching). One LLM call then picks 5-7 charts from a fixed recipe set
(`count_by_dimension`, `measure_by_dimension`, `measure_over_time`,
`crosstab`, `headline_metrics`, `cross_file_measure`) — its only job is
judging what's interesting. Before anything runs, `validate_plan` drops any
spec that references a column that doesn't exist, uses an identifier as a
dimension, or otherwise doesn't fit its recipe, and forces in a cross-file
chart if a join key exists and the model didn't pick one. Every surviving
spec is then executed by a small deterministic function per recipe — no
generated code in this path.

"What stands out" works the same way in reverse: four pandas functions look
for the largest gap between groups, a time-series trend, a group beyond
~1.5 standard deviations, and a joined-vs-not-joined difference (e.g.
leavers vs. stayers), rank them by effect size, and keep the top three. The
model's only involvement is rewriting those three already-computed findings
into one plain sentence each — never computing them — and if that call fails
or returns something malformed, the deterministic description is shown
as-is instead.

## Evals — how I know the answers are right

A data Q&A tool that is confidently wrong is worse than one that says nothing,
so correctness is measured rather than eyeballed.

```bash
pytest                          # 111 unit tests, no API key needed
python evals/run_eval.py        # runs the 12-case Q&A set against the live model
python evals/run_eval.py --runs 3            # repeat to see run-to-run variance
python evals/eval_auto_analysis.py           # checks the auto-analysis pipeline live
```

`run_eval.py` writes a scored table to `evals/RESULTS.md`. The set covers
simple aggregates, group-bys, two-file and three-file joins, a chart question,
the `DD-MM-YYYY` date trap, and one **unanswerable** question — the data has no
performance-rating column, and inventing a plausible number there is graded as
a failure, not a near miss.

The eval set is itself unit-tested (`evals/test_eval_set.py`): every case has a
hand-written correct solution that must grade as a pass. That catches both a
mistyped expected answer and a grader too strict to accept a right one — it
already caught a bug where pandas rendered a correct average as `9.08e+05` and
the grader would have failed it.

`eval_auto_analysis.py` writes to `evals/AUTO_ANALYSIS_RESULTS.md` and checks
three properties against a live run: every executed chart spec references a
real column, a cross-file chart is present given the sample data has join
keys, and each of the three "what stands out" findings — recomputed from
scratch, independently of `findings.py`'s own generator functions — matches
the number already in the finding, and that number is still present in the
model's phrased sentence.

## Tooling & constraint compliance

- No paid or metered API is used. Groq's free tier requires no credit card and
  only serves open-weight models (Llama, GPT-OSS) — no GPT/Claude/Gemini on it.
  Ollama is a fully local, offline fallback for the same reason.
- AI coding assistants were used to build and debug this repo, per the brief —
  the runtime constraint applies to the app's own Q&A engine.

## Project structure

```
app.py                    # Streamlit UI — Q&A plus the Analyze overview
core.py                   # loading, prompt building, safe execution, LLM backends
test_core.py              # pytest suite for the Q&A engine
profiler.py                # column roles + cross-file join detection (no LLM)
chart_plan.py              # LLM chart-plan call + validation
charts.py                  # deterministic execution of the 6 chart recipes
findings.py                # "what stands out": candidate generation + LLM phrasing
test_profiler.py / test_chart_plan.py / test_charts.py / test_findings.py
evals/
  eval_set.py             # the 12 Q&A questions + expected answers
  run_eval.py             # runs them against a live model, writes RESULTS.md
  test_eval_set.py        # validates the eval set with reference solutions
  eval_auto_analysis.py   # live checks for the auto-analysis pipeline
sample_data/              # employees.csv, attendance.csv, exits.csv
requirements.txt
.env.example
```

## Known limitations

See `WRITEUP.md`. In short: generated code is guarded by a regex blocklist and
restricted builtins rather than a real sandbox; everything loads into memory,
so this is not for files beyond a few hundred MB; and each question starts
fresh, with no memory of the previous one.
