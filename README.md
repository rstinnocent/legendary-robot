# Crosswalk

Upload CSV/Excel files that belong together, get an automatic overview —
headline metrics, a handful of charts, and the findings that actually stand
out — then ask follow-up questions in plain English and get back an answer:
a number, a table, or a chart, plus the exact pandas code that produced it.

Named for the data term: a *crosswalk* is a table that maps fields across
datasets so they can be joined. Detecting those joins automatically, across
whatever files you upload, is the app's differentiator — see
[How this maps to the acceptance criteria](#how-this-maps-to-the-acceptance-criteria).

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Free Groq API key, no credit card: https://console.groq.com/keys
export GROQ_API_KEY=your_key_here        # or paste it into the app sidebar

streamlit run app.py
```

Then open the local URL Streamlit prints. With no files loaded yet, drop
CSV/Excel files in or click **"or try it with sample HR data →"** to load
`sample_data/` instead. Once files are in, the sidebar lists them and the
main area previews each file's schema plus any cross-file join keys it
found; click the centered **Analyze** button for an automatic overview —
headline metrics, a "what stands out" panel, and a collapsible charts
section (closed by default, labeled with the actual chart titles so you
know what's behind the click before opening it) — before asking anything at
all. Then try a question like:

- "What is the headcount in each department?"
- "Which department has the highest average CTC?"
- "How many people from each department have exited?" *(joins exits → employees)*
- "Compare the average CTC of employees who have exited versus those still here"
- "Plot the trend of total leave days by month"
- "What is the average performance rating by department?" *(there is no such
  column — the app should say so rather than invent a number)*

Every chart card also has an **"ask"** button that runs that chart's
underlying question through the same Q&A engine and drops it into the
question history below — and the Q&A history itself supports pinning an
answer to the top and exporting a table result as CSV.

No Groq key handy, or want zero external dependency? Switch the sidebar to
**Ollama (local)** — run `ollama serve` and `ollama pull llama3.2` first.

## Deploying (Streamlit Community Cloud)

1. [share.streamlit.io](https://share.streamlit.io) → **Deploy a public app** →
   pick this repo and the branch you want live.
2. **Main file path: `app.py`** — it sits at the repo root, not in a
   subfolder, so don't prefix it with the repo or folder name.
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
   This is what lets an evaluator open the app and start asking questions
   immediately, with no key to find or paste in — the sidebar's API key
   field only appears at all when this secret is *not* set.

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
| UI logic | `ui_helpers.py` | Pure functions the UI runs on — chart captions, calm/error/normal answer classification, join-prioritized suggested questions — kept out of `app.py` so they're unit-testable the same way the engine is |
| Correctness | 12-case Q&A eval set + 3-check auto-analysis eval (`evals/`), all against hand-computed or independently-recomputed answers | "Correct" is the product. See below |
| Testing | pytest — 169 tests, LLM mocked via `FakeBackend` | Verifies loading, joins, charts, the safety filter, the profiler's column-role logic, chart-plan validation, UI display logic, and both eval sets — all without hitting a live API |

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

The first chart always renders immediately, no click required, at the same
card size every chart uses — a real chart sitting on the page is a much
stronger sign that there's more worth exploring than a label ever was; an
early version collapsed all the charts by default, and a user simply
missed them, not realizing the label was clickable. Any remaining charts
sit in a 3-across grid inside a collapsed section labeled with an explicit
invitation and their actual titles (e.g. "👀 See 3 more charts: Grade
Distribution, Attendance Over Time +1 more — click to view") rather than a
plain "Charts (4)", so it's clear there's something worth a click, and the
Q&A section underneath isn't pushed off-screen by a wall of charts on
first load. Every card, first or otherwise, carries an "ask" button that
turns that chart's spec back into a plain-English question
(`ui_helpers.question_for_spec`) and runs it through the real Q&A engine.

## Evals — how I know the answers are right

A data Q&A tool that is confidently wrong is worse than one that says nothing,
so correctness is measured rather than eyeballed.

```bash
pytest                          # 145 unit tests, no API key needed
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
app.py                    # Streamlit UI — upload, schema/join preview, Analyze overview, Q&A
core.py                   # loading, prompt building, safe execution, LLM backends
test_core.py              # pytest suite for the Q&A engine
profiler.py                # column roles + cross-file join detection (no LLM)
chart_plan.py              # LLM chart-plan call + validation
charts.py                  # deterministic execution of the 6 chart recipes
findings.py                # "what stands out": candidate generation + LLM phrasing
ui_helpers.py               # pure display logic app.py runs on (captions, states, suggestions)
test_profiler.py / test_chart_plan.py / test_charts.py / test_findings.py / test_ui_helpers.py
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
