# Write-up

## Approach

The brief fixes the *what* — upload files, ask in English, get a correct
answer — so the real ambiguity is **what counts as a good answer, and how you
know you're giving one**. I scoped this as a *text-to-pandas-code* agent
rather than a chatbot: the model sees each uploaded file's schema (columns,
dtypes, sample rows) and writes short pandas/matplotlib code, which is then
executed against the real dataframes. Narrower than a general agent, but the
right narrowness — the engine does the arithmetic instead of the model, joins
compose naturally across files, and one mechanism yields a number, a table, or
a chart without separate code paths.

I framed the user as an **HR/People Ops analyst** who exports three or four
files a month and joins them by hand in Excel. That persona is what makes the
brief's own acceptance criteria coherent — multi-file upload and cross-file
analysis only matter if your data is scattered — and it's why `sample_data/`
ships as employees, attendance, and exits.

## Key decisions

**Custom agent over PandasAI.** The brief permits PandasAI and it would have
been faster. I went custom because the pipeline is ~250 lines I can walk
through line by line, and because the generated code is surfaced next to every
answer rather than hidden in a library. For a tool whose product problem is
*trust*, visible reasoning is the feature.

**Groq's free tier as default, local Ollama as fallback.** The constraint
("open-source models, no paid APIs or metered credits") is ambiguous about
whether a *free hosted* API is allowed for the app's own runtime. Groq needs no
credit card and serves only open-weight models, so it satisfies the strict
reading; Ollama gives a fully offline path for a grader who'd rather not sign
up for anything. I flagged the ambiguity to the recruiter and built against the
stricter interpretation meanwhile, so the submission was never blocked on a
reply.

**Measured correctness instead of asserting it.** The most important decision.
A system that's 70% accurate while sounding 100% confident is worse than no
system, so `evals/` pins 12 questions to answers computed independently by
hand — aggregates, group-bys, two- and three-file joins, a chart, a
`DD-MM-YYYY` date trap that naive parsing gets silently wrong, and one
**unanswerable** question (there is no performance-rating column; inventing a
number is graded as a failure). Run `python evals/run_eval.py` for a score.
The eval set is itself unit-tested against hand-written reference solutions,
which already caught a grader bug that would have failed a correct answer
rendered in scientific notation.

**A regex safety filter over full sandboxing.** Generated code runs with
restricted builtins and a blocklist for `os`, `eval`, file I/O and dunder
access. Deliberately blunt — adequate for a demo with known dependencies, not
something I'd ship for untrusted multi-tenant use.

## What I deliberately cut

- **Live dashboards / what-if calculators** — a real need, and the most
  seductive idea I considered, but a different product: "let me explore"
  rather than "tell me the number". It would have eaten the time box and left
  the three named criteria half-done.
- **Cleaning arbitrarily messy spreadsheets** — unbounded scope with no natural
  stopping point. Scoped to reasonably tabular data and said so.

## What I'd build next, in order

1. **Conversation memory** — "now break that down by grade" should work.
   Highest user-visible value; questions currently start fresh each time.
2. **Retry on error** — feed a failed execution's traceback back to the model
   once before surfacing it. Cheap, and should lift the eval score.
3. **Real sandboxing** — subprocess or container with resource/time limits,
   before this ever sees untrusted input.
4. **Widen the eval set to ~50 cases** and track score per model, so the model
   choice becomes evidence-based rather than a default.
5. **DuckDB behind the same interface** once files outgrow memory.
