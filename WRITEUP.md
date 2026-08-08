# Write-up

## Approach

The task has three moving parts: getting tabular data into the app, letting an
LLM reason about it in plain English, and showing the answer back clearly. I
scoped this as a "text-to-pandas-code" agent rather than a general chatbot:
the LLM sees each uploaded file's schema (columns, dtypes, a few sample rows)
and is asked to write short pandas/matplotlib code that answers the question,
which is then executed against the real dataframes. This is deliberately
narrower than a general-purpose agent, but it's the right narrowness for
"analytical questions about uploaded data" — it's grounded in the actual
values rather than the model guessing at numbers, it composes naturally for
cross-file joins, and it produces a chart or a table without needing separate
code paths for each.

## Key decisions

**Custom agent over a library like PandasAI.** The brief explicitly allows
PandasAI, and I considered it. I went custom instead because the whole
pipeline is small enough (~250 lines) to walk through in the panel
conversation, and because showing the generated code next to every answer —
rather than hiding it inside a library — makes the app's reasoning auditable,
which felt like the more defensible design for a data-analysis tool.

**Groq's free tier as the default LLM, with local Ollama as a fallback.**
The brief's constraint ("open-source AI models... no paid APIs or metered
credits") is genuinely ambiguous — coding *assistants* like Claude/ChatGPT are
explicitly fine to use for building, but it's less clear whether a *hosted*
API is allowed for the app's own runtime engine, even a free one. Groq's free
tier needs no credit card and only serves open-weight models, so it satisfies
the constraint under either reading. I built in a local Ollama backend as a
second option specifically so the grader can run this with zero external
dependency at all if they'd rather not sign up for anything. I sent the
recruiter two pointed questions on this exact ambiguity rather than guessing
silently — and built against the stricter interpretation in the meantime so
the submission wouldn't be blocked on a reply.

**A regex safety filter over full sandboxing.** LLM-generated code is
executed with a restricted builtins list and a blocklist for `os`, `eval`,
file I/O, and dunder access. This is intentionally blunt — good enough for a
take-home demo with a small, known set of dependencies, not something I'd
ship for untrusted multi-tenant use (see below).

## What I'd build next

1. **Real sandboxing** — run generated code in a subprocess or container with
   a resource/time limit instead of a regex filter, before this ever sees
   untrusted input.
2. **Larger files** — everything currently loads into memory as pandas
   DataFrames; beyond a few hundred MB I'd move to DuckDB or chunked reads.
3. **An eval set** — a fixed list of ~20 representative questions (totals,
   filters, trends, joins, "no answer exists") run against known-good answers
   on every change, instead of manual spot-checking.
4. **Conversation memory** — follow-up questions ("now break that down by
   product") currently start fresh each time; carrying prior context would
   make multi-turn analysis much more natural.
5. **Better error recovery** — if generated code fails, retry once with the
   error message fed back to the LLM before surfacing it to the user.
