# Eval results

- Model: `llama-3.3-70b-versatile` (backend: `groq`)
- Dataset: `sample_data/` — employees, attendance, exits (3 files)
- Runs: 1
- **Score: 10/12 (83%)**

| ID | Capability | Result | Notes |
|---|---|---|---|
| E01 | simple aggregate | pass |  |
| E02 | simple aggregate | pass |  |
| E03 | group-by | pass |  |
| E04 | group-by + argmax | pass |  |
| E05 | filter + count | pass |  |
| E06 | single-file aggregate over many rows | **fail** | missing expected value(s) ['95.9']; got: '95.8984' |
| E07 | cross-file join (2 files) | pass |  |
| E08 | cross-file join (2 files) + comparison | pass |  |
| E09 | cross-file join (3 files) | pass |  |
| E10 | date-format trap + join | pass |  |
| E11 | chart / visual insight | pass |  |
| E12 | unanswerable — should decline, not hallucinate | **fail** | did not decline; answered: 'department Customer Success 0.96 Engineering 0.95 Finance 0.95 Marketing 0.95 People Ops 0.9 |

Regenerate with `python evals/run_eval.py`.

## Analysis of the two misses

`run_eval.py` overwrites this whole file on every run, so this section is
manually maintained — re-add it after regenerating.

**E06 (fail, but arguably a grading bug, not a model bug).** The question
asks for "the overall attendance rate, as days present over working days."
The model computed `14116 / 14720 = 0.958984...` and this run expressed it
as `95.8984` — correct arithmetic, just not rounded to the one decimal
place (`95.9`) the grader's exact-substring match expects. Same underlying
issue as previous runs (which have shown this as a bare fraction `0.959`
or as `95.8984`, depending on how the model chose to phrase the answer
that particular run) — the model's arithmetic is consistently right, only
the presentation varies. This is the eval set being too strict about
formatting, not the model getting the number wrong. Fix would be to either
reword the question to say "...rounded to one decimal place" or loosen the
grader to accept any value that rounds to 95.9 — not to keep re-rolling
the model until it happens to phrase it exactly as the grader expects.

**E12 (fail, and a real one — this is exactly the failure mode the eval
exists to catch).** Asked for "the average performance rating by
department," a column that does not exist anywhere in the data, the model
didn't decline. It silently redefined `days_present / working_days` as a
new column literally named `performance_rating` and reported it as if it
were the answer. Nothing in the output flags that this is a proxy metric
rather than the requested column. A user skimming the table would take it
as a genuine performance rating per department. This is the
confidently-wrong behavior `evals/` was built to surface, and it
consistently does its job across runs — the fix is a prompt-level
instruction to refuse when a question references a column that isn't in
any of the provided schemas, rather than substituting a plausible-looking
stand-in. That's a real follow-up, not something to paper over by tuning
the prompt until this one case happens to pass.

**Score is 10/12 as measured, but functionally more like 11/12 correct
arithmetic with 1 real safety miss** — E06's answer is right, just not in
the units/precision the grader expected.
