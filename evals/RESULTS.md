# Eval results

- Model: `llama-3.3-70b-versatile` (backend: `groq`)
- Dataset: `sample_data/` — employees, attendance, exits (3 files)
- Runs: 2
- **Score: 22/24 (92%)**

| ID | Capability | Result | Notes |
|---|---|---|---|
| E01 | simple aggregate | pass |  |
| E02 | simple aggregate | pass |  |
| E03 | group-by | pass |  |
| E04 | group-by + argmax | pass |  |
| E05 | filter + count | pass |  |
| E06 | single-file aggregate over many rows | **fail** | missing expected value(s) ['95.9']; got: '0.959' |
| E07 | cross-file join (2 files) | pass |  |
| E08 | cross-file join (2 files) + comparison | pass |  |
| E09 | cross-file join (3 files) | pass |  |
| E10 | date-format trap + join | pass |  |
| E11 | chart / visual insight | pass |  |
| E12 | unanswerable — should decline, not hallucinate | pass |  |

Regenerate with `python evals/run_eval.py`.

## Analysis of the remaining miss

`run_eval.py` overwrites this whole file on every run, so this section is
manually maintained — re-add it after regenerating.

**E06 (fail, but arguably a grading bug, not a model bug).** The question
asks for "the overall attendance rate, as days present over working days."
The model computes the arithmetic correctly (`14116 / 14720 = 0.958984...`)
but expresses it as `0.959` or `95.8984` depending on the run, rather than
the `95.9` the grader's exact-substring match expects. This is the eval
being too strict about formatting/units, not the model getting the number
wrong — the fix would be to reword the question ("...rounded to one decimal
place, as a percentage") or loosen the grader, not to keep re-rolling the
model until it happens to phrase it exactly right.

**E12 — previously a real hallucination, now fixed at the prompt level.**
Earlier runs showed the model silently redefining `days_present /
working_days` as a new column named `performance_rating` and reporting it
as if it were the requested (nonexistent) column — confidently wrong,
undetectable by a user skimming the table. Fixed by adding an explicit rule
to `build_prompt()` in `core.py`: check every concept in the question
against the real schema before computing anything, and if a column doesn't
exist, say so instead of substituting a plausible-looking stand-in. The
rule includes one domain-neutral example (unrelated to this app's actual
HR dataset, to avoid tuning the prompt to this specific eval case) so the
model generalizes the "don't repurpose a numeric column as a proxy" pattern
rather than memorizing this one question. Verified live: reruns of E12
across two full eval passes both decline correctly, and E01–E11 are
unaffected.

**Score is 22/24 as measured (2 runs), but functionally 23/24 correct
arithmetic** — E06's answer is right, just not in the exact precision the
grader's substring match expects.
