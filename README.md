# Additional experiments, Kaggle run (2 x T4), code commit 9956307

Output of `notebooks/rjudge_kaggle_extra.ipynb` as returned by the run. Nothing here is edited by hand.

| Folder | Experiment | Status |
|---|---|---|
| `guard_eval/` | Three LLM judges as guards on R-Judge, calibrated by the conformal rule (`scripts/rjudge_guard_eval.py`, 1000 random halves) | Result. Tables in `guard_eval.md`. |
| `guard_scores/` | The raw scores behind it: one row per record (`full`) or per decision step (`causal`) | Data. |
| `rjudge_steps/` | Proposition 1 on the 414 injection records with code-derived step labels, 100 splits | Result. Tables in `summary.md`. |
| `rjudge_recal_minilm/` | In-domain recalibration and per-category calibration, 100 random splits and 20 per held-out category | Result. Tables in `summary.md`. |
| `mas_pilot_v1/` | First live run of the multi-agent simulation | **Pilot with known defects. Not a result.** See below. |

## Notes on the guards

- `claude_*`: requests were sent to `claude-opus-5-5` with server-side refusal fallbacks. 354 of the 1,461 causal rows and
  209 of the 571 full rows were served by the fallback model `claude-opus-4-8` (column `served_by`). One row in each view
  was refused and has no score; the evaluator treats it as blocked. The guard evaluated is therefore "Opus 5.5 with
  fallback to Opus 4.8", not Opus 5.5 alone.
- `openai_*`: `gpt-6-astra`, 0-100 risk rating.
- `local_*`: `Qwen/Qwen2.5-7B-Instruct`, logit difference between the two answer letters.
- R-Judge is public, so any of these models may have seen its records in training. The false-block guarantee does not
  depend on that; the block rates might.

## Why `mas_pilot_v1` is not a result

It was generated with version 1 of the simulator, which had two defects that later versions remove.

1. **Claude sessions (23, all unusable).** The action format asked each agent for a free-text note before its action.
   163 of 168 agent steps were declined by the provider in the category `reasoning_extraction`; the five that were
   answered had been routed to a fallback model. The generator stopped itself after 23 sessions.
2. **OpenAI sessions (400, valid traces, little signal).** An agent that reached its step limit was cut off without a
   turn to report. The researcher usually read every document, hit the limit, and its findings were lost, so in most
   tasks the acting agent had nothing to send. Only 33 of 6,343 steps are violations, all of one kind.

`summary.md` and `results.json` are the experiment script's output on these sessions. With 33 violating steps (about 8
per test split) the block rates are too noisy to interpret, and the cross-model tables are meaningless because the
Claude sessions contain no agent behaviour. They are kept as a record of the run. The current simulator
(`SIM_VERSION = 2` in `conformalguard/mas/sim.py`) gives an agent a tool-call budget stated in its prompt and a final
turn to hand over, and its action format has no free-text field besides the message itself.
