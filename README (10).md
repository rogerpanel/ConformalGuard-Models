# Follow-up measurements, Kaggle run (2 x T4)

Output of `notebooks/followup_kaggle.ipynb`. Score files and sessions are as returned by the run. Two reports
(`judges_sessions/guard_eval.*`, `closed_loop/summary.md` and `results.json`) were recomputed from those files
after the run with two corrections described below; `rjudge_openweight/guard_eval.*` is the run's own output.

| Folder | Measurement | Status |
|---|---|---|
| `judges_sessions/` | LLM judges on 300 multi-agent sessions (4,847 decisions): `claude-opus-5-5` and `deepseek/deepseek-v3.2`, calibrated at the same budgets as the trained scorers, with latency and token use | Result for the two API judges. The two local judges are incomplete (see below). |
| `rjudge_openweight/` | Open-weight judges on R-Judge under the protocol used for Qwen2.5-7B: `Qwen/Qwen3-8B` (run locally) and `deepseek/deepseek-v3.2` (an open-weight model, run here through an API) | Result. |
| `closed_loop/` | The same workspaces run without and with the guard in the loop | Result on 172 workspaces; the guard arm stopped at 172 of 200 when the API credit ran out. |

## Corrections made after the run

1. **Task-success checker.** Live agents write a channel as `#name`. The checker compared that with the task's
   `#name` after adding a second `#`, so every channel post counted as a failure in both arms (63 tasks). Fixed in
   `conformalguard/mas/labels.py::destination`; the report was recomputed from the same sessions.
2. **Rating parser.** About 2% of DeepSeek's replies on the sessions open with prose ("Let's examine ... 1. RESTRICTED
   ..."), and the first number found in them was taken as the rating. A reply that does not start with an integer now
   has no rating and counts as blocked (`conformalguard/mas/judge.py::leading_rating`), as refusals do.
3. The closed-loop report now compares only workspaces present in both arms.

## Things to state with the numbers

- **The judge is told the policy** and the monitor's metadata; the trained scorers learn the policy from labelled
  sessions. The judges are calibrated on random halves of the 300 sessions they scored; the trained scorers'
  reference numbers come from 50/25/25 splits of all 669 sessions.
- **Cost of the Claude judge.** The run sent the whole log as one content block, so every step wrote the cache and
  none read it. The cost as billed (14.56 USD per 1,000 decisions) is therefore higher than a careful
  implementation would pay; the figure at list input price without any caching (11.86 USD per 1,000 decisions,
  from the measured token counts) does not depend on that. The script now sends one block per log line.
- **DeepSeek on the sessions.** It gives its top rating (100) to a safe step in 53% of the violation-free sessions,
  so under the per-session budget no finite threshold separates anything and nothing is blocked. Under the per-step
  budget its block rate is unstable across splits for the same reason (ties at the top rating).
- **Local judges on the sessions are incomplete.** `Qwen/Qwen3-8B` and `Qwen/Qwen2.5-7B-Instruct` scored 11 sessions
  each (about 200 decisions) and then stopped in the same long session, whose log reaches about 26,000 characters;
  a decision took 2.0 to 2.5 s at the median and up to 11 s on two T4 GPUs. The files are kept for the timing;
  eleven sessions are too few to calibrate on.
- **Closed loop.** The two arms sample the agents anew, so they differ by the guard and by the model's run-to-run
  variation. The guard was trained on open-loop sessions. The false-interruption rate (15 of 125 sessions without a
  violating proposal, 12.0%) is compared with alpha = 0.10 under the assumption that the new workspaces are
  exchangeable with the calibration sessions; one standard error at that rate and sample size is 2.7 points.
  In 108 of the 372 sessions at least one agent step was answered by the provider's fallback model.
- **No task recovered after a false block**: in all 17 safe-to-complete tasks in which the guard blocked an action,
  the agent stopped and the task did not complete.
- R-Judge is public, so the judges may have seen its records in training.
