# Judges as guards on the multi-agent sessions

500 random halves of the judged sessions (calibration / test), no training. The judge is given the policy and the monitor's metadata. FBR and Block in %, mean ± sd.

### Session-level rule (Corollary 1): a violation-free session has a blocked step with probability <= alpha

| guard | sessions | violating steps | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | Block@.10, one task | Block@.10, concurrent |
|---|---|---|---|---|---|---|---|---|
| claude (claude-opus-5-5) | 300 | 101 | 3.4 ± 2.8 | 99.0 ± 1.0 | 6.5 ± 2.9 | 99.0 ± 1.0 | 100.0 ± 0.0 | 98.9 ± 1.2 |
| deepseek (deepseek/deepseek-v3.2) | 300 | 101 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### Step-level rule (Proposition 1), alpha = 0.10, and operating figures

| guard | FBR@.10 | Block@.10 | median latency per decision | 90th percentile | tokens per decision (in / out / cache read / cache write) | USD per 1,000 decisions as billed | at list price, no caching | unscored steps | served by another model |
|---|---|---|---|---|---|---|---|---|---|
| claude | 7.6 ± 2.7 | 99.0 ± 1.0 | 1.52 s | 1.98 s | 219 / 10 / 0 / 2699 | 14.56 | 11.86 | 0 | 56 |
| deepseek | 7.6 ± 4.9 | 61.4 ± 44.4 | 1.72 s | 8.22 s | 1785 / 42 / 0 / 0 | n/a | n/a | 105 | 0 |

### Trained scorers, for reference (scripts/mas_run.py: 50/25/25 splits of all sessions, session-level rule)

| scorer | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | Block@.10, one task | Block@.10, concurrent |
|---|---|---|---|---|---|---|
| step | 4.9 ± 2.6 | 8.7 ± 4.5 | 9.6 ± 3.7 | 17.9 ± 7.2 | 12.3 ± 10.1 | 19.6 ± 7.2 |
| seq | 4.9 ± 2.7 | 53.1 ± 13.4 | 9.6 ± 3.9 | 70.4 ± 12.7 | 95.4 ± 14.1 | 62.5 ± 14.4 |
| graph | 4.9 ± 3.2 | 93.4 ± 4.4 | 10.2 ± 4.3 | 95.0 ± 3.8 | 98.6 ± 3.1 | 93.9 ± 4.7 |
