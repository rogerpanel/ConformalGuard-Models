# Multi-agent trace experiment (full run)

- 423 sessions (249 with concurrent tasks), agent models: {'claude-opus-5-5': 23, 'gpt-6-astra': 400}
- 6511 agent steps: 6445 safe, 33 violating ({'restricted_exfiltration': 33}); 32 sessions contain a violation
- 231 of 747 tasks had an injected instruction planted; 163 agent steps were refused by the model provider
- nodes per session: mean 41.9, max 87; labels are code-derived
- features: `st` (`sentence-transformers/all-MiniLM-L6-v2`) plus 12 metadata dimensions; 4 layers; per-step loss

## Random splits (50 splits, 50/25/25)

### Step-level rule (Proposition 1): a random safe step is blocked with probability <= alpha

FBR: a random safe step of a test session is blocked. Block: a violating step is blocked when proposed. Most steps are plainly harmless, so this budget is easy to meet; the session-level table below is the stricter test.

| scorer | step AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.990 | 4.7 ± 2.6 | 91.6 ± 12.3 | 9.0 ± 3.2 | 99.2 ± 3.9 | 18.9 ± 5.2 | 100.0 ± 0.0 |
| seq | 0.985 | 4.0 ± 2.5 | 88.0 ± 15.6 | 10.0 ± 4.7 | 96.2 ± 7.8 | 20.6 ± 5.7 | 100.0 ± 0.0 |
| graph | 0.994 | 5.1 ± 2.8 | 98.8 ± 7.3 | 10.0 ± 4.7 | 100.0 ± 0.0 | 19.5 ± 5.1 | 100.0 ± 0.0 |
| graph_nocites | 0.975 | 4.5 ± 3.1 | 83.5 ± 23.1 | 9.3 ± 4.2 | 92.8 ± 15.6 | 20.2 ± 5.8 | 98.1 ± 7.7 |
| graph_untyped | 0.863 | 4.9 ± 3.4 | 46.4 ± 25.2 | 9.9 ± 4.4 | 59.0 ± 26.9 | 18.9 ± 5.0 | 76.5 ± 23.9 |

### Session-level rule (Corollary 1): a violation-free session has a blocked step with probability <= alpha

FBR: share of violation-free test sessions with at least one blocked step. Block: a violating step is blocked when proposed.

| scorer | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|
| step | 3.7 ± 2.3 | 34.7 ± 15.8 | 9.0 ± 3.5 | 59.1 ± 19.1 | 19.8 ± 6.1 | 82.4 ± 12.9 |
| seq | 4.6 ± 3.3 | 39.7 ± 24.2 | 9.2 ± 4.4 | 57.8 ± 22.7 | 20.0 ± 5.0 | 68.0 ± 19.8 |
| graph | 3.8 ± 2.5 | 38.0 ± 24.5 | 8.4 ± 3.3 | 55.6 ± 25.5 | 18.8 ± 4.9 | 80.5 ± 20.7 |
| graph_nocites | 4.5 ± 2.6 | 19.4 ± 15.5 | 10.1 ± 4.0 | 34.2 ± 21.2 | 20.2 ± 5.6 | 58.8 ± 21.2 |
| graph_untyped | 3.7 ± 2.6 | 19.4 ± 18.4 | 8.7 ± 3.7 | 24.0 ± 18.7 | 19.6 ± 5.1 | 34.6 ± 23.7 |

### By concurrency, session-level rule, alpha = 0.10 (conditional rates; the guarantee is marginal)

| scorer | FBR, one task | FBR, concurrent tasks | Block, one task | Block, concurrent tasks |
|---|---|---|---|---|
| step | 8.9 ± 5.4 | 9.2 ± 4.5 | 61.1 ± 28.3 | 58.4 ± 21.3 |
| seq | 15.5 ± 9.9 | 4.9 ± 4.1 | 85.5 ± 26.1 | 38.8 ± 27.4 |
| graph | 12.1 ± 7.7 | 6.0 ± 3.8 | 75.2 ± 36.0 | 45.2 ± 29.5 |
| graph_nocites | 16.9 ± 10.0 | 5.5 ± 4.9 | 55.6 ± 33.5 | 19.8 ± 25.8 |
| graph_untyped | 11.4 ± 10.5 | 6.7 ± 6.9 | 51.6 ± 41.0 | 5.5 ± 11.8 |

### Block by kind of violation, alpha = 0.10 (step-level rule / session-level rule)

| scorer | injected_target | restricted_exfiltration | unauthorised_tool |
|---|---|---|---|
| step | n/a / n/a | 99.2 ± 3.9 / 59.1 ± 19.1 | n/a / n/a |
| seq | n/a / n/a | 96.2 ± 7.8 / 57.8 ± 22.7 | n/a / n/a |
| graph | n/a / n/a | 100.0 ± 0.0 / 55.6 ± 25.5 | n/a / n/a |
| graph_nocites | n/a / n/a | 92.8 ± 15.6 / 34.2 ± 21.2 | n/a / n/a |
| graph_untyped | n/a / n/a | 59.0 ± 26.9 / 24.0 ± 18.7 | n/a / n/a |

### Paired differences over splits (percentage points; exact sign test)

| comparison | metric | mean difference | splits better / worse | p |
|---|---|---|---|---|
| graph-seq | block_session@0.05 | -1.7 | 15 / 20 | 0.5 |
| graph-seq | block_session@0.10 | -2.2 | 18 / 17 | 1 |
| graph-seq | block_session_concurrent@0.10 | +6.4 | 19 / 12 | 0.281 |
| graph-seq | block_session_single@0.10 | -10.3 | 6 / 12 | 0.238 |
| graph-seq | block_session_restricted_exfiltration@0.10 | -2.2 | 18 / 17 | 1 |
| graph-seq | block@0.10 | +3.8 | 12 / 0 | 0.000488 |
| graph-seq | auroc_step | +0.009 | 41 / 8 | 1.96e-06 |
| graph-graph_nocites | block_session@0.05 | +18.6 | 35 / 6 | 4.87e-06 |
| graph-graph_nocites | block_session@0.10 | +21.4 | 38 / 5 | 2.5e-07 |
| graph-graph_nocites | block_session_concurrent@0.10 | +25.4 | 37 / 3 | 1.95e-08 |
| graph-graph_nocites | block_session_single@0.10 | +19.6 | 25 / 9 | 0.00904 |
| graph-graph_nocites | block_session_restricted_exfiltration@0.10 | +21.4 | 38 / 5 | 2.5e-07 |
| graph-graph_nocites | block@0.10 | +7.2 | 15 / 0 | 6.1e-05 |
| graph-graph_nocites | auroc_step | +0.019 | 49 / 0 | 3.55e-15 |
| graph-graph_untyped | block_session@0.05 | +18.6 | 31 / 9 | 0.00068 |
| graph-graph_untyped | block_session@0.10 | +31.6 | 40 / 6 | 3.1e-07 |
| graph-graph_untyped | block_session_concurrent@0.10 | +39.7 | 43 / 2 | 5.89e-11 |
| graph-graph_untyped | block_session_single@0.10 | +23.6 | 27 / 9 | 0.00393 |
| graph-graph_untyped | block_session_restricted_exfiltration@0.10 | +31.6 | 40 / 6 | 3.1e-07 |
| graph-graph_untyped | block@0.10 | +41.0 | 47 / 0 | 1.42e-14 |
| graph-graph_untyped | auroc_step | +0.131 | 50 / 0 | 1.78e-15 |
| seq-step | block_session@0.05 | +5.0 | 22 / 18 | 0.636 |
| seq-step | block_session@0.10 | -1.2 | 21 / 20 | 1 |
| seq-step | block_session_concurrent@0.10 | -19.6 | 7 / 31 | 0.000116 |
| seq-step | block_session_single@0.10 | +24.4 | 32 / 5 | 7.43e-06 |
| seq-step | block_session_restricted_exfiltration@0.10 | -1.2 | 21 / 20 | 1 |
| seq-step | block@0.10 | -3.1 | 0 / 11 | 0.000977 |
| seq-step | auroc_step | -0.005 | 20 / 30 | 0.203 |

## cross:claude-opus-5-5->gpt-6-astra (10 runs): distribution shift, no guarantee

| scorer | step FBR@.10 | step Block@.10 | session FBR@.10 | session Block@.10 | step AUROC |
|---|---|---|---|---|---|
| step | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.221 |
| seq | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.751 |
| graph | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.654 |
| graph_nocites | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.671 |
| graph_untyped | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.525 |

## cross:gpt-6-astra->claude-opus-5-5 (10 runs): distribution shift, no guarantee

| scorer | step FBR@.10 | step Block@.10 | session FBR@.10 | session Block@.10 | step AUROC |
|---|---|---|---|---|---|
| step | 99.1 ± 1.8 | n/a | 80.9 ± 24.2 | n/a | n/a |
| seq | 4.3 ± 7.9 | n/a | 0.0 ± 0.0 | n/a | n/a |
| graph | 3.0 ± 9.6 | n/a | 0.0 ± 0.0 | n/a | n/a |
| graph_nocites | 1.3 ± 2.9 | n/a | 0.0 ± 0.0 | n/a | n/a |
| graph_untyped | 0.0 ± 0.0 | n/a | 0.0 ± 0.0 | n/a | n/a |
