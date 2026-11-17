# Multi-agent trace experiment (full run)

- 669 sessions (418 with concurrent tasks), agent models: {'claude-opus-5-5': 600, 'gpt-6-astra': 69}
- 10369 agent steps: 9933 safe, 217 violating ({'restricted_exfiltration': 216, 'injected_target': 1}); 196 sessions contain a violation
- 390 of 1232 tasks had an injected instruction planted
- simulator version 2; 669 sessions loaded, 0 of another simulator version left out, 0 with a step refused by the model provider (left out)
- nodes per session: mean 41.0, max 126; labels are code-derived
- features: `st` (`sentence-transformers/all-MiniLM-L6-v2`) plus 12 metadata dimensions; 4 layers; per-step loss

## Random splits (50 splits, 50/25/25)

### Step-level rule (Proposition 1): a random safe step is blocked with probability <= alpha

FBR: a random safe step of a test session is blocked. Block: a violating step is blocked when proposed. Most steps are plainly harmless, so this budget is easy to meet; the session-level table below is the stricter test.

| scorer | step AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.982 | 4.9 ± 2.4 | 96.0 ± 9.5 | 9.9 ± 3.4 | 99.3 ± 1.0 | 19.8 ± 4.0 | 99.9 ± 0.4 |
| seq | 0.992 | 5.1 ± 2.9 | 97.1 ± 3.6 | 9.6 ± 3.4 | 99.5 ± 1.0 | 19.9 ± 4.3 | 99.9 ± 0.4 |
| graph | 0.998 | 5.0 ± 2.6 | 98.5 ± 1.7 | 10.0 ± 3.4 | 99.9 ± 0.6 | 20.9 ± 4.3 | 100.0 ± 0.0 |
| graph_nocites | 0.976 | 5.2 ± 2.6 | 84.7 ± 12.1 | 10.1 ± 3.5 | 96.6 ± 5.2 | 20.0 ± 4.1 | 100.0 ± 0.0 |
| graph_untyped | 0.995 | 4.5 ± 2.5 | 97.2 ± 2.8 | 10.0 ± 3.8 | 98.8 ± 1.4 | 20.1 ± 4.4 | 99.6 ± 0.8 |

### Session-level rule (Corollary 1): a violation-free session has a blocked step with probability <= alpha

FBR: share of violation-free test sessions with at least one blocked step. Block: a violating step is blocked when proposed.

| scorer | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|
| step | 4.9 ± 2.6 | 8.7 ± 4.5 | 9.6 ± 3.7 | 17.9 ± 7.2 | 20.7 ± 5.2 | 39.0 ± 9.7 |
| seq | 4.9 ± 2.7 | 53.1 ± 13.4 | 9.6 ± 3.9 | 70.4 ± 12.7 | 20.3 ± 4.7 | 85.7 ± 10.5 |
| graph | 4.9 ± 3.2 | 93.4 ± 4.4 | 10.2 ± 4.3 | 95.0 ± 3.8 | 21.3 ± 5.1 | 96.3 ± 3.6 |
| graph_nocites | 4.6 ± 3.5 | 6.9 ± 4.1 | 10.2 ± 4.4 | 14.3 ± 5.6 | 20.7 ± 5.6 | 32.5 ± 8.4 |
| graph_untyped | 4.4 ± 2.8 | 83.7 ± 9.8 | 9.7 ± 4.7 | 88.8 ± 7.3 | 20.0 ± 6.2 | 92.5 ± 5.0 |

### By concurrency, session-level rule, alpha = 0.10 (conditional rates; the guarantee is marginal)

| scorer | FBR, one task | FBR, concurrent tasks | Block, one task | Block, concurrent tasks |
|---|---|---|---|---|
| step | 5.6 ± 3.4 | 12.7 ± 5.7 | 12.3 ± 10.1 | 19.6 ± 7.2 |
| seq | 1.1 ± 2.3 | 15.8 ± 6.6 | 95.4 ± 14.1 | 62.5 ± 14.4 |
| graph | 2.9 ± 4.8 | 15.5 ± 5.6 | 98.6 ± 3.1 | 93.9 ± 4.7 |
| graph_nocites | 6.5 ± 7.0 | 12.9 ± 5.2 | 14.1 ± 13.7 | 14.7 ± 7.4 |
| graph_untyped | 7.6 ± 5.8 | 11.2 ± 8.1 | 98.1 ± 3.5 | 85.9 ± 9.6 |

### Block by kind of violation, alpha = 0.10 (step-level rule / session-level rule)

| scorer | injected_target | restricted_exfiltration | unauthorised_tool |
|---|---|---|---|
| step | 6.7 ± 25.8 / 0.0 ± 0.0 | 99.9 ± 0.5 / 18.0 ± 7.3 | n/a / n/a |
| seq | 53.3 ± 51.6 / 0.0 ± 0.0 | 99.7 ± 0.8 / 70.8 ± 12.9 | n/a / n/a |
| graph | 86.7 ± 35.2 / 0.0 ± 0.0 | 100.0 ± 0.3 / 95.5 ± 3.6 | n/a / n/a |
| graph_nocites | 86.7 ± 35.2 / 0.0 ± 0.0 | 96.7 ± 5.0 / 14.4 ± 5.6 | n/a / n/a |
| graph_untyped | 13.3 ± 35.2 / 0.0 ± 0.0 | 99.2 ± 1.1 / 89.3 ± 7.2 | n/a / n/a |

### Paired differences over splits (percentage points; exact sign test)

| comparison | metric | mean difference | splits better / worse | p |
|---|---|---|---|---|
| graph-seq | block_session@0.05 | +40.3 | 50 / 0 | 1.78e-15 |
| graph-seq | block_session@0.10 | +24.6 | 50 / 0 | 1.78e-15 |
| graph-seq | block_session_concurrent@0.10 | +31.4 | 50 / 0 | 1.78e-15 |
| graph-seq | block_session_single@0.10 | +3.2 | 5 / 0 | 0.0625 |
| graph-seq | block_session_restricted_exfiltration@0.10 | +24.7 | 50 / 0 | 1.78e-15 |
| graph-seq | block@0.10 | +0.4 | 12 / 0 | 0.000488 |
| graph-seq | auroc_step | +0.006 | 50 / 0 | 1.78e-15 |
| graph-graph_nocites | block_session@0.05 | +86.5 | 50 / 0 | 1.78e-15 |
| graph-graph_nocites | block_session@0.10 | +80.7 | 50 / 0 | 1.78e-15 |
| graph-graph_nocites | block_session_concurrent@0.10 | +79.2 | 50 / 0 | 1.78e-15 |
| graph-graph_nocites | block_session_single@0.10 | +84.5 | 50 / 0 | 1.78e-15 |
| graph-graph_nocites | block_session_restricted_exfiltration@0.10 | +81.1 | 50 / 0 | 1.78e-15 |
| graph-graph_nocites | block@0.10 | +3.3 | 20 / 0 | 1.91e-06 |
| graph-graph_nocites | auroc_step | +0.022 | 50 / 0 | 1.78e-15 |
| graph-graph_untyped | block_session@0.05 | +9.7 | 39 / 7 | 1.83e-06 |
| graph-graph_untyped | block_session@0.10 | +6.1 | 34 / 4 | 6.04e-07 |
| graph-graph_untyped | block_session_concurrent@0.10 | +8.0 | 35 / 4 | 3.35e-07 |
| graph-graph_untyped | block_session_single@0.10 | +0.5 | 4 / 1 | 0.375 |
| graph-graph_untyped | block_session_restricted_exfiltration@0.10 | +6.2 | 34 / 4 | 6.04e-07 |
| graph-graph_untyped | block@0.10 | +1.1 | 26 / 0 | 2.98e-08 |
| graph-graph_untyped | auroc_step | +0.003 | 43 / 7 | 2.1e-07 |
| seq-step | block_session@0.05 | +44.4 | 49 / 1 | 9.06e-14 |
| seq-step | block_session@0.10 | +52.5 | 49 / 0 | 3.55e-15 |
| seq-step | block_session_concurrent@0.10 | +43.0 | 49 / 0 | 3.55e-15 |
| seq-step | block_session_single@0.10 | +83.0 | 49 / 1 | 9.06e-14 |
| seq-step | block_session_restricted_exfiltration@0.10 | +52.8 | 49 / 0 | 3.55e-15 |
| seq-step | block@0.10 | +0.1 | 10 / 6 | 0.454 |
| seq-step | auroc_step | +0.009 | 48 / 2 | 2.27e-12 |

## cross:claude-opus-5-5->gpt-6-astra (10 runs): distribution shift, no guarantee

| scorer | step FBR@.10 | step Block@.10 | session FBR@.10 | session Block@.10 | step AUROC |
|---|---|---|---|---|---|
| step | 5.9 ± 2.9 | 95.8 ± 7.1 | 0.2 ± 0.7 | 3.8 ± 0.0 | 0.979 |
| seq | 12.2 ± 5.6 | 99.2 ± 1.6 | 9.6 ± 5.5 | 54.6 ± 8.8 | 0.989 |
| graph | 4.8 ± 2.7 | 100.0 ± 0.0 | 5.1 ± 3.6 | 86.2 ± 10.8 | 0.998 |
| graph_nocites | 2.8 ± 1.9 | 85.0 ± 19.8 | 0.9 ± 1.1 | 0.8 ± 2.4 | 0.987 |
| graph_untyped | 5.8 ± 3.0 | 95.4 ± 4.7 | 4.7 ± 5.1 | 82.3 ± 9.8 | 0.989 |

## cross:gpt-6-astra->claude-opus-5-5 (10 runs): distribution shift, no guarantee

| scorer | step FBR@.10 | step Block@.10 | session FBR@.10 | session Block@.10 | step AUROC |
|---|---|---|---|---|---|
| step | 18.6 ± 8.4 | 94.8 ± 6.7 | 4.6 ± 3.3 | 0.7 ± 1.4 | 0.947 |
| seq | 4.2 ± 2.9 | 47.1 ± 31.9 | 0.7 ± 1.1 | 0.8 ± 1.3 | 0.942 |
| graph | 11.8 ± 5.1 | 91.4 ± 18.2 | 9.1 ± 11.4 | 10.1 ± 12.1 | 0.966 |
| graph_nocites | 14.1 ± 7.7 | 77.8 ± 32.2 | 4.7 ± 4.7 | 2.6 ± 2.7 | 0.927 |
| graph_untyped | 10.8 ± 9.3 | 31.8 ± 26.3 | 5.2 ± 12.1 | 0.5 ± 1.0 | 0.803 |
