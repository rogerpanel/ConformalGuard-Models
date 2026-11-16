# Step-level guarantee on R-Judge injection records (full run)

- 414 of 414 injection records labelled by code (0 left out: label and tool evidence disagree); 200 have a violating step; 628 safe steps
- features: `st` (`sentence-transformers/all-MiniLM-L6-v2`); 100 random splits 50/25/25; per-step loss
- FBR: a random safe step of a test trace is blocked. Block at: the violating step is blocked. Block by: some step up to and including it is blocked.

### One calibration step per trace (Proposition 1: FBR <= alpha)

| scorer | step AUROC | FBR@.05 | Block at@.05 | FBR@.10 | Block at@.10 | Block by@.10 | FBR@.20 | Block at@.20 |
|---|---|---|---|---|---|---|---|---|
| step | 0.995 | 4.5 ± 2.9 | 97.5 ± 5.5 | 9.8 ± 4.3 | 98.8 ± 3.3 | 98.8 ± 3.3 | 18.0 ± 5.1 | 99.8 ± 1.0 |
| seq | 1.000 | 4.8 ± 3.2 | 100.0 ± 0.0 | 9.8 ± 4.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 19.9 ± 4.8 | 100.0 ± 0.0 |
| graph | 1.000 | 4.8 ± 3.3 | 100.0 ± 0.0 | 9.4 ± 4.7 | 100.0 ± 0.0 | 100.0 ± 0.0 | 19.4 ± 5.6 | 100.0 ± 0.0 |

### All calibration steps pooled (no exact guarantee), alpha = 0.10

| scorer | FBR, random safe step | FBR, averaged over safe steps | Block at |
|---|---|---|---|
| step | 8.5 ± 3.9 | 8.2 ± 3.4 | 98.8 ± 3.1 |
| seq | 7.7 ± 2.9 | 7.4 ± 2.6 | 100.0 ± 0.0 |
| graph | 7.7 ± 3.8 | 7.6 ± 3.6 | 100.0 ± 0.0 |
