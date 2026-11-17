# R-Judge experiment (full run)

- records: 571 (301 unsafe, 270 safe), 151 distinct user prompts, 1461 decision steps
- features: `st` (`sentence-transformers/all-MiniLM-L6-v2`), dimension 384
- scorer parameters: {'step': 98561, 'seq': 866817, 'graph': 1040034, 'graph_untyped': 298119}
- record score = maximum step logit; threshold = conformal quantile of the safe calibration records; split 50/25/25

### Random splits (exchangeable: the FBR guarantee applies)

Mean ± sd over 100 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.958 ± 0.019 | 3.5 ± 3.5 | 65.2 ± 13.6 | 7.3 ± 4.4 | 77.7 ± 9.7 | 17.8 ± 5.8 | 93.7 ± 5.5 |
| seq | 0.962 ± 0.021 | 4.5 ± 3.3 | 69.4 ± 16.7 | 8.4 ± 4.1 | 82.3 ± 9.8 | 17.6 ± 6.3 | 97.3 ± 3.5 |
| graph | 0.956 ± 0.020 | 3.8 ± 2.8 | 63.1 ± 19.1 | 8.0 ± 4.2 | 77.8 ± 13.8 | 17.9 ± 6.1 | 96.6 ± 4.6 |
| graph_untyped | 0.905 ± 0.030 | 3.7 ± 3.0 | 54.3 ± 11.6 | 8.2 ± 4.5 | 63.2 ± 9.8 | 18.3 ± 6.1 | 81.0 ± 8.8 |
| logreg_pooled | 0.912 ± 0.026 | 4.3 ± 3.7 | 31.7 ± 19.8 | 7.9 ± 4.3 | 60.3 ± 17.5 | 17.6 ± 5.0 | 87.2 ± 6.0 |
| rule_last_tool | 0.902 ± 0.016 | 1.4 ± 2.4 | 26.1 ± 39.2 | 3.8 ± 2.1 | 78.3 ± 21.8 | 3.9 ± 2.0 | 84.3 ± 2.9 |

### Prompt-grouped splits (no user prompt shared between train and test)

Mean ± sd over 100 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.922 ± 0.055 | 9.0 ± 13.0 | 67.4 ± 16.4 | 16.0 ± 19.0 | 75.9 ± 16.1 | 29.0 ± 26.6 | 86.3 ± 14.0 |
| seq | 0.942 ± 0.045 | 7.9 ± 11.8 | 67.0 ± 20.7 | 16.6 ± 18.9 | 81.0 ± 16.4 | 27.8 ± 24.7 | 92.0 ± 10.3 |
| graph | 0.929 ± 0.054 | 7.5 ± 12.5 | 57.5 ± 26.4 | 16.7 ± 22.0 | 75.6 ± 20.9 | 27.9 ± 25.3 | 90.0 ± 13.9 |
| graph_untyped | 0.879 ± 0.071 | 8.6 ± 10.9 | 62.0 ± 16.7 | 15.6 ± 17.6 | 72.5 ± 16.7 | 27.8 ± 24.0 | 82.2 ± 15.1 |
| logreg_pooled | 0.860 ± 0.071 | 8.1 ± 12.2 | 34.1 ± 24.6 | 16.4 ± 19.3 | 56.3 ± 25.3 | 28.0 ± 25.2 | 76.5 ± 18.3 |
| rule_last_tool | 0.891 ± 0.047 | 2.4 ± 3.7 | 33.5 ± 40.7 | 4.6 ± 4.0 | 67.2 ± 33.5 | 5.2 ± 3.6 | 83.5 ± 7.7 |

Calibration unit on the grouped splits: FBR (%) at alpha = 0.10 with the threshold set
from all safe calibration records (pooled) and from one random safe record per
calibration prompt group, evaluated on one random record per test group.

| scorer | FBR pooled | FBR one per group | Block one per group |
|---|---|---|---|
| step | 16.0 ± 19.0 | 6.7 ± 9.3 | 25.5 ± 15.6 |
| seq | 16.6 ± 18.9 | 7.3 ± 8.6 | 27.5 ± 18.2 |
| graph | 16.7 ± 22.0 | 6.1 ± 7.6 | 21.8 ± 14.1 |
| graph_untyped | 15.6 ± 17.6 | 7.6 ± 8.3 | 26.8 ± 16.2 |
| logreg_pooled | 16.4 ± 19.3 | 6.2 ± 8.1 | 18.7 ± 14.6 |
| rule_last_tool | 4.6 ± 4.0 | 1.9 ± 6.3 | 6.1 ± 18.7 |

### Leave-one-category-out, test on Application (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.878 ± 0.023 | 20.4 ± 7.2 | 72.5 ± 14.0 | 29.0 ± 10.9 | 84.6 ± 9.1 | 74.7 ± 19.4 | 98.5 ± 3.0 |
| seq | 0.882 ± 0.125 | 11.4 ± 19.7 | 55.4 ± 35.6 | 22.2 ± 21.1 | 79.0 ± 28.9 | 73.5 ± 31.1 | 95.4 ± 11.3 |
| graph | 0.916 ± 0.062 | 6.6 ± 6.9 | 51.0 ± 40.9 | 11.3 ± 8.6 | 79.2 ± 31.5 | 74.4 ± 28.7 | 98.1 ± 2.3 |
| graph_untyped | 0.827 ± 0.004 | 6.5 ± 5.6 | 54.8 ± 10.9 | 11.8 ± 7.5 | 64.6 ± 5.8 | 33.6 ± 20.2 | 79.4 ± 9.7 |
| logreg_pooled | 0.791 ± 0.017 | 26.8 ± 13.2 | 69.8 ± 10.5 | 50.3 ± 10.5 | 84.3 ± 3.3 | 81.3 ± 9.1 | 96.1 ± 2.4 |
| rule_last_tool | 0.922 ± 0.000 | 3.6 ± 3.8 | 45.8 ± 48.3 | 7.2 ± 0.0 | 91.6 ± 0.0 | 7.2 ± 0.0 | 91.6 ± 0.0 |

### Leave-one-category-out, test on Finance (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.996 ± 0.001 | 0.6 ± 0.6 | 71.0 ± 15.1 | 1.1 ± 0.5 | 85.4 ± 12.6 | 13.3 ± 29.6 | 95.9 ± 5.6 |
| seq | 0.996 ± 0.003 | 0.5 ± 0.8 | 58.5 ± 31.4 | 0.9 ± 1.1 | 71.5 ± 24.3 | 11.4 ± 30.3 | 92.6 ± 9.3 |
| graph | 0.996 ± 0.002 | 0.6 ± 0.8 | 74.1 ± 12.7 | 1.0 ± 1.0 | 86.4 ± 8.9 | 13.9 ± 29.8 | 98.2 ± 2.4 |
| graph_untyped | 0.977 ± 0.020 | 0.2 ± 0.5 | 69.0 ± 20.6 | 0.5 ± 0.6 | 83.1 ± 7.3 | 2.1 ± 3.4 | 93.6 ± 5.2 |
| logreg_pooled | 0.988 ± 0.003 | 1.0 ± 0.8 | 39.0 ± 12.5 | 1.5 ± 0.6 | 76.7 ± 16.0 | 4.5 ± 2.8 | 98.2 ± 2.7 |
| rule_last_tool | 0.866 ± 0.000 | 0.1 ± 0.4 | 7.4 ± 23.5 | 1.1 ± 0.0 | 74.4 ± 0.0 | 1.1 ± 0.0 | 74.4 ± 0.0 |

### Leave-one-category-out, test on IoT (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.623 ± 0.021 | 38.2 ± 21.8 | 51.1 ± 20.5 | 61.8 ± 17.0 | 77.4 ± 20.8 | 85.5 ± 6.4 | 97.9 ± 5.1 |
| seq | 0.602 ± 0.104 | 20.0 ± 22.2 | 28.9 ± 22.6 | 57.3 ± 16.6 | 71.6 ± 23.4 | 96.4 ± 4.7 | 100.0 ± 0.0 |
| graph | 0.668 ± 0.044 | 32.7 ± 17.8 | 53.2 ± 20.2 | 61.8 ± 14.7 | 83.2 ± 10.2 | 94.5 ± 6.4 | 100.0 ± 0.0 |
| graph_untyped | 0.639 ± 0.077 | 39.1 ± 18.7 | 57.9 ± 21.2 | 59.1 ± 17.3 | 78.4 ± 9.4 | 87.3 ± 9.8 | 93.2 ± 5.6 |
| logreg_pooled | 0.569 ± 0.025 | 50.9 ± 8.8 | 62.6 ± 16.9 | 58.2 ± 4.7 | 81.6 ± 5.1 | 75.5 ± 6.1 | 91.6 ± 4.4 |
| rule_last_tool | 0.658 ± 0.000 | 0.0 ± 0.0 | 15.8 ± 16.6 | 0.0 ± 0.0 | 31.6 ± 0.0 | 0.0 ± 0.0 | 31.6 ± 0.0 |

### Leave-one-category-out, test on Program (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.967 ± 0.006 | 2.0 ± 2.3 | 58.2 ± 15.1 | 4.5 ± 1.1 | 77.9 ± 7.1 | 13.7 ± 3.7 | 92.8 ± 6.3 |
| seq | 0.896 ± 0.041 | 0.7 ± 1.6 | 44.1 ± 6.0 | 7.0 ± 3.0 | 54.1 ± 4.1 | 24.2 ± 25.2 | 86.6 ± 12.1 |
| graph | 0.877 ± 0.084 | 1.8 ± 1.7 | 43.5 ± 7.2 | 7.7 ± 2.6 | 52.9 ± 4.5 | 32.2 ± 27.6 | 89.6 ± 13.8 |
| graph_untyped | 0.839 ± 0.091 | 3.2 ± 2.3 | 45.1 ± 4.3 | 8.7 ± 3.0 | 50.3 ± 10.9 | 19.0 ± 9.4 | 71.9 ± 22.5 |
| logreg_pooled | 0.891 ± 0.017 | 1.7 ± 3.1 | 26.8 ± 18.4 | 7.2 ± 2.1 | 64.0 ± 9.0 | 16.2 ± 6.7 | 87.6 ± 2.3 |
| rule_last_tool | 0.925 ± 0.000 | 0.3 ± 0.7 | 17.4 ± 36.6 | 1.7 ± 0.0 | 86.8 ± 0.0 | 1.7 ± 0.0 | 86.8 ± 0.0 |

### Leave-one-category-out, test on Web (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.691 ± 0.026 | 4.0 ± 5.6 | 31.0 ± 8.4 | 28.7 ± 18.1 | 49.0 ± 14.7 | 66.7 ± 23.9 | 86.0 ± 14.3 |
| seq | 0.782 ± 0.074 | 10.0 ± 8.5 | 37.5 ± 18.9 | 34.0 ± 24.0 | 67.5 ± 17.7 | 84.0 ± 20.9 | 98.5 ± 3.4 |
| graph | 0.722 ± 0.036 | 10.0 ± 7.9 | 28.0 ± 16.5 | 24.7 ± 8.9 | 54.0 ± 7.0 | 92.7 ± 12.7 | 97.5 ± 3.5 |
| graph_untyped | 0.675 ± 0.043 | 10.0 ± 6.5 | 31.0 ± 9.1 | 19.3 ± 11.1 | 40.5 ± 12.1 | 52.0 ± 23.3 | 71.0 ± 18.2 |
| logreg_pooled | 0.527 ± 0.018 | 6.0 ± 9.7 | 28.0 ± 8.6 | 32.7 ± 20.0 | 38.5 ± 8.5 | 68.0 ± 19.1 | 65.5 ± 14.6 |
| rule_last_tool | 0.825 ± 0.000 | 12.0 ± 10.3 | 51.0 ± 43.9 | 20.0 ± 0.0 | 85.0 ± 0.0 | 20.0 ± 0.0 | 85.0 ± 0.0 |

### Paired comparisons (a minus b, per split; exact sign test)

| protocol | a - b | metric | mean diff | a better / b better / ties | p |
|---|---|---|---|---|---|
| random | graph-seq | auroc | -0.01 | 38 / 62 / 0 | 0.021 |
| random | graph-seq | block@0.10 | -4.52 pp | 31 / 63 / 6 | 0.00126 |
| random | graph-step | auroc | -0.00 | 49 / 50 / 1 | 1 |
| random | graph-step | block@0.10 | +0.07 pp | 49 / 49 / 2 | 1 |
| random | seq-step | auroc | +0.00 | 58 / 41 / 1 | 0.107 |
| random | seq-step | block@0.10 | +4.59 pp | 62 / 32 / 6 | 0.00259 |
| random | graph-graph_untyped | auroc | +0.05 | 94 / 6 / 0 | 2.01e-21 |
| random | graph-graph_untyped | block@0.10 | +14.61 pp | 83 / 14 / 3 | 4.22e-13 |
| random | graph-logreg_pooled | auroc | +0.04 | 95 / 5 / 0 | 1.25e-22 |
| random | graph-logreg_pooled | block@0.10 | +17.49 pp | 87 / 12 / 1 | 3.37e-15 |
| grouped | graph-seq | auroc | -0.01 | 33 / 66 / 1 | 0.00119 |
| grouped | graph-seq | block@0.10 | -5.41 pp | 31 / 54 / 15 | 0.0165 |
| grouped | graph-step | auroc | +0.01 | 60 / 40 / 0 | 0.0569 |
| grouped | graph-step | block@0.10 | -0.25 pp | 51 / 43 / 6 | 0.47 |
| grouped | seq-step | auroc | +0.02 | 70 / 30 / 0 | 7.85e-05 |
| grouped | seq-step | block@0.10 | +5.16 pp | 62 / 31 / 7 | 0.00171 |
| grouped | graph-graph_untyped | auroc | +0.05 | 78 / 21 / 1 | 6.88e-09 |
| grouped | graph-graph_untyped | block@0.10 | +3.17 pp | 58 / 33 / 9 | 0.0115 |
| grouped | graph-logreg_pooled | auroc | +0.07 | 94 / 6 / 0 | 2.01e-21 |
| grouped | graph-logreg_pooled | block@0.10 | +19.35 pp | 78 / 21 / 1 | 6.88e-09 |

### LLM judge released with R-Judge (all records; fixed operating point, no calibration)

Meta-Llama-3.1-8B-Instruct (R-Judge authors' released predictions): Block 86.7%, FBR 85.2% (injection: 97.5 / 90.7; unintended: 65.3 / 64.3); 3 unparseable outputs counted as not flagged.

### Dataset artefact

The rule "block a record that ends on a tool call" (it sees the end of the record, which a runtime guard does not) gives Block 84.3% at FBR 3.9% on the test parts of the random splits.
