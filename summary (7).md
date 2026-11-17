# R-Judge experiment (full run)

- records: 571 (301 unsafe, 270 safe), 151 distinct user prompts, 1461 decision steps
- features: `st` (`BAAI/bge-base-en-v1.5`), dimension 768
- scorer parameters: {'step': 196865, 'seq': 915969, 'graph': 1089186, 'graph_untyped': 347271}
- record score = maximum step logit; threshold = conformal quantile of the safe calibration records; split 50/25/25

### Random splits (exchangeable: the FBR guarantee applies)

Mean ± sd over 100 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.957 ± 0.019 | 3.9 ± 3.8 | 63.2 ± 13.0 | 7.9 ± 4.5 | 77.0 ± 10.7 | 17.9 ± 5.7 | 95.0 ± 5.1 |
| seq | 0.962 ± 0.021 | 4.0 ± 3.3 | 68.0 ± 16.7 | 8.0 ± 4.2 | 81.6 ± 10.6 | 17.7 ± 6.3 | 97.6 ± 3.3 |
| graph | 0.958 ± 0.023 | 4.1 ± 3.2 | 66.5 ± 16.6 | 8.2 ± 4.6 | 80.7 ± 9.8 | 18.0 ± 6.2 | 96.9 ± 4.4 |
| graph_untyped | 0.906 ± 0.035 | 3.9 ± 2.9 | 53.6 ± 14.2 | 8.1 ± 4.4 | 64.1 ± 10.7 | 17.5 ± 6.2 | 81.2 ± 9.3 |
| logreg_pooled | 0.912 ± 0.026 | 4.3 ± 3.8 | 40.1 ± 17.1 | 7.8 ± 4.8 | 60.7 ± 14.3 | 18.2 ± 5.8 | 84.5 ± 8.3 |
| rule_last_tool | 0.902 ± 0.016 | 1.4 ± 2.4 | 26.1 ± 39.2 | 3.8 ± 2.1 | 78.3 ± 21.8 | 3.9 ± 2.0 | 84.3 ± 2.9 |

### Prompt-grouped splits (no user prompt shared between train and test)

Mean ± sd over 100 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.912 ± 0.060 | 10.0 ± 15.5 | 64.2 ± 18.8 | 17.2 ± 20.5 | 74.3 ± 19.4 | 32.2 ± 29.2 | 85.5 ± 15.4 |
| seq | 0.927 ± 0.064 | 8.8 ± 13.4 | 60.6 ± 27.1 | 16.8 ± 20.6 | 75.5 ± 23.0 | 29.9 ± 28.3 | 89.7 ± 14.8 |
| graph | 0.927 ± 0.057 | 9.1 ± 15.8 | 57.8 ± 27.9 | 17.1 ± 20.8 | 77.4 ± 20.4 | 28.7 ± 27.6 | 89.8 ± 13.3 |
| graph_untyped | 0.865 ± 0.076 | 7.8 ± 10.9 | 60.3 ± 13.2 | 17.4 ± 21.9 | 70.9 ± 14.9 | 28.3 ± 27.5 | 78.9 ± 14.5 |
| logreg_pooled | 0.858 ± 0.076 | 9.0 ± 15.2 | 40.1 ± 25.3 | 16.8 ± 21.3 | 59.3 ± 25.0 | 28.7 ± 28.6 | 75.2 ± 20.2 |
| rule_last_tool | 0.891 ± 0.047 | 2.4 ± 3.7 | 33.5 ± 40.7 | 4.6 ± 4.0 | 67.2 ± 33.5 | 5.2 ± 3.6 | 83.5 ± 7.7 |

Calibration unit on the grouped splits: FBR (%) at alpha = 0.10 with the threshold set
from all safe calibration records (pooled) and from one random safe record per
calibration prompt group, evaluated on one random record per test group.

| scorer | FBR pooled | FBR one per group | Block one per group |
|---|---|---|---|
| step | 17.2 ± 20.5 | 6.4 ± 8.3 | 26.4 ± 16.7 |
| seq | 16.8 ± 20.6 | 7.2 ± 8.4 | 21.8 ± 15.8 |
| graph | 17.1 ± 20.8 | 7.6 ± 8.7 | 21.1 ± 15.0 |
| graph_untyped | 17.4 ± 21.9 | 6.4 ± 7.7 | 25.6 ± 14.9 |
| logreg_pooled | 16.8 ± 21.3 | 5.6 ± 8.6 | 19.0 ± 14.6 |
| rule_last_tool | 4.6 ± 4.0 | 1.9 ± 6.3 | 6.1 ± 18.7 |

### Leave-one-category-out, test on Application (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.862 ± 0.048 | 22.0 ± 10.0 | 73.3 ± 18.6 | 40.0 ± 17.5 | 86.1 ± 16.1 | 82.0 ± 18.2 | 95.7 ± 9.5 |
| seq | 0.889 ± 0.101 | 5.3 ± 5.9 | 41.9 ± 32.6 | 12.9 ± 17.7 | 67.8 ± 33.6 | 72.1 ± 32.7 | 97.9 ± 3.1 |
| graph | 0.949 ± 0.017 | 2.9 ± 2.9 | 39.0 ± 36.5 | 6.5 ± 4.6 | 54.3 ± 45.2 | 65.8 ± 36.3 | 98.9 ± 0.7 |
| graph_untyped | 0.836 ± 0.030 | 4.8 ± 2.4 | 56.3 ± 7.5 | 9.1 ± 2.5 | 64.9 ± 5.3 | 45.5 ± 33.4 | 82.5 ± 12.9 |
| logreg_pooled | 0.788 ± 0.023 | 57.5 ± 21.1 | 88.2 ± 6.1 | 86.5 ± 2.5 | 95.2 ± 1.4 | 93.7 ± 2.8 | 98.5 ± 0.9 |
| rule_last_tool | 0.922 ± 0.000 | 3.6 ± 3.8 | 45.8 ± 48.3 | 7.2 ± 0.0 | 91.6 ± 0.0 | 7.2 ± 0.0 | 91.6 ± 0.0 |

### Leave-one-category-out, test on Finance (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.997 ± 0.001 | 0.1 ± 0.4 | 71.0 ± 11.7 | 0.8 ± 0.8 | 86.9 ± 5.0 | 11.7 ± 27.0 | 97.9 ± 3.4 |
| seq | 0.995 ± 0.002 | 0.9 ± 1.1 | 69.5 ± 28.2 | 1.6 ± 1.1 | 79.2 ± 28.7 | 2.5 ± 0.9 | 95.9 ± 2.8 |
| graph | 0.996 ± 0.001 | 0.9 ± 1.1 | 77.2 ± 15.5 | 1.4 ± 1.1 | 87.2 ± 7.5 | 3.4 ± 1.1 | 96.7 ± 2.7 |
| graph_untyped | 0.978 ± 0.017 | 0.3 ± 0.8 | 82.3 ± 7.5 | 0.9 ± 0.9 | 86.2 ± 5.7 | 1.7 ± 0.8 | 91.5 ± 5.5 |
| logreg_pooled | 0.982 ± 0.005 | 1.3 ± 0.8 | 45.9 ± 16.3 | 2.4 ± 1.1 | 76.2 ± 11.5 | 13.3 ± 23.1 | 96.4 ± 2.5 |
| rule_last_tool | 0.866 ± 0.000 | 0.1 ± 0.4 | 7.4 ± 23.5 | 1.1 ± 0.0 | 74.4 ± 0.0 | 1.1 ± 0.0 | 74.4 ± 0.0 |

### Leave-one-category-out, test on IoT (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.547 ± 0.028 | 61.8 ± 11.2 | 63.7 ± 28.1 | 75.5 ± 9.6 | 89.5 ± 14.3 | 88.2 ± 7.5 | 100.0 ± 0.0 |
| seq | 0.688 ± 0.065 | 13.6 ± 12.3 | 36.8 ± 19.2 | 52.7 ± 20.5 | 79.5 ± 20.1 | 94.5 ± 4.7 | 100.0 ± 0.0 |
| graph | 0.660 ± 0.047 | 28.2 ± 25.9 | 39.5 ± 26.9 | 50.0 ± 25.1 | 65.8 ± 27.8 | 94.5 ± 4.7 | 99.5 ± 1.7 |
| graph_untyped | 0.660 ± 0.041 | 31.8 ± 18.8 | 50.5 ± 21.4 | 50.9 ± 15.0 | 69.5 ± 14.2 | 84.5 ± 6.1 | 94.2 ± 4.6 |
| logreg_pooled | 0.573 ± 0.027 | 58.2 ± 14.3 | 74.7 ± 17.7 | 66.4 ± 4.4 | 94.2 ± 9.1 | 81.8 ± 8.6 | 100.0 ± 0.0 |
| rule_last_tool | 0.658 ± 0.000 | 0.0 ± 0.0 | 15.8 ± 16.6 | 0.0 ± 0.0 | 31.6 ± 0.0 | 0.0 ± 0.0 | 31.6 ± 0.0 |

### Leave-one-category-out, test on Program (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.915 ± 0.025 | 2.7 ± 2.5 | 46.5 ± 21.7 | 8.0 ± 3.8 | 71.6 ± 12.4 | 45.0 ± 26.2 | 93.5 ± 4.6 |
| seq | 0.903 ± 0.020 | 3.2 ± 2.4 | 45.7 ± 19.9 | 9.8 ± 6.2 | 61.9 ± 18.2 | 55.8 ± 37.2 | 91.3 ± 10.4 |
| graph | 0.879 ± 0.083 | 2.7 ± 1.8 | 48.8 ± 6.2 | 7.5 ± 2.3 | 52.9 ± 6.2 | 52.5 ± 36.3 | 88.5 ± 14.2 |
| graph_untyped | 0.858 ± 0.094 | 1.5 ± 1.7 | 46.5 ± 5.2 | 7.0 ± 2.6 | 55.9 ± 13.2 | 46.8 ± 34.9 | 86.2 ± 18.0 |
| logreg_pooled | 0.800 ± 0.040 | 7.5 ± 17.3 | 50.3 ± 17.9 | 48.5 ± 22.8 | 78.4 ± 12.0 | 77.3 ± 11.8 | 93.2 ± 4.5 |
| rule_last_tool | 0.925 ± 0.000 | 0.3 ± 0.7 | 17.4 ± 36.6 | 1.7 ± 0.0 | 86.8 ± 0.0 | 1.7 ± 0.0 | 86.8 ± 0.0 |

### Leave-one-category-out, test on Web (distribution shift: no guarantee)

Mean ± sd over 10 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.773 ± 0.028 | 4.7 ± 4.5 | 32.0 ± 12.7 | 16.0 ± 14.8 | 50.0 ± 13.5 | 66.0 ± 25.0 | 88.0 ± 14.4 |
| seq | 0.804 ± 0.068 | 8.0 ± 6.1 | 41.0 ± 15.2 | 33.3 ± 18.1 | 73.0 ± 14.6 | 79.3 ± 22.1 | 99.0 ± 3.2 |
| graph | 0.739 ± 0.044 | 12.7 ± 6.6 | 35.5 ± 19.2 | 26.7 ± 7.0 | 60.0 ± 14.9 | 92.0 ± 12.5 | 97.5 ± 2.6 |
| graph_untyped | 0.780 ± 0.059 | 4.0 ± 7.2 | 35.0 ± 7.5 | 16.0 ± 9.5 | 51.5 ± 10.8 | 54.0 ± 26.7 | 81.5 ± 15.8 |
| logreg_pooled | 0.680 ± 0.029 | 2.0 ± 4.5 | 21.5 ± 10.3 | 7.3 ± 10.6 | 35.5 ± 15.0 | 44.0 ± 11.4 | 67.5 ± 11.1 |
| rule_last_tool | 0.825 ± 0.000 | 12.0 ± 10.3 | 51.0 ± 43.9 | 20.0 ± 0.0 | 85.0 ± 0.0 | 20.0 ± 0.0 | 85.0 ± 0.0 |

### Paired comparisons (a minus b, per split; exact sign test)

| protocol | a - b | metric | mean diff | a better / b better / ties | p |
|---|---|---|---|---|---|
| random | graph-seq | auroc | -0.00 | 34 / 66 / 0 | 0.00179 |
| random | graph-seq | block@0.10 | -0.91 pp | 37 / 56 / 7 | 0.0614 |
| random | graph-step | auroc | +0.00 | 56 / 43 / 1 | 0.228 |
| random | graph-step | block@0.10 | +3.69 pp | 60 / 36 / 4 | 0.0184 |
| random | seq-step | auroc | +0.01 | 64 / 36 / 0 | 0.00664 |
| random | seq-step | block@0.10 | +4.60 pp | 64 / 35 / 1 | 0.00464 |
| random | graph-graph_untyped | auroc | +0.05 | 92 / 8 / 0 | 3.21e-19 |
| random | graph-graph_untyped | block@0.10 | +16.57 pp | 89 / 9 / 2 | 1.1e-17 |
| random | graph-logreg_pooled | auroc | +0.05 | 94 / 6 / 0 | 2.01e-21 |
| random | graph-logreg_pooled | block@0.10 | +19.99 pp | 92 / 7 / 1 | 5.07e-20 |
| grouped | graph-seq | auroc | -0.00 | 35 / 64 / 1 | 0.00464 |
| grouped | graph-seq | block@0.10 | +1.93 pp | 41 / 42 / 17 | 1 |
| grouped | graph-step | auroc | +0.01 | 64 / 36 / 0 | 0.00664 |
| grouped | graph-step | block@0.10 | +3.12 pp | 51 / 41 / 8 | 0.348 |
| grouped | seq-step | auroc | +0.02 | 70 / 30 / 0 | 7.85e-05 |
| grouped | seq-step | block@0.10 | +1.18 pp | 48 / 46 / 6 | 0.918 |
| grouped | graph-graph_untyped | auroc | +0.06 | 82 / 18 / 0 | 6.15e-11 |
| grouped | graph-graph_untyped | block@0.10 | +6.55 pp | 69 / 23 / 8 | 1.67e-06 |
| grouped | graph-logreg_pooled | auroc | +0.07 | 92 / 8 / 0 | 3.21e-19 |
| grouped | graph-logreg_pooled | block@0.10 | +18.11 pp | 77 / 21 / 2 | 1.09e-08 |

### LLM judge released with R-Judge (all records; fixed operating point, no calibration)

Meta-Llama-3.1-8B-Instruct (R-Judge authors' released predictions): Block 86.7%, FBR 85.2% (injection: 97.5 / 90.7; unintended: 65.3 / 64.3); 3 unparseable outputs counted as not flagged.

### Dataset artefact

The rule "block a record that ends on a tool call" (it sees the end of the record, which a runtime guard does not) gives Block 84.3% at FBR 3.9% on the test parts of the random splits.
