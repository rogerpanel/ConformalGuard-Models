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
| graph | 0.956 ± 0.020 | 3.8 ± 2.9 | 63.1 ± 19.2 | 8.0 ± 4.2 | 77.8 ± 13.8 | 17.9 ± 6.4 | 96.6 ± 4.6 |
| graph_untyped | 0.906 ± 0.030 | 3.8 ± 3.0 | 54.5 ± 11.5 | 8.2 ± 4.3 | 63.3 ± 9.6 | 18.0 ± 6.2 | 80.6 ± 9.1 |
| logreg_pooled | 0.912 ± 0.026 | 4.3 ± 3.7 | 31.7 ± 19.8 | 7.9 ± 4.3 | 60.3 ± 17.5 | 17.6 ± 5.0 | 87.2 ± 6.0 |
| rule_last_tool | 0.902 ± 0.016 | 1.4 ± 2.4 | 26.1 ± 39.2 | 3.8 ± 2.1 | 78.3 ± 21.8 | 3.9 ± 2.0 | 84.3 ± 2.9 |

Category-conditional (Mondrian) calibration on the random splits, alpha = 0.10: FBR (%) per
category with one pooled threshold, and with one threshold per category.

| scorer | Application pooled / per-cat | Finance pooled / per-cat | IoT pooled / per-cat | Program pooled / per-cat | Web pooled / per-cat | all: FBR per-cat | all: Block per-cat |
|---|---|---|---|---|---|---|---|
| step | 8.7 ± 7.4 / 5.7 ± 6.3 | 1.1 ± 2.0 / 6.9 ± 8.0 | 47.0 ± 37.3 / 0.0 ± 0.0 | 4.9 ± 6.3 / 6.1 ± 7.8 | 14.2 ± 21.0 / 0.0 ± 0.0 | 5.6 ± 3.9 | 69.2 ± 10.8 |
| seq | 8.4 ± 6.1 / 8.3 ± 7.8 | 1.5 ± 2.2 / 6.8 ± 9.3 | 43.9 ± 33.4 / 0.0 ± 0.0 | 8.9 ± 6.9 / 6.4 ± 7.6 | 22.2 ± 26.7 / 0.0 ± 0.0 | 6.5 ± 4.5 | 70.8 ± 13.4 |
| graph | 7.8 ± 6.3 / 7.1 ± 7.4 | 1.5 ± 2.3 / 5.9 ± 7.6 | 43.9 ± 34.7 / 0.0 ± 0.0 | 7.6 ± 7.0 / 5.6 ± 7.8 | 24.2 ± 25.0 / 0.0 ± 0.0 | 5.6 ± 3.9 | 67.1 ± 13.6 |
| graph_untyped | 9.8 ± 7.0 / 6.3 ± 6.7 | 0.5 ± 1.4 / 6.6 ± 6.9 | 39.1 ± 34.9 / 0.0 ± 0.0 | 8.0 ± 7.5 / 6.2 ± 7.4 | 23.4 ± 26.0 / 0.0 ± 0.0 | 5.7 ± 3.6 | 56.4 ± 9.8 |
| logreg_pooled | 9.0 ± 6.7 / 7.5 ± 7.4 | 1.3 ± 2.4 / 7.0 ± 7.3 | 49.2 ± 30.3 / 0.0 ± 0.0 | 6.7 ± 6.4 / 6.6 ± 7.6 | 15.0 ± 20.8 / 0.0 ± 0.0 | 6.3 ± 3.8 | 49.4 ± 16.9 |
| rule_last_tool | 5.7 ± 4.7 / 2.8 ± 4.5 | 0.9 ± 1.9 / 0.9 ± 1.9 | 0.0 ± 0.0 / 0.0 ± 0.0 | 1.8 ± 3.1 / 1.8 ± 3.1 | 18.7 ± 21.5 / 0.0 ± 0.0 | 1.7 ± 1.9 | 42.7 ± 25.0 |

### In-domain recalibration: train on four categories, calibrate and test on halves of the fifth

Mean ± sd over 20 splits per category, alpha = 0.10. Calibration and test records are
exchangeable, so the guarantee applies. 'finite' is the share of splits in which the category had
enough safe calibration records for a finite threshold (otherwise nothing is blocked).

| held-out category | scorer | FBR@.10 | Block@.10 | finite | safe cal records |
|---|---|---|---|---|---|
| Application | step | 6.0 ± 5.3 | 58.2 ± 8.8 | 100.0 ± 0.0 | 48 |
| Application | seq | 6.8 ± 4.7 | 52.0 ± 39.8 | 100.0 ± 0.0 | 48 |
| Application | graph | 6.4 ± 5.7 | 55.8 ± 34.8 | 100.0 ± 0.0 | 48 |
| Application | graph_untyped | 6.5 ± 5.0 | 61.4 ± 9.7 | 100.0 ± 0.0 | 48 |
| Application | logreg_pooled | 7.0 ± 5.4 | 50.1 ± 6.4 | 100.0 ± 0.0 | 48 |
| Application | rule_last_tool | 5.0 ± 5.4 | 46.2 ± 47.5 | 100.0 ± 0.0 | 48 |
| Finance | step | 5.3 ± 5.1 | 100.0 ± 0.0 | 100.0 ± 0.0 | 44 |
| Finance | seq | 6.5 ± 4.9 | 98.9 ± 2.8 | 100.0 ± 0.0 | 44 |
| Finance | graph | 6.4 ± 3.9 | 100.0 ± 0.0 | 100.0 ± 0.0 | 44 |
| Finance | graph_untyped | 5.0 ± 4.4 | 98.4 ± 3.5 | 100.0 ± 0.0 | 44 |
| Finance | logreg_pooled | 10.2 ± 7.5 | 100.0 ± 0.0 | 100.0 ± 0.0 | 44 |
| Finance | rule_last_tool | 0.8 ± 1.1 | 73.7 ± 4.8 | 100.0 ± 0.0 | 44 |
| IoT | step | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6 |
| IoT | seq | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6 |
| IoT | graph | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6 |
| IoT | graph_untyped | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6 |
| IoT | logreg_pooled | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6 |
| IoT | rule_last_tool | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6 |
| Program | step | 10.7 ± 7.9 | 90.0 ± 9.1 | 100.0 ± 0.0 | 30 |
| Program | seq | 8.2 ± 6.9 | 65.9 ± 14.0 | 100.0 ± 0.0 | 30 |
| Program | graph | 8.8 ± 8.8 | 68.4 ± 16.6 | 100.0 ± 0.0 | 30 |
| Program | graph_untyped | 11.7 ± 7.8 | 67.6 ± 15.6 | 100.0 ± 0.0 | 30 |
| Program | logreg_pooled | 10.7 ± 7.1 | 70.6 ± 18.3 | 100.0 ± 0.0 | 30 |
| Program | rule_last_tool | 1.8 ± 1.7 | 87.4 ± 4.9 | 100.0 ± 0.0 | 30 |
| Web | step | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8 |
| Web | seq | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8 |
| Web | graph | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8 |
| Web | graph_untyped | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8 |
| Web | logreg_pooled | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8 |
| Web | rule_last_tool | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8 |

### Paired comparisons (a minus b, per split; exact sign test)

| protocol | a - b | metric | mean diff | a better / b better / ties | p |
|---|---|---|---|---|---|
| random | graph-seq | auroc | -0.01 | 38 / 62 / 0 | 0.021 |
| random | graph-seq | block@0.10 | -4.51 pp | 31 / 62 / 7 | 0.00171 |
| random | graph-step | auroc | -0.00 | 49 / 50 / 1 | 1 |
| random | graph-step | block@0.10 | +0.08 pp | 49 / 49 / 2 | 1 |
| random | seq-step | auroc | +0.00 | 58 / 41 / 1 | 0.107 |
| random | seq-step | block@0.10 | +4.59 pp | 62 / 32 / 6 | 0.00259 |
| random | graph-graph_untyped | auroc | +0.05 | 94 / 6 / 0 | 2.01e-21 |
| random | graph-graph_untyped | block@0.10 | +14.55 pp | 83 / 14 / 3 | 4.22e-13 |
| random | graph-logreg_pooled | auroc | +0.04 | 95 / 5 / 0 | 1.25e-22 |
| random | graph-logreg_pooled | block@0.10 | +17.51 pp | 87 / 11 / 2 | 8.07e-16 |

### LLM judge released with R-Judge (all records; fixed operating point, no calibration)

Meta-Llama-3.1-8B-Instruct (R-Judge authors' released predictions): Block 86.7%, FBR 85.2% (injection: 97.5 / 90.7; unintended: 65.3 / 64.3); 3 unparseable outputs counted as not flagged.

### Dataset artefact

The rule "block a record that ends on a tool call" (it sees the end of the record, which a runtime guard does not) gives Block 84.3% at FBR 3.9% on the test parts of the random splits.
