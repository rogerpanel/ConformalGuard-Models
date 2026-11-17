# R-Judge results: CPU run with TF-IDF + SVD features

Produced by `scripts/rjudge_run.py --features lsa --reps 40 --loco-reps 3` (one process per
protocol) on a 4-core CPU. Raw numbers for every split are in `results_*.json` next to this file.
The protocol and its limits are in `docs/RJUDGE_EXPERIMENT.md`; read the limits before quoting
anything here. A GPU run with sentence-encoder features and more splits supersedes this one.

- R-Judge commit `83ce301da3`: 571 records (301 unsafe, 270 safe), 151 distinct user prompts, 1461 decision steps
- node features: TF-IDF (1-2 grams) + truncated SVD, 256 dimensions, fitted on node texts only
- scorer parameters: {'step': 65793, 'seq': 850433, 'graph': 1023650, 'graph_untyped': 281735}
- 40 epochs, hidden size 128, 3 layers; split 50/25/25; alpha in [0.05, 0.1, 0.2]
- torch 2.15.0+cu132, numpy 2.5.3, python 3.13.18

## What the numbers say

1. **The false-block guarantee holds on exchangeable splits, for every scorer.** Mean false-block
   rate over 40 random splits, in %: at alpha = 0.05: step 4.7, seq 4.3, graph 4.3, graph_untyped 3.6, logreg_pooled 4.6; at alpha = 0.10: step 8.3, seq 8.3, graph 9.1, graph_untyped 9.0, logreg_pooled 8.0;
   at alpha = 0.20: step 18.7, seq 19.3, graph 18.7, graph_untyped 18.1, logreg_pooled 18.2.
2. **It does not hold when exchangeability fails**, as the theory says. With prompt-grouped
   splits the mean false-block rate at alpha = 0.10 is between 12.8% and 17.6% for the
   learned scorers. With a held-out application category it reaches 54.5% (IoT) and
   31.1% (Web) for the graph scorer; those two categories have only 11 and 15 safe records.
3. **The graph scorer is not better than the sequence scorer or the single-step scorer here.**
   On random splits at alpha = 0.10 the graph scorer blocks 80.6% of unsafe records, the
   sequence scorer 77.1% and the single-step scorer 81.6%. Neither paired difference (graph minus
   sequence, graph minus single step) is significant on either split protocol (paired table). On R-Judge, history and graph
   structure add nothing measurable beyond the current step.
4. **Types matter within the graph model.** Removing node and relation types costs
   18.1 points of block rate on random splits and 12.5 on grouped splits.
5. **A trivial rule does as well as the learned scorers.** "The record ends on a tool call" blocks
   84.2% of unsafe records at a false-block rate of 3.8%. A runtime guard cannot use this
   rule, but it shows how much of R-Judge's label is carried by where the record stops.
6. **The released LLM judge flags almost everything**: Block 86.7% at a false-block rate of
   85.2% (Llama-3.1-8B, predictions released by the R-Judge authors).
7. **Two-sided guard** (alpha = beta = 0.10, graph scorer, random splits): 9.1% of safe records
   blocked, 9.8% of unsafe records auto-allowed, 7.0% of records escalated.

## Tables

### Random splits (exchangeable: the FBR guarantee applies)

Mean ± sd over 40 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.959 ± 0.017 | 4.7 ± 3.4 | 71.6 ± 11.2 | 8.3 ± 4.8 | 81.6 ± 7.4 | 18.7 ± 7.2 | 94.4 ± 3.8 |
| seq | 0.950 ± 0.026 | 4.3 ± 3.1 | 62.5 ± 19.1 | 8.3 ± 4.9 | 77.1 ± 12.6 | 19.3 ± 6.8 | 97.3 ± 2.4 |
| graph | 0.951 ± 0.022 | 4.3 ± 3.1 | 67.9 ± 15.5 | 9.1 ± 4.9 | 80.6 ± 10.8 | 18.7 ± 6.4 | 96.6 ± 3.8 |
| graph_untyped | 0.895 ± 0.028 | 3.6 ± 3.0 | 47.9 ± 14.8 | 9.0 ± 4.7 | 62.5 ± 9.0 | 18.1 ± 5.4 | 80.7 ± 7.2 |
| logreg_pooled | 0.946 ± 0.018 | 4.6 ± 4.4 | 55.9 ± 15.4 | 8.0 ± 5.0 | 74.2 ± 12.9 | 18.2 ± 5.5 | 93.9 ± 4.0 |
| rule_last_tool | 0.902 ± 0.015 | 1.4 ± 2.6 | 25.3 ± 39.1 | 3.6 ± 2.3 | 78.0 ± 22.7 | 3.8 ± 2.1 | 84.2 ± 3.1 |


### Prompt-grouped splits (no user prompt shared between train and test)

Mean ± sd over 40 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.924 ± 0.042 | 7.6 ± 8.8 | 67.0 ± 15.8 | 15.2 ± 16.2 | 80.5 ± 11.7 | 25.7 ± 20.2 | 90.2 ± 7.0 |
| seq | 0.926 ± 0.049 | 6.7 ± 7.7 | 57.5 ± 26.3 | 15.4 ± 17.7 | 75.2 ± 19.8 | 28.2 ± 23.3 | 90.7 ± 11.7 |
| graph | 0.925 ± 0.044 | 7.0 ± 8.2 | 58.8 ± 21.3 | 12.8 ± 13.1 | 75.4 ± 17.0 | 25.9 ± 20.6 | 88.8 ± 12.6 |
| graph_untyped | 0.823 ± 0.074 | 7.4 ± 10.5 | 53.3 ± 16.9 | 17.6 ± 22.2 | 62.9 ± 18.7 | 29.8 ± 28.6 | 74.3 ± 17.6 |
| logreg_pooled | 0.924 ± 0.047 | 7.6 ± 11.2 | 73.2 ± 14.3 | 16.2 ± 14.9 | 83.6 ± 10.4 | 27.5 ± 21.0 | 90.2 ± 6.4 |
| rule_last_tool | 0.887 ± 0.046 | 2.1 ± 3.3 | 32.1 ± 40.2 | 4.1 ± 3.6 | 63.2 ± 35.3 | 4.9 ± 3.1 | 82.2 ± 8.0 |


### Leave-one-category-out, test on Application (distribution shift: no guarantee)

Mean ± sd over 3 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.926 ± 0.022 | 0.7 ± 0.6 | 29.5 ± 21.0 | 1.7 ± 2.1 | 49.5 ± 36.5 | 34.7 ± 34.9 | 87.7 ± 14.7 |
| seq | 0.956 ± 0.033 | 1.4 ± 2.4 | 43.7 ± 42.9 | 7.9 ± 3.0 | 92.0 ± 3.2 | 55.3 ± 10.3 | 98.9 ± 0.4 |
| graph | 0.974 ± 0.005 | 2.4 ± 1.6 | 60.4 ± 51.8 | 7.2 ± 3.1 | 94.6 ± 3.0 | 93.8 ± 7.1 | 99.4 ± 0.6 |
| graph_untyped | 0.793 ± 0.031 | 0.7 ± 1.2 | 49.0 ± 9.4 | 7.9 ± 1.6 | 65.6 ± 1.5 | 11.3 ± 0.0 | 67.7 ± 0.0 |
| logreg_pooled | 0.943 ± 0.006 | 6.5 ± 4.2 | 77.8 ± 16.1 | 37.5 ± 27.9 | 94.0 ± 5.5 | 86.3 ± 5.2 | 98.1 ± 0.6 |
| rule_last_tool | 0.922 ± 0.000 | 4.8 ± 4.2 | 61.1 ± 52.9 | 7.2 ± 0.0 | 91.6 ± 0.0 | 7.2 ± 0.0 | 91.6 ± 0.0 |


### Leave-one-category-out, test on Finance (distribution shift: no guarantee)

Mean ± sd over 3 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.945 ± 0.023 | 0.0 ± 0.0 | 65.0 ± 15.5 | 0.4 ± 0.7 | 77.8 ± 10.4 | 5.4 ± 6.3 | 91.5 ± 6.5 |
| seq | 0.982 ± 0.013 | 0.0 ± 0.0 | 57.3 ± 32.1 | 0.8 ± 1.3 | 90.6 ± 5.3 | 3.1 ± 1.3 | 94.9 ± 0.0 |
| graph | 0.994 ± 0.004 | 0.0 ± 0.0 | 73.5 ± 1.5 | 1.5 ± 1.3 | 84.6 ± 0.0 | 2.3 ± 0.0 | 94.9 ± 0.0 |
| graph_untyped | 0.954 ± 0.000 | 0.0 ± 0.0 | 65.0 ± 5.9 | 0.0 ± 0.0 | 79.5 ± 2.6 | 1.1 ± 0.0 | 85.5 ± 5.3 |
| logreg_pooled | 0.988 ± 0.001 | 1.1 ± 0.0 | 76.1 ± 15.0 | 1.9 ± 0.7 | 86.3 ± 5.3 | 7.7 ± 1.3 | 98.3 ± 1.5 |
| rule_last_tool | 0.866 ± 0.000 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.1 ± 0.0 | 74.4 ± 0.0 | 1.1 ± 0.0 | 74.4 ± 0.0 |


### Leave-one-category-out, test on IoT (distribution shift: no guarantee)

Mean ± sd over 3 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.566 ± 0.047 | 15.2 ± 5.2 | 17.5 ± 11.0 | 33.3 ± 18.9 | 40.4 ± 8.0 | 81.8 ± 15.7 | 93.0 ± 12.2 |
| seq | 0.617 ± 0.059 | 18.2 ± 18.2 | 33.3 ± 31.7 | 51.5 ± 22.9 | 64.9 ± 31.7 | 100.0 ± 0.0 | 96.5 ± 3.0 |
| graph | 0.647 ± 0.075 | 21.2 ± 13.9 | 38.6 ± 8.0 | 54.5 ± 0.0 | 78.9 ± 10.5 | 97.0 ± 5.2 | 96.5 ± 3.0 |
| graph_untyped | 0.569 ± 0.094 | 42.4 ± 22.9 | 54.4 ± 24.9 | 57.6 ± 10.5 | 77.2 ± 3.0 | 81.8 ± 9.1 | 87.7 ± 3.0 |
| logreg_pooled | 0.684 ± 0.025 | 39.4 ± 5.2 | 63.2 ± 15.8 | 75.8 ± 18.9 | 94.7 ± 9.1 | 87.9 ± 5.2 | 100.0 ± 0.0 |
| rule_last_tool | 0.658 ± 0.000 | 0.0 ± 0.0 | 10.5 ± 18.2 | 0.0 ± 0.0 | 31.6 ± 0.0 | 0.0 ± 0.0 | 31.6 ± 0.0 |


### Leave-one-category-out, test on Program (distribution shift: no guarantee)

Mean ± sd over 3 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.953 ± 0.018 | 0.6 ± 1.0 | 51.0 ± 9.5 | 3.9 ± 5.4 | 66.2 ± 15.4 | 26.7 ± 27.4 | 89.7 ± 7.6 |
| seq | 0.926 ± 0.039 | 4.4 ± 2.5 | 57.4 ± 18.8 | 8.3 ± 1.7 | 66.7 ± 13.7 | 41.1 ± 46.7 | 93.6 ± 4.7 |
| graph | 0.879 ± 0.007 | 3.3 ± 2.9 | 33.8 ± 24.7 | 7.2 ± 3.5 | 49.5 ± 13.6 | 43.3 ± 44.9 | 89.7 ± 5.3 |
| graph_untyped | 0.895 ± 0.009 | 6.1 ± 3.8 | 39.7 ± 8.9 | 11.1 ± 3.8 | 46.1 ± 1.7 | 22.2 ± 15.4 | 66.2 ± 25.5 |
| logreg_pooled | 0.921 ± 0.006 | 1.7 ± 1.7 | 61.8 ± 17.8 | 11.7 ± 18.8 | 72.5 ± 14.8 | 57.2 ± 24.1 | 95.6 ± 1.5 |
| rule_last_tool | 0.925 ± 0.000 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.7 ± 0.0 | 86.8 ± 0.0 | 1.7 ± 0.0 | 86.8 ± 0.0 |


### Leave-one-category-out, test on Web (distribution shift: no guarantee)

Mean ± sd over 3 splits. FBR and Block in %. Target FBR = alpha.

| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|---|
| step | 0.867 ± 0.068 | 6.7 ± 6.7 | 58.3 ± 5.8 | 15.6 ± 3.8 | 71.7 ± 10.4 | 82.2 ± 7.7 | 100.0 ± 0.0 |
| seq | 0.864 ± 0.046 | 11.1 ± 3.8 | 58.3 ± 12.6 | 33.3 ± 0.0 | 80.0 ± 17.3 | 80.0 ± 29.1 | 100.0 ± 0.0 |
| graph | 0.819 ± 0.098 | 24.4 ± 10.2 | 70.0 ± 10.0 | 31.1 ± 3.8 | 78.3 ± 16.1 | 62.2 ± 32.9 | 96.7 ± 5.8 |
| graph_untyped | 0.744 ± 0.068 | 13.3 ± 13.3 | 36.7 ± 12.6 | 24.4 ± 10.2 | 48.3 ± 7.6 | 33.3 ± 6.7 | 70.0 ± 8.7 |
| logreg_pooled | 0.877 ± 0.024 | 2.2 ± 3.8 | 28.3 ± 7.6 | 17.8 ± 3.8 | 68.3 ± 7.6 | 28.9 ± 3.8 | 96.7 ± 2.9 |
| rule_last_tool | 0.825 ± 0.000 | 13.3 ± 11.5 | 56.7 ± 49.1 | 20.0 ± 0.0 | 85.0 ± 0.0 | 20.0 ± 0.0 | 85.0 ± 0.0 |


### Paired comparisons (a minus b, per split; exact sign test)

| protocol | a - b | metric | mean diff | a better / b better / ties | p |
|---|---|---|---|---|---|
| random | graph-seq | auroc | +0.00 | 18 / 22 / 0 | 0.636 |
| random | graph-seq | block@0.10 | +3.50 pp | 23 / 12 / 5 | 0.0895 |
| random | graph-step | auroc | -0.01 | 14 / 25 / 1 | 0.108 |
| random | graph-step | block@0.10 | -1.03 pp | 20 / 18 / 2 | 0.871 |
| random | graph-graph_untyped | auroc | +0.06 | 39 / 1 / 0 | 7.46e-11 |
| random | graph-graph_untyped | block@0.10 | +18.10 pp | 35 / 5 / 0 | 1.38e-06 |
| random | graph-logreg_pooled | auroc | +0.01 | 26 / 13 / 1 | 0.0533 |
| random | graph-logreg_pooled | block@0.10 | +6.40 pp | 29 / 8 / 3 | 0.000753 |
| grouped | graph-seq | auroc | -0.00 | 16 / 24 / 0 | 0.268 |
| grouped | graph-seq | block@0.10 | +0.11 pp | 18 / 20 / 2 | 0.871 |
| grouped | graph-step | auroc | +0.00 | 18 / 22 / 0 | 0.636 |
| grouped | graph-step | block@0.10 | -5.18 pp | 15 / 23 / 2 | 0.256 |
| grouped | graph-graph_untyped | auroc | +0.10 | 37 / 3 / 0 | 1.95e-08 |
| grouped | graph-graph_untyped | block@0.10 | +12.49 pp | 31 / 8 / 1 | 0.000294 |
| grouped | graph-logreg_pooled | auroc | +0.00 | 21 / 19 / 0 | 0.875 |
| grouped | graph-logreg_pooled | block@0.10 | -8.22 pp | 12 / 26 / 2 | 0.0336 |

### LLM judge released with R-Judge (all records; fixed operating point, no calibration)

Meta-Llama-3.1-8B-Instruct (R-Judge authors' released predictions): Block 86.7%, FBR 85.2% (injection: 97.5 / 90.7; unintended: 65.3 / 64.3); 3 unparseable outputs counted as not flagged.
