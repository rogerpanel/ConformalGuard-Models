# Conformal calibration of guard scores on R-Judge

1000 random halves (calibration / test) per protocol. FBR and Block in %, mean ± sd. Target FBR = alpha.

| guard | view | records | no-score rows | AUROC | own operating point: FBR / Block |
|---|---|---|---|---|---|
| local_causal (Qwen/Qwen2.5-7B-Instruct) | causal | 571 | 0 | 0.780 | 85.6 / 93.0 |
| local_full (Qwen/Qwen2.5-7B-Instruct) | full | 571 | 0 | 0.758 | 81.5 / 89.0 |
| claude_causal (claude-opus-5-5) | causal | 571 | 1 | 0.992 | 2.6 / 94.0 |
| claude_full (claude-opus-5-5) | full | 571 | 1 | 0.994 | 1.5 / 94.0 |
| openai_causal (gpt-6-astra) | causal | 571 | 0 | 0.972 | 8.5 / 91.4 |
| openai_full (gpt-6-astra) | full | 571 | 0 | 0.971 | 10.4 / 92.0 |

### Random halves (exchangeable: the guarantee applies)

| guard | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|
| local_causal | 4.4 ± 2.5 | 41.3 ± 6.0 | 9.4 ± 3.5 | 51.3 ± 4.1 | 19.6 ± 4.8 | 66.0 ± 3.9 |
| local_full | 4.4 ± 2.4 | 40.3 ± 4.4 | 9.7 ± 3.5 | 54.3 ± 4.5 | 20.0 ± 4.9 | 66.2 ± 3.6 |
| claude_causal | 4.1 ± 2.5 | 95.2 ± 1.5 | 8.7 ± 3.4 | 97.4 ± 1.2 | 14.9 ± 2.3 | 99.3 ± 0.5 |
| claude_full | 4.1 ± 2.2 | 96.5 ± 1.2 | 8.6 ± 3.4 | 98.3 ± 0.9 | 12.2 ± 3.2 | 98.7 ± 0.7 |
| openai_causal | 3.5 ± 2.6 | 88.3 ± 2.1 | 9.0 ± 3.4 | 91.5 ± 1.7 | 18.5 ± 4.3 | 94.3 ± 1.7 |
| openai_full | 3.6 ± 2.6 | 88.6 ± 2.4 | 8.9 ± 3.6 | 91.7 ± 1.7 | 18.5 ± 4.2 | 94.4 ± 1.6 |

### Prompt-grouped halves, alpha = 0.10

| guard | FBR pooled | Block pooled | FBR one per group | Block one per group |
|---|---|---|---|---|
| local_causal | 10.7 ± 6.6 | 51.6 ± 9.5 | 8.7 ± 6.7 | 28.2 ± 10.4 |
| local_full | 10.1 ± 6.5 | 54.0 ± 10.8 | 8.6 ± 6.9 | 35.7 ± 9.7 |
| claude_causal | 9.6 ± 5.4 | 97.5 ± 1.4 | 8.7 ± 6.7 | 78.8 ± 8.6 |
| claude_full | 9.0 ± 4.6 | 98.2 ± 1.0 | 7.8 ± 6.1 | 86.4 ± 7.1 |
| openai_causal | 10.2 ± 6.3 | 91.3 ± 2.7 | 6.5 ± 6.5 | 68.3 ± 6.4 |
| openai_full | 9.8 ± 5.3 | 91.6 ± 2.6 | 8.0 ± 7.0 | 73.1 ± 7.1 |

### Application category as the test domain, alpha = 0.10

Cross-domain: threshold from the other four categories. In-domain: threshold from half of the
category's own records (finite = share of splits with enough safe calibration records).

| guard | category | FBR cross-domain | Block cross-domain | FBR in-domain | Block in-domain | finite |
|---|---|---|---|---|---|---|
| local_causal | Application | 12.1 ± 3.4 | 63.9 ± 3.8 | 7.8 ± 5.3 | 47.0 ± 10.5 | 100.0 ± 0.0 |
| local_causal | Finance | 2.4 ± 1.6 | 49.2 ± 8.1 | 8.9 ± 5.6 | 68.8 ± 8.0 | 100.0 ± 0.0 |
| local_causal | IoT | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| local_causal | Program | 18.0 ± 5.1 | 50.0 ± 6.1 | 9.4 ± 7.5 | 44.0 ± 6.9 | 100.0 ± 0.0 |
| local_causal | Web | 26.7 ± 12.6 | 35.3 ± 11.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| local_full | Application | 6.3 ± 2.4 | 59.3 ± 3.9 | 8.1 ± 5.5 | 62.6 ± 7.4 | 100.0 ± 0.0 |
| local_full | Finance | 14.9 ± 3.9 | 68.9 ± 7.7 | 8.5 ± 5.7 | 49.5 ± 10.6 | 100.0 ± 0.0 |
| local_full | IoT | 0.0 ± 0.0 | 5.2 ± 5.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| local_full | Program | 8.4 ± 3.7 | 51.6 ± 6.1 | 9.7 ± 7.9 | 44.0 ± 14.1 | 100.0 ± 0.0 |
| local_full | Web | 7.1 ± 7.1 | 40.1 ± 11.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| claude_causal | Application | 3.1 ± 1.8 | 100.0 ± 0.0 | 7.2 ± 4.4 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| claude_causal | Finance | 4.6 ± 2.4 | 100.0 ± 0.0 | 7.6 ± 4.6 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| claude_causal | IoT | 36.6 ± 16.7 | 89.6 ± 7.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| claude_causal | Program | 11.7 ± 4.2 | 95.6 ± 2.4 | 9.4 ± 7.2 | 89.6 ± 6.2 | 100.0 ± 0.0 |
| claude_causal | Web | 20.3 ± 11.4 | 94.9 ± 5.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| claude_full | Application | 3.0 ± 1.8 | 100.0 ± 0.0 | 5.6 ± 4.1 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| claude_full | Finance | 4.5 ± 2.3 | 100.0 ± 0.0 | 7.2 ± 4.5 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| claude_full | IoT | 35.8 ± 16.5 | 94.7 ± 5.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| claude_full | Program | 12.0 ± 4.2 | 97.1 ± 2.0 | 9.4 ± 6.8 | 93.9 ± 3.8 | 100.0 ± 0.0 |
| claude_full | Web | 13.6 ± 9.9 | 94.8 ± 5.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| openai_causal | Application | 2.1 ± 1.4 | 96.8 ± 1.4 | 7.3 ± 5.0 | 97.7 ± 1.2 | 100.0 ± 0.0 |
| openai_causal | Finance | 1.2 ± 1.2 | 94.7 ± 3.6 | 6.9 ± 5.0 | 94.7 ± 3.6 | 100.0 ± 0.0 |
| openai_causal | IoT | 27.5 ± 15.7 | 84.0 ± 8.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| openai_causal | Program | 13.6 ± 4.5 | 85.3 ± 4.3 | 9.3 ± 7.6 | 82.0 ± 5.5 | 100.0 ± 0.0 |
| openai_causal | Web | 40.7 ± 14.2 | 70.0 ± 10.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| openai_full | Application | 2.0 ± 1.4 | 96.1 ± 1.6 | 7.3 ± 5.3 | 97.1 ± 1.4 | 100.0 ± 0.0 |
| openai_full | Finance | 5.8 ± 2.5 | 94.7 ± 3.6 | 8.6 ± 5.7 | 94.7 ± 3.6 | 100.0 ± 0.0 |
| openai_full | IoT | 27.2 ± 15.3 | 89.8 ± 7.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| openai_full | Program | 14.9 ± 4.5 | 83.9 ± 4.6 | 8.3 ± 7.3 | 78.7 ± 6.1 | 100.0 ± 0.0 |
| openai_full | Web | 27.1 ± 12.3 | 74.8 ± 10.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
