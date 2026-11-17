# Conformal calibration of guard scores on R-Judge

1000 random halves (calibration / test) per protocol. FBR and Block in %, mean ± sd. Target FBR = alpha.

| guard | view | records | no-score rows | AUROC | own operating point: FBR / Block |
|---|---|---|---|---|---|
| qwen25_causal (Qwen/Qwen2.5-7B-Instruct) | causal | 571 | 0 | 0.780 | 85.6 / 93.0 |
| new_local_causal (Qwen/Qwen3-8B) | causal | 571 | 0 | 0.767 | 46.3 / 81.4 |
| deepseek_causal (deepseek/deepseek-v3.2) | causal | 571 | 2 | 0.943 | 22.2 / 93.0 |

### Random halves (exchangeable: the guarantee applies)

| guard | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |
|---|---|---|---|---|---|---|
| qwen25_causal | 4.3 ± 2.5 | 41.1 ± 6.0 | 9.4 ± 3.4 | 51.3 ± 4.2 | 19.7 ± 4.7 | 66.0 ± 3.9 |
| new_local_causal | 4.4 ± 2.4 | 23.1 ± 5.5 | 9.5 ± 3.6 | 38.5 ± 6.8 | 19.6 ± 4.8 | 57.4 ± 4.4 |
| deepseek_causal | 3.6 ± 2.2 | 72.8 ± 9.7 | 8.2 ± 3.6 | 83.5 ± 2.9 | 18.4 ± 4.7 | 92.9 ± 1.6 |

### Prompt-grouped halves, alpha = 0.10

| guard | FBR pooled | Block pooled | FBR one per group | Block one per group |
|---|---|---|---|---|
| qwen25_causal | 11.0 ± 7.0 | 51.4 ± 9.4 | 8.6 ± 7.0 | 27.7 ± 11.3 |
| new_local_causal | 11.2 ± 7.8 | 39.6 ± 11.4 | 8.8 ± 6.8 | 33.2 ± 10.1 |
| deepseek_causal | 8.7 ± 4.8 | 83.9 ± 4.6 | 7.7 ± 6.9 | 69.1 ± 11.5 |

### Application category as the test domain, alpha = 0.10

Cross-domain: threshold from the other four categories. In-domain: threshold from half of the
category's own records (finite = share of splits with enough safe calibration records).

| guard | category | FBR cross-domain | Block cross-domain | FBR in-domain | Block in-domain | finite |
|---|---|---|---|---|---|---|
| qwen25_causal | Application | 12.4 ± 3.5 | 63.8 ± 3.8 | 8.5 ± 5.8 | 48.3 ± 10.8 | 100.0 ± 0.0 |
| qwen25_causal | Finance | 2.4 ± 1.6 | 49.2 ± 8.5 | 8.9 ± 5.7 | 68.8 ± 7.9 | 100.0 ± 0.0 |
| qwen25_causal | IoT | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| qwen25_causal | Program | 18.4 ± 5.0 | 50.1 ± 5.9 | 9.7 ± 7.2 | 44.3 ± 6.6 | 100.0 ± 0.0 |
| qwen25_causal | Web | 27.2 ± 12.6 | 34.9 ± 10.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| new_local_causal | Application | 9.4 ± 2.9 | 41.3 ± 3.8 | 8.2 ± 5.3 | 41.4 ± 8.0 | 100.0 ± 0.0 |
| new_local_causal | Finance | 1.2 ± 1.2 | 25.5 ± 7.2 | 9.2 ± 6.0 | 67.2 ± 10.5 | 100.0 ± 0.0 |
| new_local_causal | IoT | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| new_local_causal | Program | 26.3 ± 5.7 | 54.2 ± 6.1 | 9.6 ± 7.2 | 20.3 ± 6.9 | 100.0 ± 0.0 |
| new_local_causal | Web | 6.7 ± 7.1 | 25.0 ± 9.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| deepseek_causal | Application | 11.3 ± 3.3 | 92.3 ± 2.2 | 7.2 ± 5.3 | 87.0 ± 5.4 | 100.0 ± 0.0 |
| deepseek_causal | Finance | 3.3 ± 1.9 | 81.9 ± 6.2 | 7.5 ± 5.5 | 86.9 ± 6.9 | 100.0 ± 0.0 |
| deepseek_causal | IoT | 0.0 ± 0.0 | 42.2 ± 12.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| deepseek_causal | Program | 14.9 ± 4.8 | 77.6 ± 5.0 | 7.7 ± 7.7 | 73.4 ± 6.8 | 100.0 ± 0.0 |
| deepseek_causal | Web | 0.0 ± 0.0 | 75.2 ± 10.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
