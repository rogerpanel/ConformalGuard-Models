# Hyperparameters

All values match the manuscript (Sec. 5) and are reproduced verbatim by the
default YAML configs under `configs/`.

## Encoder

| Symbol         | Value | Where                                            |
|----------------|------:|--------------------------------------------------|
| `d_model`      | 256   | `EncoderConfig.d_model`                          |
| `n_heads`      | 8     | `EncoderConfig.n_heads`                          |
| `n_layers`     | 3     | `EncoderConfig.n_layers`                         |
| `dropout`      | 0.10  | `EncoderConfig.dropout`                          |
| `time_dim`     | 64    | `EncoderConfig.time_dim`                         |
| `feature_dim`  | 128   | `EncoderConfig.feature_dim`                      |
| `pool`         | `mean+gru` | `EncoderConfig.pool`                        |
| `n_node_types` | 4     | derived from `NodeType`                          |
| `n_edge_types` | 6     | derived from `EdgeType`                          |

## Optimizer / schedule

| Hyper-parameter | Value             |
|-----------------|-------------------|
| Optimizer       | AdamW             |
| Learning rate   | 5 × 10⁻⁴          |
| Weight decay    | 0.01              |
| Epochs          | 30                |
| Batch size      | 64 traces         |
| Grad clip       | 1.0 (ℓ₂ norm)     |
| LR schedule     | cosine annealing  |
| Coverage λ      | 0.10              |
| Mixed precision | optional, fp16    |

## Calibration

| Symbol      | Value                              |
|-------------|------------------------------------|
| `n_cal`     | 2000 traces (held-out split)       |
| `α`         | 0.05  ⇒  95 % marginal coverage    |
| Score `s`   | `− log p_θ(y | h_θ(G_<t))`         |
| Quantile    | `⌈(n+1)(1−α)⌉ / n` (finite-sample) |

## Online (Gibbs-Candès)

| Symbol         | Value                       |
|----------------|-----------------------------|
| Target α       | 0.05                        |
| Learning rate γ| 0.05                        |
| α clip         | [0.001, 0.5]                |

## Drift monitor

| Symbol           | Value |
|------------------|------:|
| Window size      | 512   |
| KS p-value α     | 0.01  |

## Routing

| Symbol                | Value |
|-----------------------|------:|
| `w_severity`          | 0.6   |
| `w_width`             | 0.2   |
| `w_drift`             | 0.2   |
| Escalate priority τ   | 0.7   |
| Auto-block priority τ | 0.9   |

## Hardware

The paper trains on **4 × NVIDIA A100 80 GB**; per-step inference latency is
**23 ms on a single A100**. Single-GPU and CPU runs are supported (CPU is
≈ 10× slower at inference).
