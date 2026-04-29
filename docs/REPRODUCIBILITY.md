# Reproducibility

This document is the canonical recipe to reproduce every numerical claim in
the paper from the source contained in this folder.

## 0. Environment

```bash
python -V             # 3.10 +
pip install -e ".[dev]"
# Optional, for live-instrumentation tests:
pip install -e ".[agents]"
```

GPU is recommended but not required. CPU runs all unit tests in well under
a minute.

## 1. Materialize datasets

```bash
python scripts/download_datasets.py
```

This generates a deterministic copy of AgentChain-26 (12,400 traces) under
`~/.cache/conformalguard/agentchain26/`. To use the *curated* release, set
`CONFORMALGUARD_DATA=/path/to/agentchain26` and place the train/val/test
JSONL manifests inside that directory.

## 2. Train the encoder

```bash
python scripts/train.py --config configs/encoder.yaml
```

Outputs:

```
runs/conformalguard/
    epoch_00.pt … epoch_29.pt
    model.pt              # final checkpoint
    history.json          # per-epoch loss / acc
```

## 3. Calibrate the conformal head

```bash
python scripts/calibrate.py --config configs/conformal.yaml
```

Outputs `runs/conformalguard/threshold.json` with `{alpha, threshold, n}`.

## 4. Evaluate against baselines

```bash
python scripts/evaluate.py --config configs/default.yaml
```

This reproduces Table 2 of the paper. The 7 baselines are listed in
`conformalguard/evaluation/baselines.py`. When weights or API keys are
absent, baselines fall back to deterministic stubs so the pipeline still
runs end-to-end on a fresh laptop. To use the real baselines:

```
export CG_LLAMAGUARD2=1
export CG_WILDGUARD=1
export OPENAI_API_KEY=sk-…
```

## 5. One-shot reproduction

```bash
python scripts/reproduce_results.py
```

Runs steps 1–4 in order with identical seeds and writes
`runs/conformalguard/report.json`.

## 6. Streaming / online evaluation (Sec. 5.4)

```bash
python scripts/run_inference.py --config configs/adaptive.yaml
```

This evaluates the Gibbs-Candès controller under four shift regimes
(`stationary, mild_drift, abrupt_shift, recovery`) and writes
`runs/conformalguard/online.json`.

## 7. RobustIDPS deployment

Local dev:

```bash
export CONFORMALGUARD_CKPT=runs/conformalguard/model.pt
uvicorn conformalguard.deployment.server:app --port 8088
```

The endpoint is registered as the *agent-layer safety module* of
RobustIDPS.ai. See `examples/` for end-to-end integration scripts using
AutoGen, MetaGPT, LangGraph, and CrewAI.

## 8. Determinism

All scripts seed Python, NumPy, and PyTorch through `set_seed`. CUDA cuDNN
non-determinism is disabled when the seed is set. With `seed = 1234` you
should reproduce the headline numbers within ±0.3 % on identical hardware
and ±1.0 % across hardware.
