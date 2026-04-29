# ConformalGuard

**Distribution-Free Safety Certification for Multi-Agent LLM Systems via Conformal Prediction on Dynamic Execution Graphs**

This repository is the official reference implementation accompanying the manuscript:

> R. N. Anaedevha and A. G. Trofimov,
> *ConformalGuard: Distribution-Free Safety Certification for Multi-Agent LLM Systems via Conformal Prediction on Dynamic Execution Graphs*,
> ICIS, MEPhI Moscow, 2026.

It is part of the [RobustIDPS.ai](https://robustidps.ai) platform research line. The companion manuscript
and documentation are released at
[github.com/rogerpanel/ConformalGuard-Models](https://github.com/rogerpanel/ConformalGuard-Models).

The codebase is organized so that all results in the paper can be regenerated end-to-end by running
`python scripts/reproduce_results.py` after installing the package and downloading
datasets via `python scripts/download_datasets.py`.

---

## What ConformalGuard does

ConformalGuard treats a multi-agent LLM execution as a **heterogeneous continuous-time dynamic graph**
(four node types: agent invocations, tool calls, memory entries, inter-agent messages; six edge types:
`invokes`, `reads`, `writes`, `delegates`, `replies`, `cites`) and produces, at every step `t`, a
*prediction set* of safe continuations with **distribution-free finite-sample coverage**:

```
P[ y_t ∈ C_α(G_<t) ] ≥ 1 − α            (Theorem 1: marginal coverage)
```

When deployed online, the **Gibbs–Candès** update keeps long-run empirical miscoverage at the target
level even under distribution drift (Theorem 2). Whenever a candidate continuation falls *outside*
`C_α(G_<t)`, ConformalGuard either (i) blocks it, (ii) routes the trace to a SOC analyst review queue,
or (iii) hands it to an RL active-investigation policy. This mechanism is integrated as a dedicated
**agent-layer safety module** in RobustIDPS.ai.

## Key results reproduced

| Metric                                  | ConformalGuard | Best baseline      |
|-----------------------------------------|---------------:|-------------------:|
| Empirical coverage @ α = 0.05           | 94.9 %         | n/a                |
| Critical-violation block rate           | **96.1 %**     | 31.2 % (LlamaGuard-2) |
| False-block rate on benign traces       | 3.4 %          | 14.7 %             |
| Average prediction-set width            | 1.8            | n/a                |
| Per-step latency (A100)                 | 23 ms          | 18 ms              |
| Production: precision / recall (30 d)   | 97.1 % / 94.6 % | n/a               |

## Layout

```
ConformalGuard/
├── conformalguard/
│   ├── graph/          Heterogeneous CT-DG, real-time builder, framework hooks
│   ├── encoder/        DyGFormer-HGT 3-layer relation-aware temporal encoder
│   ├── conformal/      Nonconformity scores, calibration, prediction sets
│   ├── adaptive/       Gibbs–Candès online coverage, drift monitor
│   ├── violations/     11 critical violation classes (PII, prompt injection, …)
│   ├── data/           AgentChain-26 loader + synthetic generator + external refs
│   ├── training/       Trainer, losses, schedulers
│   ├── evaluation/     Metrics + 7 baselines (LlamaGuard-2, WildGuard, …)
│   ├── deployment/     RobustIDPS integration, analyst routing, FastAPI server
│   └── utils/
├── configs/            YAML configs for every experiment in the paper
├── scripts/            train, calibrate, evaluate, reproduce, download_datasets
├── tests/              Unit tests covering coverage, exchangeability, hooks
├── examples/           AutoGen / MetaGPT / LangGraph / CrewAI integrations
├── docs/               ALGORITHMS, DATASETS, HYPERPARAMETERS, METHODOLOGY, REPRODUCIBILITY
└── notebooks/          Stand-alone Python scripts that mirror the paper figures
```

## Quickstart

```bash
git clone https://github.com/rogerpanel/CV.git
cd CV/ConformalGuard
pip install -e ".[dev]"

# 1. Materialize AgentChain-26 (downloads or generates the synthetic version)
python scripts/download_datasets.py

# 2. Train the DyGFormer-HGT encoder
python scripts/train.py --config configs/encoder.yaml

# 3. Calibrate the conformal head on n_cal = 2000 traces
python scripts/calibrate.py --config configs/conformal.yaml

# 4. Evaluate vs. baselines and reproduce paper Table 2
python scripts/evaluate.py --config configs/default.yaml

# 5. One-shot full reproduction
python scripts/reproduce_results.py
```

## Citing

```bibtex
@article{anaedevha2026conformalguard,
  title   = {ConformalGuard: Distribution-Free Safety Certification for
             Multi-Agent LLM Systems via Conformal Prediction on Dynamic
             Execution Graphs},
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich},
  journal = {Preprint},
  year    = {2026},
  url     = {https://github.com/rogerpanel/CV/tree/main/ConformalGuard}
}
```

## Datasets

| Dataset            | Purpose                              | Source / DOI                                                          |
|--------------------|--------------------------------------|-----------------------------------------------------------------------|
| AgentChain-26      | Primary benchmark (12,400 traces)    | This repo — `data/agentchain26/` (synthetic generator + manifest)     |
| ToolEmu            | Out-of-distribution agent traces     | https://toolemu.com / https://github.com/ryoungj/ToolEmu              |
| AgentBench         | Cross-platform task suite            | https://github.com/THUDM/AgentBench                                   |
| TrustLLM           | Safety alignment evaluations         | https://github.com/HowieHwong/TrustLLM                                |
| InjecAgent         | Indirect prompt injection corpus     | https://github.com/uiuc-kang-lab/InjecAgent                           |
| HarmBench          | Red-team prompts                     | https://github.com/centerforaisafety/HarmBench                        |
| AdvBench           | Universal harmful prompt suite       | https://github.com/llm-attacks/llm-attacks                            |
| BeaverTails        | Safety preference data               | https://huggingface.co/datasets/PKU-Alignment/BeaverTails             |
| RobustIDPS PCAPs   | Network co-evaluation                | https://robustidps.ai (DOI 10.5281/zenodo.19129512)                   |
| PQC Traffic        | Cross-stack agent traces             | https://doi.org/10.34740/kaggle/dsv/15424420                          |

See [docs/DATASETS.md](docs/DATASETS.md) for full preparation details.

## License

MIT — see [LICENSE](LICENSE).
