# ConformalGuard: supplementary code and results

Everything in the paper can be regenerated from this package.

| Paper | Command | Notes |
|---|---|---|
| Section 6.1, Table 1 (trained scorers), Appendix D (R-Judge) | `python scripts/rjudge_run.py --rjudge-dir R-Judge --features st --reps 100 --loco-reps 10 --out runs/rjudge_st_minilm` | Needs `git clone https://github.com/Lordog/R-Judge` and `pip install sentence-transformers`. About 2 hours on two T4 GPUs. Add `--st-model BAAI/bge-base-en-v1.5` for the second encoder. |
| The results we report | `results/rjudge_kaggle_t4/{minilm,bge}/results.json` | Every per-split number, the aggregates, the paired tests and the full configuration. `summary.md` next to each file has the tables. |
| A paired comparison from a results file | `python scripts/rjudge_paired.py results/rjudge_kaggle_t4/minilm/results.json seq step block@0.10 random` | |
| Section 6.3, Figure 2, Appendix G (simulations) | `python scripts/sim_checks.py --figure fig2_aci_simulation.pdf` | Synthetic scores, fixed seeds, CPU, about 3 minutes. |
| The guarantees (Propositions 1 and 2, Corollary 1, the two-sided guard) | `pip install numpy torch scikit-learn pytest && python -m pytest -q` | Monte-Carlo checks of each bound; Proposition 2 asserted at every horizon on shifted, adversarial and delayed-feedback sequences; a test that no scorer sees anything after the action it scores. |

## The experiments added in the revision

| Paper | Command | Results in this package |
|---|---|---|
| Table 1 (judge rows), Appendix E: LLM judges calibrated as guards | `scripts/rjudge_judge_local.py`, `rjudge_judge_claude.py`, `rjudge_judge_openai.py` (one score file each), then `scripts/rjudge_guard_eval.py --scores name=path ...` | `results/kaggle_extra/guard_eval/`, raw scores in `guard_scores/` |
| Section 6.1 and Appendix D: Proposition 1 on step labels | `scripts/rjudge_steps.py --rjudge-dir R-Judge --injecagent-dir InjecAgent --features st` | `results/kaggle_extra/rjudge_steps/` |
| Section 6.1 and Appendix D: in-domain recalibration, per-category calibration | `scripts/rjudge_run.py --protocols random,recal --features st` | `results/kaggle_extra/rjudge_recal_minilm/` |
| Table 2, Section 6.2, Appendix F: multi-agent sessions | `scripts/mas_generate_claude.py`, `scripts/mas_generate_openai.py`, then `scripts/mas_run.py --sessions ... --features st`; `scripts/mas_rule_baseline.py` for the hand-written rule | `results/mas_kaggle_v2/` (sessions, per-split results, rule baseline) |

The judge and generation scripts call commercial APIs and need keys; their outputs are included, so every table can be
recomputed from the files in `results/` without calling a model. `notebooks/rjudge_kaggle_extra.ipynb` and
`notebooks/mas_rerun_kaggle.ipynb` are the notebooks the runs were made with. `docs/MAS_EXPERIMENT.md` describes the
simulator, and `results/*/README.md` state the limits that belong with the numbers.

A CPU-only variant of the R-Judge experiment with TF-IDF features
(`--features lsa`, no model download) is in `results/rjudge_lsa_cpu/`.

## Layout

```
conformalguard/conformal/core.py        split-conformal threshold, calibration units, two-sided guard
conformalguard/adaptive/gibbs_candes.py adaptive conformal inference with delayed labels (no clipping)
conformalguard/rjudge/                  R-Judge converter, features, four scorers, protocols, judges, step labels
conformalguard/mas/                     multi-agent simulator, code-derived labels, graph conversion, evaluation
conformalguard/encoder/                 relation-aware attention layer used by the graph scorer
conformalguard/graph/                   execution-graph schema, builder, framework hooks (hooks not exercised)
scripts/                                rjudge_run.py, rjudge_paired.py, rjudge_llm_judge.py, sim_checks.py, make_fig2.py
tests/                                  unit tests
docs/RJUDGE_EXPERIMENT.md               protocol and limits of the R-Judge experiment
results/                                result files
```

`scripts/rjudge_llm_judge.py` runs R-Judge's own two-stage judging protocol with a local chat model.
It is not used for any number in the paper; its prompt construction and parsing are unit-tested, its
model-loading path is not.

R-Judge is not redistributed here. We used commit `83ce301da3` of its repository.
