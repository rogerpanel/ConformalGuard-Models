# R-Judge experiment

A real-data experiment for the conformal guard. Everything here runs on public
data with labels produced by other people, and every number it reports comes
out of `scripts/rjudge_run.py`.

## What it tests

| Question | How it is tested |
|---|---|
| Does the false-block guarantee hold on real agent records? | Over many random splits, the share of *safe* test records with a blocked step must average at most alpha, for every scorer. |
| How many unsafe records does a guard block at that budget? | Block rate on the unsafe test records at alpha = 0.05, 0.10, 0.20. Empirical; depends on the scorer. |
| Does a graph help beyond history? | Same features, same loss, same calibration, close parameter counts: a causal Transformer over the node sequence (`seq`) against relation-aware attention over the typed execution graph (`graph`), with paired per-split differences and a sign test. |
| Do the types matter, or only the topology? | `graph_untyped`: same graph and layers, one node type and one relation. |
| Does history matter at all? | `step`: the current step only. |
| How far do the results depend on templates and on the application domain? | Prompt-grouped splits; leave-one-category-out. |
| How does a full-history LLM judge compare? | The per-record predictions released by the R-Judge authors (Llama-3.1-8B), and optionally a judge you run with `scripts/rjudge_llm_judge.py`. |

## Data

[R-Judge](https://github.com/Lordog/R-Judge) (Yuan et al., Findings of EMNLP
2024): 571 records in the repository at the commit recorded in `results.json`,
301 labelled unsafe and 270 safe by its annotators, in five application
categories and two attack types (injection, unintended). The data are not
copied into this repository.

Each record is converted to an execution graph (`conformalguard/rjudge/data.py`):
agent invocations, tool calls, memory entries (environment returns) and
messages, linked by `invokes`, `writes`, `reads`, `replies` and `next`. A
decision step is one agent message; the proposed action is its tool call or
its reply.

## Protocol

For each split (50% train, 25% calibration, 25% test):

1. each scorer is trained on the training records only;
2. a record's score is its maximum step score;
3. the threshold is the ceil((n+1)(1-alpha))-th smallest score of the n safe
   calibration records;
4. a test record is blocked if its score exceeds the threshold.

Step 3 is the trace-level rule (Corollary 1 of the paper). R-Judge labels
records, not steps, so the step-level statement (Proposition 1) cannot be
tested on it.

## Limits of this experiment

State these whenever the numbers are quoted.

- **Single agent.** R-Judge records have one agent with tools. The graphs are
  small (at most 37 nodes) and contain no delegation between agents, so this
  is not a test of multi-agent composition.
- **In-domain training.** Scorers are trained and tested on R-Judge by
  cross-fitting. This is not transfer from another benchmark.
- **Record-level labels.** Nothing is known about which step is the unsafe
  one, so "blocked at or before the violating action" cannot be measured.
- **A dataset artefact.** Unsafe records usually end on the risky tool call
  and safe records on a final answer. The rule "the record ends on a tool
  call" separates the labels well. A runtime guard cannot use it, because it
  does not know which step is the last, and all four neural scorers are
  causal (unit-tested). The pooled logistic-regression baseline and the rule
  itself do see the end of the record and are reported for that reason.
- **Templates.** Only 151 distinct user prompts occur in 571 records. Random
  splits share prompts between training and testing. The grouped splits do
  not, and are the better guide to behaviour on unseen tasks.
- **Small test parts.** About 68 safe and 75 unsafe test records per split;
  the Web and IoT categories have 35 and 30 records in total.
- **No online adaptation.** The record stream is too short for adaptive
  conformal inference to be informative (the Proposition 2 band at T = 100 is
  20 points wide).

## Running it

On a laptop CPU, no downloads beyond R-Judge (TF-IDF + SVD node features):

```bash
git clone https://github.com/Lordog/R-Judge
pip install numpy torch scikit-learn
python scripts/rjudge_run.py --rjudge-dir R-Judge --features lsa --reps 40 --out runs/rjudge_lsa
```

On a GPU, with a frozen sentence encoder for node features: open
`notebooks/rjudge_colab.ipynb` in Colab, or

```bash
pip install sentence-transformers scikit-learn
python scripts/rjudge_run.py --rjudge-dir R-Judge --features st --reps 100 --loco-reps 10 --out runs/rjudge_st
python scripts/rjudge_llm_judge.py --rjudge-dir R-Judge --model Qwen/Qwen2.5-7B-Instruct --out runs/rjudge_judge
```

`--quick` runs three splits with six epochs to check the installation; its
output is marked as a smoke run and is not a result.

Outputs: `results.json` (every per-split number, aggregates, paired tests,
configuration, code and data commits) and `summary.md` (the tables).

`scripts/rjudge_llm_judge.py` has unit tests for its prompt construction and
verdict parsing; its model-loading path needs a GPU and the Hugging Face hub
and has not been executed in the environment where this code was written.

## Additional experiments

Run together by `notebooks/rjudge_kaggle_extra.ipynb` (which also runs the multi-agent experiment of
`docs/MAS_EXPERIMENT.md`), or one at a time:

| Experiment | Command | What it tests |
|---|---|---|
| Calibrated LLM guards | `scripts/rjudge_judge_local.py`, `rjudge_judge_claude.py`, `rjudge_judge_openai.py`, then `scripts/rjudge_guard_eval.py --scores name=path ...` | Whether the conformal layer gives an existing guard a false-block budget. Each judge writes a graded risk score per record (`--mode full`) or per decision step from the prefix only (`--mode causal`); the evaluator calibrates any score file without training. |
| Step-level guarantee | `scripts/rjudge_steps.py --injecagent-dir InjecAgent` | Proposition 1 on real records: step labels for the 414 injection records are derived by code from InjecAgent's attacker-tool list; calibration uses one safe step per trace. |
| In-domain recalibration | `scripts/rjudge_run.py --protocols random,recal` | Whether calibrating on a few records of the new application category restores the guarantee that a held-out category breaks, and what one threshold per category does on mixed data. |

Results of a full run of all three are in `results/kaggle_extra/` (see its README).

Limits to state with their results:

- **The released Llama judge cannot be calibrated.** Its verdicts are binary and flag 85% of safe
  records; as a score they have AUROC 0.50, so any threshold either blocks nearly everything or
  nothing. A guard needs a graded score to benefit from calibration.
- **The step-level task is nearly trivial for detection.** In the injection records a safe second
  step is always a final answer and an unsafe one a tool call, so step scorers separate them almost
  perfectly. The experiment tests the calibration rule on real step labels; it says little about
  detection.
- **API judges.** Ratings are integers from 0 to 100, so thresholds have ties; the conformal rule is
  then conservative (false blocks at or below the target). A refused or unparseable reply counts as
  the highest risk. The Claude script enables server-side refusal fallbacks and records which model
  served each row: in the archived run about a quarter of the causal rows and a third of the
  full-record rows were served by the fallback model, so that guard is "Opus 5.5 with fallback to
  Opus 4.8". R-Judge is public, so these models may have seen its records in training; the
  false-block guarantee does not depend on that, the block rates might.
- **Small categories.** IoT and Web have 11 and 15 safe records; half of them is too few for a
  finite threshold at small alpha, and the output reports how often that happens.
