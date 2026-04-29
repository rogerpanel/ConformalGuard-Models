# Methodology

ConformalGuard combines three established techniques into a single
distribution-free safety certificate for multi-agent LLM systems:

1. **Heterogeneous continuous-time dynamic graph (CT-DG)** representation of an
   agent execution.
2. **DyGFormer-HGT** encoder producing per-step subgraph embeddings.
3. **Split-conformal prediction** with **online (Gibbs-Candès) adaptation** on
   top of those embeddings.

This document is a self-contained companion to Sec. 3 and Sec. 4 of the
manuscript.

## 1. Execution-graph formalization

A multi-agent execution is modeled as a directed, heterogeneous, continuous-
time dynamic graph

```
G_t = (V_t, E_t, X_t, T_t, R_t)
```

with **four node types**:

| Symbol | Type             | Examples                                              |
|--------|------------------|-------------------------------------------------------|
| `a`    | agent_invocation | `Coder`, `Planner`, `Critic`, `Coordinator`           |
| `o`    | tool_call        | shell, search, vector-store query, HTTP request       |
| `m`    | memory_entry     | scratchpad, long-term memory, RAG chunk               |
| `μ`    | message          | inter-agent textual exchange                          |

and **six edge types**:

| Edge       | (src, dst)                                  | Semantic                |
|------------|---------------------------------------------|-------------------------|
| `invokes`  | (agent, tool)                               | agent calls a tool      |
| `delegates`| (agent, agent)                              | A asks B to do a sub-task |
| `reads`    | (agent, memory)                             | agent retrieves entry   |
| `writes`   | (agent or tool, memory)                     | new entry persisted     |
| `replies`  | (agent, message) and (message, message)     | reply tree              |
| `cites`    | (agent, memory)                             | claim is grounded by … |

Causality: every edge has a timestamp, and the encoder only attends to nodes
and edges with `t_e ≤ t`.

## 2. Encoder

The encoder is a 3-layer interleaving of HGT (Hu et al., 2020) and DyGFormer
(Yu et al., 2023):

```
h^{0}_v = MLP(feat(v)) + emb(type(v)) + φ(t − t_v)
for ℓ = 1..3:
    h^{ℓ}   = HGT(h^{ℓ-1}, edge_index, edge_type, node_type)
    h^{ℓ}   = h^{ℓ} + TemporalAttn(h^{ℓ}, h^{ℓ}, h^{ℓ}, dt)
h_t       = LayerNorm(GRU(mean(h^{3}), latest_agent_emb))
```

with `d_model = 256`, `n_heads = 8`, time-encoding dim = 64, dropout = 0.1.

## 3. Calibration (Theorem 1)

Given calibration set `D_cal = {(G_<t^{(i)}, y^{(i)})}_{i=1..n}` and target
miscoverage α, we compute scores `s_i = − log p_θ(y^{(i)} | h_θ(G_<t^{(i)}))`
and set the threshold

```
τ = Quantile_{⌈(n+1)(1−α)⌉ / n}(s_1, …, s_n).
```

For any new test point exchangeable with `D_cal`,

```
1 − α ≤ P[ y_t ∈ C_α(G_<t) ] ≤ 1 − α + 1/(n+1),
C_α(G_<t) := { y ∈ V : s(h_θ(G_<t), y) ≤ τ }.
```

## 4. Online adaptation (Theorem 2)

In deployment we replace α by an online estimate

```
α_{t+1} = α_t + γ (1{y_t ∉ C_{α_t}} − α),
```

where γ = 0.05. Theorem 2 (Gibbs & Candès, 2021) proves that
`(1/T) Σ_{t≤T} 1{err_t} → α` regardless of distribution shift.

In code this lives in `conformalguard.adaptive.gibbs_candes`.

## 5. Verdict mapping

The deployment server (Sec. 6) translates a prediction set into one of three
verdicts:

| Condition                                            | Verdict |
|------------------------------------------------------|---------|
| Proposed action ∈ `C_α(G_<t)`                        | ALLOW   |
| Proposed action ∉ `C_α`, `min_y s(.,y) < τ`          | BLOCK   |
| Otherwise                                            | REVIEW  |

REVIEW items are pushed to the SOC analyst queue inside RobustIDPS.ai with
priority weights `(0.6, 0.2, 0.2)` over (severity, set-width, drift).
