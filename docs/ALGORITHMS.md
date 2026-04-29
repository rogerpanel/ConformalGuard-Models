# Algorithms

The pseudocode in this document mirrors the manuscript Sec. 4. Each
algorithm is a faithful reflection of the corresponding code in the
`conformalguard/` package.

## Algorithm 1 — Split-conformal calibration

```text
Input:  trained encoder h_θ, head g_θ, calibration set D_cal,
        target miscoverage α ∈ (0, 1), nonconformity score s.
Output: threshold τ.

1.  for (G_<t^{(i)}, y^{(i)}) ∈ D_cal:
2.      h^{(i)} ← h_θ(G_<t^{(i)})
3.      logits  ← g_θ(h^{(i)})
4.      s_i     ← s(logits, y^{(i)})
5.  q ← ⌈(n + 1)(1 − α)⌉ / n
6.  τ ← Quantile_q(s_1, …, s_n)
7.  return τ
```

Implementation: `conformalguard.conformal.calibration.ConformalCalibrator.calibrate`.

## Algorithm 2 — Inference (offline / per request)

```text
Input:  trained (h_θ, g_θ), threshold τ, current subgraph G_<t.
Output: prediction set C_α(G_<t) ⊆ V.

1.  h     ← h_θ(G_<t)
2.  logits ← g_θ(h)
3.  scores ← {s(logits, y) for y ∈ V}
4.  return {y ∈ V : scores[y] ≤ τ}
```

Implementation: `conformalguard.conformal.calibration.ConformalCalibrator.predict_set`.

## Algorithm 3 — Online adaptive (Gibbs-Candès)

```text
Input:  base τ from Alg. 1, target α*, learning rate γ.
State:  α_t.

Initialise α_0 ← α*.
On stream of (G_<t, y_t):
    τ_eff   ← τ + (α* − α_t) · |τ|     # linear surrogate
    C       ← {y : s(g_θ(h_θ(G_<t)), y) ≤ τ_eff}
    err_t   ← 1{y_t ∉ C}
    α_{t+1} ← clip(α_t + γ (err_t − α*), [α_min, α_max])
    yield C
```

Implementation: `conformalguard.adaptive.gibbs_candes.GibbsCandesController`.

## Algorithm 4 — Verdict / routing

```text
Input:  prediction set C, proposed action y*, severity sev,
        set width w, drift δ.
Output: verdict ∈ {ALLOW, BLOCK, REVIEW},
        routing  ∈ {AUTO_BLOCK, ESCALATE_ONCALL, REVIEW_QUEUE, ALLOW}.

1.  if y* ∈ C: return (ALLOW, ALLOW)
2.  if min_y s(., y) < τ: verdict ← BLOCK
3.  else:                  verdict ← REVIEW
4.  priority ← 0.6·sev + 0.2·norm(w) + 0.2·min(δ, 1)
5.  if verdict = BLOCK and priority ≥ 0.9:  routing ← AUTO_BLOCK
6.  elif priority ≥ 0.7:                     routing ← ESCALATE_ONCALL
7.  elif verdict = REVIEW:                   routing ← REVIEW_QUEUE
8.  else:                                    routing ← ALLOW
9.  return (verdict, routing)
```

Implementation: `conformalguard.deployment.routing.AnalystRouter.route`.

## Algorithm 5 — Drift monitor (Kolmogorov-Smirnov)

```text
Input:  rolling window W of recent nonconformity scores,
        calibration scores S_cal, drift level α_d.
Output: triggered ∈ {0, 1}.

1.  d ← KS-statistic(S_cal, W)
2.  p ← asymptotic_p_value(d, |S_cal|, |W|)
3.  triggered ← p < α_d
```

Implementation: `conformalguard.adaptive.drift_monitor.DriftMonitor`.
