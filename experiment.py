"""Evaluation protocol for the R-Judge experiment.

For each random split of the records into train / calibration / test:

1. fit each scorer on the training records (both classes; an internal 15%
   validation part of the training records selects the epoch);
2. score every calibration and test record: the record score is the maximum
   step logit;
3. set the blocking threshold to the ceil((n+1)(1-alpha))-th smallest score of
   the n *safe* calibration records (the trace-level rule of Corollary 1);
4. on the test records report
     FBR    share of safe records with at least one blocked step
            (guaranteed <= alpha on average over splits, for any scorer),
     Block  share of unsafe records with at least one blocked step
            (empirical; depends on the scorer),
     AUROC  of the record score.

Split protocols
  random   records are permuted uniformly, stratified by label.  Calibration
           and test records are exchangeable, so the FBR guarantee applies.
  grouped  records that share a user prompt (R-Judge reuses prompts heavily)
           always fall in the same part.  Harder: no template is seen in
           training and testing.  Groups, not records, are exchangeable, so
           the guarantee is not exact here.
  loco     leave one application category out: train and calibrate on four
           categories, test on the fifth.  A real distribution shift; no
           guarantee applies and the FBR is reported to show what happens.
  recal    the repair for that shift: train on four categories, then calibrate
           on half of the fifth category's records and test on the other half.
           Calibration and test records are exchangeable again, so the
           guarantee applies, with as few calibration records as the category has.

On random splits the evaluation also reports category-conditional ("Mondrian")
calibration: one threshold per application category, from that category's safe
calibration records, which bounds the false-block rate within each category.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from conformalguard.conformal.core import conformal_quantile, lower_conformal_quantile
from conformalguard.rjudge import models as M
from conformalguard.rjudge.data import TOOL, Trace

NEURAL = ("step", "seq", "graph", "graph_untyped")
BASELINES = ("logreg_pooled", "rule_last_tool")
ALPHAS = (0.05, 0.10, 0.20)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def split_random(traces, rng, fracs=(0.5, 0.25, 0.25)):
    y = np.array([t.label for t in traces])
    parts = ([], [], [])
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        a, b = int(round(fracs[0] * idx.size)), int(round((fracs[0] + fracs[1]) * idx.size))
        for p, chunk in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            p.extend(chunk.tolist())
    return tuple(np.array(sorted(p)) for p in parts)


def split_grouped(traces, rng, fracs=(0.5, 0.25, 0.25)):
    """Whole prompt groups go to one part; parts are filled towards their target sizes."""
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(traces):
        groups.setdefault(t.group, []).append(i)
    keys = list(groups)
    order = rng.permutation(len(keys))
    target = np.array(fracs) * len(traces)
    parts, size = ([], [], []), np.zeros(3)
    for k in order:
        members = groups[keys[k]]
        j = int(np.argmax(target - size))            # the part furthest below its target
        parts[j].extend(members)
        size[j] += len(members)
    return tuple(np.array(sorted(p)) for p in parts)


def split_loco(traces, rng, held_out: str, cal_frac=1 / 3):
    inside = np.array([i for i, t in enumerate(traces) if t.category != held_out])
    test = np.array([i for i, t in enumerate(traces) if t.category == held_out])
    sub = [traces[i] for i in inside]
    tr, cal, _ = split_random(sub, rng, fracs=(1 - cal_frac, cal_frac, 0.0))
    return inside[tr], inside[cal], test


def split_recal(traces, rng, held_out: str):
    """Train on the other categories; calibrate and test on halves of the held-out one."""
    train = np.array([i for i, t in enumerate(traces) if t.category != held_out])
    inside = np.array([i for i, t in enumerate(traces) if t.category == held_out])
    _, cal, te = split_random([traces[i] for i in inside], rng, fracs=(0.0, 0.5, 0.5))
    return train, inside[cal], inside[te]


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(order.size); ranks[order] = np.arange(1, order.size + 1)
    allv = np.concatenate([neg, pos])
    for v in np.unique(allv):                         # average ranks over ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[neg.size:].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def fit_neural(name, traces, feats, train_idx, seed, device, epochs=40, lr=1e-3, wd=1e-2,
               batch_size=32, d=128, layers=3, dropout=0.2):
    """Train one scorer on the training records; epoch chosen on a held-out part of them."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(train_idx)
    n_val = max(8, int(0.15 * idx.size))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    model = M.build(name, feats[0].shape[1], d=d, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    val_batch = M.collate([traces[i] for i in val_idx], [feats[i] for i in val_idx], device)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(fit_idx)
        for s in range(0, perm.size, batch_size):
            chunk = perm[s:s + batch_size]
            b = M.collate([traces[i] for i in chunk], [feats[i] for i in chunk], device)
            logit = M.trace_max(model(b), b["step_graph"], b["n_graphs"])
            loss = loss_fn(logit, b["y"])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = M.trace_max(model(val_batch), val_batch["step_graph"], val_batch["n_graphs"])
        auc = _auroc(val_batch["y"].cpu().numpy(), v.cpu().numpy())
        if auc > best:
            best, best_state = auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def score_neural(model, traces, feats, idx, device) -> np.ndarray:
    b = M.collate([traces[i] for i in idx], [feats[i] for i in idx], device)
    return M.trace_max(model(b), b["step_graph"], b["n_graphs"]).cpu().numpy().astype(np.float64)


def _pooled(feats, idx) -> np.ndarray:
    return np.stack([feats[i].mean(0) for i in idx])


def fit_score_logreg(traces, feats, train_idx, eval_idx, seed) -> np.ndarray:
    """Non-causal baseline: logistic regression on the mean node feature of the whole record."""
    from sklearn.linear_model import LogisticRegression

    y = np.array([traces[i].label for i in train_idx])
    clf = LogisticRegression(C=10.0, max_iter=2000, random_state=seed).fit(_pooled(feats, train_idx), y)
    return clf.decision_function(_pooled(feats, eval_idx)).astype(np.float64)


def rule_last_tool(traces, idx) -> np.ndarray:
    """Non-causal dataset-artefact baseline: 1 if the record ends on a tool call."""
    return np.array([float(traces[i].node_type[traces[i].steps[-1][1]] == TOOL) for i in idx])


# --------------------------------------------------------------------------
# Conformal evaluation of one scorer on one split
# --------------------------------------------------------------------------

def _one_per_group(groups, mask, rng) -> np.ndarray:
    """Index of one uniformly random member of each group, among records where mask holds."""
    groups = np.asarray(groups)
    picks = []
    for g in np.unique(groups[mask]):
        members = np.flatnonzero(mask & (groups == g))
        picks.append(members[rng.integers(members.size)])
    return np.array(picks, dtype=int)


def evaluate(scores_cal, y_cal, scores_test, y_test, attack_test=None, alphas=ALPHAS,
             g_cal=None, g_test=None, rng=None, c_cal=None, c_test=None) -> dict:
    scores_cal, scores_test = np.asarray(scores_cal, float), np.asarray(scores_test, float)
    y_cal, y_test = np.asarray(y_cal), np.asarray(y_test)
    out = {"auroc": _auroc(y_test, scores_test),
           "n_cal_safe": int((y_cal == 0).sum()), "n_test_safe": int((y_test == 0).sum()),
           "n_test_unsafe": int((y_test == 1).sum())}
    safe_cal, unsafe_cal = scores_cal[y_cal == 0], scores_cal[y_cal == 1]
    for a in alphas:
        q = conformal_quantile(safe_cal, a)
        blocked = scores_test > q
        key = f"{a:.2f}"
        out[f"fbr@{key}"] = float(blocked[y_test == 0].mean()) if (y_test == 0).any() else float("nan")
        out[f"block@{key}"] = float(blocked[y_test == 1].mean()) if (y_test == 1).any() else float("nan")
        out[f"q_finite@{key}"] = float(math.isfinite(q))
        if attack_test is not None:
            for at in ("injection", "unintended"):
                m = (y_test == 1) & (np.asarray(attack_test) == at)
                out[f"block_{at}@{key}"] = float(blocked[m].mean()) if m.any() else float("nan")
    # Group as the calibration unit: one random safe record per calibration prompt
    # group sets the threshold; it is evaluated on one random record per test group.
    # When groups are exchangeable this is exact (the pooled rule above is not).
    if g_cal is not None and g_test is not None:
        rng = rng or np.random.default_rng(0)
        cal_pick = _one_per_group(g_cal, y_cal == 0, rng)
        te_safe = _one_per_group(g_test, y_test == 0, rng)
        te_unsafe = _one_per_group(g_test, y_test == 1, rng)
        out["n_cal_safe_groups"] = int(cal_pick.size)
        for a in alphas:
            q = conformal_quantile(scores_cal[cal_pick], a)
            key = f"{a:.2f}"
            out[f"fbr_g1@{key}"] = float((scores_test[te_safe] > q).mean()) if te_safe.size else float("nan")
            out[f"block_g1@{key}"] = float((scores_test[te_unsafe] > q).mean()) if te_unsafe.size else float("nan")
    # Category-conditional (Mondrian) calibration at alpha = 0.10: one threshold per category.
    if c_cal is not None and c_test is not None:
        c_cal, c_test = np.asarray(c_cal), np.asarray(c_test)
        q_pool = conformal_quantile(safe_cal, 0.10)
        blocked_m = np.zeros(scores_test.size, dtype=bool)
        for c in np.unique(c_test):
            m = c_test == c
            q_c = conformal_quantile(scores_cal[(y_cal == 0) & (c_cal == c)], 0.10)
            blocked_m[m] = scores_test[m] > q_c
            safe_c = m & (y_test == 0)
            if safe_c.any():
                out[f"fbr_cat:{c}@0.10"] = float((scores_test[safe_c] > q_pool).mean())
                out[f"fbr_mond_cat:{c}@0.10"] = float(blocked_m[safe_c].mean())
        out["fbr_mond@0.10"] = float(blocked_m[y_test == 0].mean()) if (y_test == 0).any() else float("nan")
        out["block_mond@0.10"] = float(blocked_m[y_test == 1].mean()) if (y_test == 1).any() else float("nan")
    # Two-sided guard (Appendix C) at alpha = beta = 0.10.
    q = conformal_quantile(safe_cal, 0.10)
    lam = lower_conformal_quantile(unsafe_cal, 0.10)
    block = scores_test > q
    allow = (~block) & (scores_test < lam)
    out["ts_block_safe"] = float(block[y_test == 0].mean()) if (y_test == 0).any() else float("nan")
    out["ts_allow_unsafe"] = float(allow[y_test == 1].mean()) if (y_test == 1).any() else float("nan")
    out["ts_escalate"] = float((~block & ~allow).mean())
    return out


def run_split(traces, feats, parts, seed, device, model_names, train_kw=None) -> dict:
    """Fit and evaluate every requested scorer on one split. Returns {model: metrics}."""
    tr, cal, te = parts
    y_cal = np.array([traces[i].label for i in cal]); y_te = np.array([traces[i].label for i in te])
    attack = [traces[i].attack_type for i in te]
    g_cal = [traces[i].group for i in cal]; g_te = [traces[i].group for i in te]
    c_cal = [traces[i].category for i in cal]; c_te = [traces[i].category for i in te]
    res = {}
    for name in model_names:
        t0 = time.time()
        if name in NEURAL:
            model = fit_neural(name, traces, feats, tr, seed, device, **(train_kw or {}))
            s_cal, s_te = (score_neural(model, traces, feats, p, device) for p in (cal, te))
        elif name == "logreg_pooled":
            s_cal = fit_score_logreg(traces, feats, tr, cal, seed)
            s_te = fit_score_logreg(traces, feats, tr, te, seed)
        elif name == "rule_last_tool":
            s_cal, s_te = rule_last_tool(traces, cal), rule_last_tool(traces, te)
        else:
            raise ValueError(name)
        res[name] = evaluate(s_cal, y_cal, s_te, y_te, attack, g_cal=g_cal, g_test=g_te,
                             rng=np.random.default_rng(seed), c_cal=c_cal, c_test=c_te)
        res[name]["seconds"] = round(time.time() - t0, 2)
        if name == "rule_last_tool":                 # its natural operating point, uncalibrated
            res[name]["block@rule"] = float(s_te[y_te == 1].mean())
            res[name]["fbr@rule"] = float(s_te[y_te == 0].mean())
    return res


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(rows: list[dict]) -> dict:
    """rows: one {model: metrics} dict per split -> {model: {metric: {mean, sd, n}}}."""
    out: dict = {}
    for name in rows[0]:
        out[name] = {}
        for k in rows[0][name]:
            v = np.array([r[name][k] for r in rows if name in r and k in r[name]], dtype=float)
            v = v[~np.isnan(v)]
            if v.size:
                out[name][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                                "n": int(v.size)}
    return out


def paired(rows: list[dict], a: str, b: str, metric: str) -> dict:
    """Per-split difference a - b, with an exact two-sided sign test."""
    d = np.array([r[a][metric] - r[b][metric] for r in rows if a in r and b in r], dtype=float)
    d = d[~np.isnan(d)]
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    n = wins + losses
    p = 1.0
    if n:
        k = min(wins, losses)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"mean_diff": float(d.mean()) if d.size else float("nan"),
            "sd_diff": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            "a_better": wins, "b_better": losses, "ties": int((d == 0).sum()), "sign_test_p": p}
