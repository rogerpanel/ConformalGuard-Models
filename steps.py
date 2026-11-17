"""Step-labelled experiment: Proposition 1 on real records.

R-Judge's injection records are built from InjecAgent test cases.  In an unsafe
record the agent calls one of InjecAgent's attacker tools after reading injected
content; in a safe record it does not.  That makes the violating step decidable
by code: it is the first agent step after the first one whose action invokes a
tool on InjecAgent's attacker list.  (The first step is the user's own tool
call; one tool appears on both lists, which is why it is skipped.)  Steps after
the violating one are ignored: a guard would already have blocked.

With step labels the paper's step-level rule can be tested as stated:
  calibration  one random safe step from each calibration trace
  guarantee    a random safe step of a fresh trace is blocked w.p. <= alpha
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from conformalguard.conformal.core import conformal_quantile
from conformalguard.rjudge import experiment as E
from conformalguard.rjudge import models as M
from conformalguard.rjudge.data import TOOL, Trace

SAFE, VIOL, IGNORE = 0, 1, -1
_NAME = re.compile(r"\s*([A-Za-z][A-Za-z0-9_]*)")


def attacker_tools(injecagent_dir: str | Path) -> set[str]:
    tools: set[str] = set()
    for f in sorted((Path(injecagent_dir) / "data").glob("attacker_cases_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tools.update(json.loads(line).get("Attacker Tools") or [])
    if not tools:
        raise FileNotFoundError(f"no attacker tool list under {injecagent_dir}/data; run: "
                                f"git clone https://github.com/uiuc-kang-lab/InjecAgent {injecagent_dir}")
    return tools


def step_targets(trace: Trace, attackers: set[str]) -> np.ndarray | None:
    """Per-step labels (SAFE / VIOL / IGNORE), or None if the trace cannot be labelled."""
    if trace.attack_type != "injection":
        return None
    hit = None
    for k, (_, act) in enumerate(trace.steps):
        if k >= 1 and trace.node_type[act] == TOOL:
            m = _NAME.match(trace.node_text[act])
            if m and m.group(1) in attackers:
                hit = k
                break
    if (trace.label == 1) != (hit is not None):
        return None                                   # label and tool evidence disagree: leave out
    y = np.zeros(len(trace.steps), dtype=np.int64)
    if hit is not None:
        y[hit] = VIOL
        y[hit + 1:] = IGNORE
    return y


def labelled_subset(traces: list[Trace], attackers: set[str]):
    keep, targets = [], []
    for i, t in enumerate(traces):
        y = step_targets(t, attackers)
        if y is not None:
            keep.append(i); targets.append(y)
    return keep, targets


def _step_logits(model, traces, feats, idx, device) -> list[np.ndarray]:
    b = M.collate([traces[i] for i in idx], [feats[i] for i in idx], device)
    logits = model(b).detach().cpu().numpy().astype(np.float64)
    sg = b["step_graph"].cpu().numpy()
    return [logits[sg == g] for g in range(len(idx))]


def fit_steps(name, traces, feats, targets, train_idx, seed, device, epochs=40, lr=1e-3, wd=1e-2,
              batch_size=32, d=128, layers=3, dropout=0.2, max_len=M.MAX_LEN):
    """Per-step binary cross-entropy; the epoch is chosen on 15% of the training traces."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(train_idx)
    n_val = max(8, int(0.15 * idx.size))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    model = M.build(name, feats[0].shape[1], d=d, layers=layers, dropout=dropout, max_len=max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    y_val = np.concatenate([targets[i] for i in val_idx])
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(fit_idx)
        for s in range(0, perm.size, batch_size):
            chunk = perm[s:s + batch_size]
            b = M.collate([traces[i] for i in chunk], [feats[i] for i in chunk], device)
            y = torch.as_tensor(np.concatenate([targets[i] for i in chunk]), device=device)
            mask = y >= 0
            loss = loss_fn(model(b)[mask], y[mask].float())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            s_val = np.concatenate(_step_logits(model, traces, feats, val_idx, device))
        auc = E._auroc(y_val[y_val >= 0], s_val[y_val >= 0])
        if not np.isnan(auc) and auc > best:
            best, best_state = auc, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def evaluate_steps(s_cal, y_cal, s_test, y_test, rng, alphas=E.ALPHAS) -> dict:
    """s_*: list of per-trace score arrays; y_*: matching arrays of SAFE / VIOL / IGNORE."""
    def one_safe(scores, labels):
        return np.array([s[rng.choice(np.flatnonzero(y == SAFE))] for s, y in zip(scores, labels)
                         if (y == SAFE).any()])
    cal_one = one_safe(s_cal, y_cal)
    cal_pool = np.concatenate([s[y == SAFE] for s, y in zip(s_cal, y_cal)])
    test_one = one_safe(s_test, y_test)
    viol = [(s, int(np.flatnonzero(y == VIOL)[0])) for s, y in zip(s_test, y_test) if (y == VIOL).any()]
    all_s = np.concatenate(s_test); all_y = np.concatenate(y_test)
    out = {"auroc_step": E._auroc(all_y[all_y >= 0], all_s[all_y >= 0]),
           "n_cal": int(cal_one.size), "n_cal_pooled": int(cal_pool.size), "n_test_violating": len(viol)}
    for a in alphas:
        key = f"{a:.2f}"
        for tag, cal in (("", cal_one), ("_pooled", cal_pool)):
            q = conformal_quantile(cal, a)
            out[f"fbr{tag}@{key}"] = float((test_one > q).mean())
            out[f"fbr_allsteps{tag}@{key}"] = float(np.mean([(s[y == SAFE] > q).mean() for s, y in zip(s_test, y_test)
                                                             if (y == SAFE).any()]))
            out[f"block_at{tag}@{key}"] = float(np.mean([s[k] > q for s, k in viol])) if viol else float("nan")
            out[f"block_by{tag}@{key}"] = float(np.mean([(s[:k + 1] > q).any() for s, k in viol])) if viol else float("nan")
    return out


def run_split(traces, feats, targets, parts, seed, device, model_names, train_kw=None) -> dict:
    tr, cal, te = parts
    res = {}
    for name in model_names:
        model = fit_steps(name, traces, feats, targets, tr, seed, device, **(train_kw or {}))
        with torch.no_grad():
            s_cal = _step_logits(model, traces, feats, cal, device)
            s_te = _step_logits(model, traces, feats, te, device)
        res[name] = evaluate_steps(s_cal, [targets[i] for i in cal], s_te, [targets[i] for i in te],
                                   np.random.default_rng(seed))
    return res
