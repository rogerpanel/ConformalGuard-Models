"""Step-level conformal evaluation on simulated multi-agent sessions.

Scorers are trained with a per-step loss on the code-derived labels (as in the
R-Judge step experiment) and compared on identical features:

``step``           the current step only
``seq``            causal Transformer over all nodes of the session in log order
``graph``          relation-aware attention over the typed execution graph
``graph_nocites``  the same without the provenance (``cites``) edges
``graph_untyped``  the same graph with one node type and one relation

Calibration uses one random safe step from each calibration session, so the
false-block rate of a random safe step of a fresh session is at most alpha
(Proposition 1).  Reported per split:

  fbr              a random safe step of a test session is blocked (guaranteed <= alpha)
  fbr_allsteps     share of a test session's safe steps that are blocked, averaged over sessions
  block            share of violating steps that are blocked when proposed
  block_<kind>     the same for one kind of violation
  *_single, *_concurrent   the same restricted to sessions with one / several task threads
  fbr_session, block_session   the session-level rule (Corollary 1): the threshold comes from the
                   largest step score of each violation-free calibration session; a violation-free
                   test session is falsely blocked if any step exceeds it (guaranteed <= alpha).
                   This is the stricter and more telling budget: most steps of a session are
                   plainly harmless, so a guard can meet the per-step budget while blocking every
                   outgoing message, but it cannot meet the per-session budget that way.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from conformalguard.conformal.core import conformal_quantile
from conformalguard.mas import labels as L
from conformalguard.rjudge import experiment as E
from conformalguard.rjudge import steps as S
from conformalguard.rjudge.data import REL

VARIANTS = {"step": ("step", False), "seq": ("seq", False), "graph": ("graph", False),
            "graph_nocites": ("graph", True), "graph_untyped": ("graph_untyped", False)}
ALPHAS = E.ALPHAS


def without_cites(traces):
    out = []
    for t in traces:
        c = copy.copy(t)
        c.edges = [e for e in t.edges if e[2] != REL["cites"]]
        out.append(c)
    return out


def evaluate(s_cal, y_cal, s_test, y_test, info_test, threads_test, rng, alphas=ALPHAS) -> dict:
    """s_*: per-session score arrays; y_*: matching SAFE / VIOL / IGNORE arrays;
    info_test: per-session list of {"kinds"} per step; threads_test: number of task threads per session."""
    def one_safe(scores, labels):
        return np.array([s[rng.choice(np.flatnonzero(y == L.SAFE))] if (y == L.SAFE).any() else np.nan
                         for s, y in zip(scores, labels)])
    cal_one = one_safe(s_cal, y_cal)
    cal_one = cal_one[~np.isnan(cal_one)]
    test_one = one_safe(s_test, y_test)
    conc = np.asarray(threads_test) > 1
    clean_cal = np.array([s[y == L.SAFE].max() for s, y in zip(s_cal, y_cal) if not (y == L.VIOL).any()])
    viol = [(s[k], info[k]["kinds"][0], bool(c)) for s, y, info, c in zip(s_test, y_test, info_test, conc)
            for k in np.flatnonzero(y == L.VIOL)]
    v_s = np.array([v[0] for v in viol]); v_kind = np.array([v[1] for v in viol]); v_conc = np.array([v[2] for v in viol], dtype=bool)
    all_s, all_y = np.concatenate(s_test), np.concatenate(y_test)
    mean = lambda x: float(np.mean(x)) if np.size(x) else float("nan")
    out = {"auroc_step": E._auroc(all_y[all_y >= 0], all_s[all_y >= 0]),
           "n_cal": int(cal_one.size), "n_cal_clean_sessions": int(clean_cal.size),
           "n_test_violating_steps": int(v_s.size), "n_test_violating_concurrent": int(v_conc.sum())}
    for a in alphas:
        key = f"{a:.2f}"
        q = conformal_quantile(cal_one, a)
        blocked_one = test_one > q                       # NaN (no safe step) compares False; excluded below
        ok = ~np.isnan(test_one)
        out[f"fbr@{key}"] = mean(blocked_one[ok])
        out[f"fbr_single@{key}"] = mean(blocked_one[ok & ~conc])
        out[f"fbr_concurrent@{key}"] = mean(blocked_one[ok & conc])
        out[f"fbr_allsteps@{key}"] = mean([(s[y == L.SAFE] > q).mean() for s, y in zip(s_test, y_test) if (y == L.SAFE).any()])
        qs = conformal_quantile(clean_cal, a) if clean_cal.size else float("inf")
        clean = np.array([not (y == L.VIOL).any() for y in y_test])
        hit = np.array([bool((s[y == L.SAFE] > qs).any()) for s, y in zip(s_test, y_test)])
        out[f"fbr_session@{key}"] = mean(hit[clean])
        out[f"fbr_session_single@{key}"] = mean(hit[clean & ~conc])
        out[f"fbr_session_concurrent@{key}"] = mean(hit[clean & conc])
        for tag, thr in (("block", q), ("block_session", qs)):
            out[f"{tag}@{key}"] = mean(v_s > thr)
            out[f"{tag}_single@{key}"] = mean(v_s[~v_conc] > thr)
            out[f"{tag}_concurrent@{key}"] = mean(v_s[v_conc] > thr)
            for kind in L.KINDS:
                out[f"{tag}_{kind}@{key}"] = mean(v_s[v_kind == kind] > thr)
    return out


def run_split(traces, feats, targets, infos, parts, seed, device, names, train_kw=None) -> dict:
    tr, cal, te = parts
    stripped = None
    res = {}
    for name in names:
        base, drop_cites = VARIANTS[name]
        if drop_cites and stripped is None:
            stripped = without_cites(traces)
        use = stripped if drop_cites else traces
        model = S.fit_steps(base, use, feats, targets, tr, seed, device, **(train_kw or {}))
        with torch.no_grad():
            s_cal = S._step_logits(model, use, feats, cal, device)
            s_te = S._step_logits(model, use, feats, te, device)
        res[name] = evaluate(s_cal, [targets[i] for i in cal], s_te, [targets[i] for i in te],
                             [infos[i] for i in te], [1 + (traces[i].attack_type == "concurrent") for i in te],
                             np.random.default_rng(seed))
    return res


def provenance_rule(trace, meta) -> np.ndarray:
    """A hand-written reference rule, one 0/1 score per step: the proposed tool call names an external
    destination and has a ``cites`` edge from a memory entry whose resource is restricted.

    It uses exactly what the graph scorer is given.  Where it matches the labels, a learned scorer on the
    graph cannot be credited with discovering anything a monitor could not have written down.
    """
    from conformalguard.mas import convert as C
    restricted = C._C0 + C.CLASSES.index("restricted")
    sources: dict[int, list[int]] = {}
    for src, dst, rel in trace.edges:
        if rel == REL["cites"]:
            sources.setdefault(dst, []).append(src)
    return np.array([float(meta[act, C._D0] == 1 and any(meta[s, restricted] == 1 for s in sources.get(act, [])))
                     for _, act in trace.steps])
