"""A trained graph scorer with a conformal threshold, usable inside the simulator's loop.

``train_guard`` fits the scorer on one part of a set of logged sessions and takes the threshold
from the other part by the session-level rule (Corollary 1): the conformal quantile of the largest
safe-step score of each violation-free calibration session.  ``GraphGuard.decide`` then scores the
most recent proposed action of a running session from everything that precedes it.
"""

from __future__ import annotations

import hashlib
import threading
import time

import numpy as np
import torch

from conformalguard.conformal.core import conformal_quantile
from conformalguard.mas import convert as C
from conformalguard.mas import labels as L
from conformalguard.rjudge import models as M
from conformalguard.rjudge import steps as S


class STEncoder:
    """Frozen sentence encoder with a cache, so a running session only encodes its new nodes."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [t if t else "[empty]" for t in texts]
        new = sorted({t for t in texts if t not in self.cache})
        if new:
            Z = self.model.encode(new, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            self.cache.update({t: z.astype(np.float32) for t, z in zip(new, Z)})
        return np.stack([self.cache[t] for t in texts])


class HashEncoder:
    """Deterministic stand-in for tests: no model, no download.  Not for results."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha1((t or "[empty]").encode()).digest()
            out[i] = np.frombuffer((h * (self.dim // len(h) + 1))[:self.dim], dtype=np.uint8) / 255.0 - 0.5
        return out


def featurise(session: dict, encoder, sid: int = 0, thread_feature: bool = True):
    tr, meta, y, info = C.session_to_trace(session, sid, thread_feature=thread_feature)
    return tr, np.concatenate([encoder.encode(tr.node_text), meta], axis=1), y, info


class GraphGuard:
    def __init__(self, model, encoder, threshold: float, device: str = "cpu", thread_feature: bool = True, info: dict | None = None):
        self.model, self.encoder, self.threshold = model.eval(), encoder, float(threshold)
        self.device, self.thread_feature, self.info = device, thread_feature, dict(info or {})
        self._lock = threading.Lock()

    def step_scores(self, session: dict) -> np.ndarray:
        tr, feats, _, _ = featurise(session, self.encoder, thread_feature=self.thread_feature)
        with torch.no_grad():
            return self.model(M.collate([tr], [feats], self.device)).detach().cpu().numpy().astype(np.float64)

    def decide(self, session: dict) -> tuple[bool, float, float]:
        """(blocked, score, seconds) for the latest proposed action of a running session."""
        t0 = time.perf_counter()
        with self._lock:
            score = float(self.step_scores(session)[-1])
        return score > self.threshold, score, time.perf_counter() - t0


def train_guard(sessions: list[dict], encoder, alpha: float = 0.10, seed: int = 0, device: str = "cpu",
                model_name: str = "graph", cal_frac: float = 0.5, thread_feature: bool = True, **train_kw) -> GraphGuard:
    traces, feats, targets = [], [], []
    for i, s in enumerate(sessions):
        tr, f, y, _ = featurise(s, encoder, i, thread_feature)
        traces.append(tr); feats.append(f); targets.append(y)
    rng = np.random.default_rng(seed)
    label = np.array([t.label for t in traces])
    train, cal = [], []
    for c in (0, 1):                                            # stratified by whether a session has a violation
        idx = rng.permutation(np.flatnonzero(label == c))
        k = int(round((1 - cal_frac) * idx.size))
        train += idx[:k].tolist(); cal += idx[k:].tolist()
    kw = dict(epochs=40, layers=4, max_len=256); kw.update(train_kw)
    model = S.fit_steps(model_name, traces, feats, targets, np.array(sorted(train)), seed, device, **kw)
    with torch.no_grad():
        s_cal = S._step_logits(model, traces, feats, sorted(cal), device)
    clean = np.array([s[y == L.SAFE].max() for s, i in zip(s_cal, sorted(cal)) for y in [targets[i]] if not (y == L.VIOL).any()])
    q = conformal_quantile(clean, alpha)
    info = {"alpha": alpha, "threshold": float(q), "n_train": len(train), "n_cal": len(cal), "n_cal_violation_free": int(clean.size),
            "model": model_name, **{k: v for k, v in kw.items()}}
    return GraphGuard(model, encoder, q, device, thread_feature, info)
