"""Frozen text features for trace nodes.

Two interchangeable backends give every node a fixed vector:

``lsa``  TF-IDF (1-2 grams) followed by truncated SVD.  CPU only, no download.
``st``   a frozen sentence-transformers encoder (needs the model weights from
         the Hugging Face hub; use this on a GPU machine).

Both are fitted or applied to node texts only and never see a label, and they
treat every record symmetrically, so they do not disturb the exchangeability
that the conformal guarantee relies on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from conformalguard.rjudge.data import Trace


def _all_texts(traces: list[Trace]) -> list[str]:
    return [t if t else "[empty]" for tr in traces for t in tr.node_text]


def lsa_features(texts: list[str], dim: int = 256, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                            max_features=60_000, lowercase=True)
    X = tfidf.fit_transform(texts)
    dim = min(dim, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    return normalize(Z).astype(np.float32)


def st_features(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    Z = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True)
    return Z.astype(np.float32)


def node_features(traces: list[Trace], backend: str = "lsa", st_model: str =
                  "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str | Path | None = None,
                  dim: int = 256) -> list[np.ndarray]:
    """One (n_nodes, F) array per trace, in the order of ``traces``."""
    texts = _all_texts(traces)
    tag = f"{backend}-{st_model if backend == 'st' else dim}"
    key = hashlib.sha1(("\x00".join(texts) + tag).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"feat-{backend}-{key}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        Z = np.load(cache)
    else:
        if backend == "lsa":
            Z = lsa_features(texts, dim=dim)
        elif backend == "st":
            Z = st_features(texts, st_model)
        else:
            raise ValueError(f"unknown feature backend {backend!r}")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, Z)
    out, i = [], 0
    for tr in traces:
        out.append(Z[i:i + tr.n_nodes])
        i += tr.n_nodes
    assert i == Z.shape[0]
    return out
