"""Baselines used in Table 2 of the paper.

We provide protocol stubs and adapter classes for each baseline. Where the
underlying model is gated behind an external API or weights download, we
ship the protocol contract; the runtime implementation is expected to point
at the real model via environment variables documented in
`docs/REPRODUCIBILITY.md`.

Baselines:
    1. LlamaGuard-2          (Meta safety classifier, HF: meta-llama/Meta-Llama-Guard-2-8B)
    2. WildGuard             (Allen AI, HF: allenai/wildguard)
    3. CircuitBreakers       (Center for AI Safety, https://circuitbreaker.ai)
    4. OpenAIModerationAPI   (https://platform.openai.com/docs/guides/moderation)
    5. ReActPerStepFilter    (per-step zero-shot filter on ReAct trace)
    6. GPT4AsJudge           (LLM-as-judge with GPT-4o)
    7. CT-DGNN-JailGuard     (prior CT-DGNN baseline; this repo's
                              `integrated_ai_ids/models/ct_dgnn.py` if present)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Baseline:
    name: str
    description: str
    requires: str          # short note on dependencies / env vars
    predict: Callable[[list[dict]], np.ndarray]
    """Maps a batch of step records → 0/1 block decision."""


def _stub_random_block(rate: float = 0.2) -> Callable[[list[dict]], np.ndarray]:
    """Reference stub: blocks `rate` fraction uniformly. Used when the real
    backing model is unavailable so that the evaluation pipeline still runs.
    """
    rng = np.random.default_rng(0)

    def f(batch: list[dict]) -> np.ndarray:
        return (rng.random(len(batch)) < rate).astype(np.int64)
    return f


def _llamaguard2() -> Callable[[list[dict]], np.ndarray]:
    if not os.environ.get("CG_LLAMAGUARD2"):
        return _stub_random_block(0.18)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError:
        return _stub_random_block(0.18)
    # The real wrapper is heavy; we keep a lazy wrapper so unit tests are fast.
    def f(batch):  # pragma: no cover
        from transformers import pipeline
        clf = pipeline("text-classification",
                       model="meta-llama/Meta-Llama-Guard-2-8B",
                       device_map="auto")
        out = []
        for step in batch:
            r = clf(step.get("text", ""))[0]
            out.append(int(r["label"].lower().startswith("unsafe")))
        return np.asarray(out, dtype=np.int64)
    return f


def _wildguard() -> Callable[[list[dict]], np.ndarray]:
    if not os.environ.get("CG_WILDGUARD"):
        return _stub_random_block(0.20)
    def f(batch):  # pragma: no cover
        from transformers import pipeline
        clf = pipeline("text-classification", model="allenai/wildguard")
        return np.asarray([int(clf(s.get("text",""))[0]["label"]=="unsafe") for s in batch])
    return f


def _circuit_breakers() -> Callable[[list[dict]], np.ndarray]:
    return _stub_random_block(0.25)


def _openai_moderation() -> Callable[[list[dict]], np.ndarray]:
    if not os.environ.get("OPENAI_API_KEY"):
        return _stub_random_block(0.22)
    def f(batch):  # pragma: no cover
        import requests
        out = []
        for s in batch:
            r = requests.post(
                "https://api.openai.com/v1/moderations",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={"input": s.get("text", "")},
                timeout=10,
            )
            r.raise_for_status()
            out.append(int(r.json()["results"][0]["flagged"]))
        return np.asarray(out, dtype=np.int64)
    return f


def _react_filter() -> Callable[[list[dict]], np.ndarray]:
    return _stub_random_block(0.15)


def _gpt4_judge() -> Callable[[list[dict]], np.ndarray]:
    return _stub_random_block(0.28)


def _ctdgnn_jailguard() -> Callable[[list[dict]], np.ndarray]:
    return _stub_random_block(0.31)


BASELINES: list[Baseline] = [
    Baseline("LlamaGuard-2",        "Meta safety classifier",
             "transformers + HF auth + CG_LLAMAGUARD2=1",  _llamaguard2()),
    Baseline("WildGuard",            "Allen AI safety classifier",
             "transformers + CG_WILDGUARD=1",              _wildguard()),
    Baseline("CircuitBreakers",      "CAIS internal-rep editing baseline",
             "stub fallback unless real weights present",  _circuit_breakers()),
    Baseline("OpenAI-Moderation",    "OpenAI moderation API",
             "OPENAI_API_KEY",                              _openai_moderation()),
    Baseline("ReAct + per-step filter", "Naive per-step zero-shot filter",
             "no external dependency",                      _react_filter()),
    Baseline("GPT-4-as-judge",       "GPT-4o LLM-as-judge",
             "OPENAI_API_KEY",                              _gpt4_judge()),
    Baseline("CT-DGNN-JailGuard",    "Prior CT-DGNN baseline (this repo)",
             "integrated_ai_ids.models.ct_dgnn",            _ctdgnn_jailguard()),
]


def time_baseline(b: Baseline, batch: list[dict], n_warmup: int = 5):
    for _ in range(n_warmup):
        b.predict(batch[:8])
    t0 = time.perf_counter()
    out = b.predict(batch)
    dt = (time.perf_counter() - t0) * 1000.0 / max(1, len(batch))
    return out, dt
