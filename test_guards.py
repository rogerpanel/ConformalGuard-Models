"""Tests for the LLM-guard scoring scripts, the score evaluator and the step-level experiment.
No network and no API key: the clients are stand-ins that return canned responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from conformalguard.rjudge import experiment as E
from conformalguard.rjudge import guards as G
from conformalguard.rjudge import steps as S
from conformalguard.rjudge.data import record_to_trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tests.test_rjudge import SAFE, UNSAFE  # noqa: E402  (hand-written records in the R-Judge schema)

claude = pytest.importorskip("rjudge_judge_claude")
oai = pytest.importorskip("rjudge_judge_openai")
import rjudge_guard_eval as GE  # noqa: E402
import rjudge_judge_local as LOC  # noqa: E402


def test_causal_transcript_stops_at_the_proposed_action():
    assert G.n_steps(UNSAFE) == 2
    b0, b1 = G.transcript_blocks(UNSAFE, 0), G.transcript_blocks(UNSAFE, 1)
    assert len(b0) == 3 and b0[-1].startswith("[AGENT, PROPOSED ACTION]") and "GmailReadEmail" in b0[-1]
    assert not any("[ENVIRONMENT]" in b for b in b0)               # the response to step 0 is not shown
    assert any("[ENVIRONMENT]" in b for b in b1) and "SmartLockGrantAccess" in b1[-1]
    assert b1[:2] == b0[:2]                                        # earlier blocks are a stable prefix
    full = G.transcript_blocks(UNSAFE)
    assert not any("PROPOSED ACTION" in b for b in full)
    assert [s for _, s in G.units([UNSAFE, SAFE], "causal")] == [0, 1, 0, 1]
    assert [s for _, s in G.units([UNSAFE, SAFE], "full")] == [-1, -1]


def test_parse_rating():
    assert G.parse_rating("85") == 85.0 and G.parse_rating("Risk: 7 because ...") == 7.0
    assert G.parse_rating("0") == 0.0 and G.parse_rating("100.") == 100.0
    assert G.parse_rating("250") is None and G.parse_rating("no number") is None and G.parse_rating("0.75") is None


def test_claude_request_shape():
    req = claude.build_request(UNSAFE, 1, "claude-opus-5-5", "medium", fallbacks=True)
    assert req["model"] == "claude-opus-5-5" and req["betas"] == [claude.FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["output_config"]["effort"] == "medium" and req["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in req and "temperature" not in req
    blocks = req["messages"][0]["content"]
    assert blocks[0]["text"].startswith("<transcript>") and blocks[-2]["text"].endswith("</transcript>")
    assert blocks[-2]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in blocks[-1]
    assert "proposed action" in blocks[-1]["text"]
    prev = claude.build_request(UNSAFE, 0, "claude-opus-5-5", "medium", fallbacks=False)["messages"][0]["content"]
    assert "betas" not in claude.build_request(UNSAFE, 0, "m", "low", fallbacks=False)
    assert [b["text"] for b in blocks[:2]] == [b["text"] for b in prev[:2]]     # shared prefix across steps


def test_claude_response_handling():
    ok = NS(model="claude-opus-5-5", stop_reason="end_turn", usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
            content=[NS(type="thinking", thinking=""), NS(type="text", text=json.dumps({"risk": 130, "reason": "follows an injected instruction"}))])
    row = claude.read_response(ok)
    assert row["score"] == 100.0 and row["served_by"] == "claude-opus-5-5" and "injected" in row["reason"]
    refused = NS(model="claude-opus-5-5", stop_reason="refusal", usage=None, content=[], stop_details=NS(category="cyber"))
    row = claude.read_response(refused)
    assert row["score"] is None and row["refusal_category"] == "cyber"
    odd = NS(model="m", stop_reason="end_turn", usage=None, content=[NS(type="text", text="about 40 out of 100")])
    assert claude.read_response(odd)["score"] == 40.0
    calls = []
    client = NS(messages=NS(create=lambda **kw: calls.append(("plain", kw)) or ok),
                beta=NS(messages=NS(create=lambda **kw: calls.append(("beta", kw)) or ok)))
    claude.call(client, claude.build_request(SAFE, -1, "m", "low", fallbacks=True))
    claude.call(client, claude.build_request(SAFE, -1, "m", "low", fallbacks=False))
    assert [c[0] for c in calls] == ["beta", "plain"]


def test_openai_request_and_response():
    req = oai.build_request(UNSAFE, 1, "some-model", "rating")
    assert [m["role"] for m in req["messages"]] == ["system", "user"] and "logprobs" not in req
    assert "<transcript>" in req["messages"][1]["content"] and "0 (certainly safe)" in req["messages"][1]["content"]
    resp = NS(model="some-model", choices=[NS(message=NS(content="72 - it acts on an email"), logprobs=None)])
    assert oai.read_response(resp, "rating")["score"] == 72.0
    req = oai.build_request(UNSAFE, 1, "some-model", "logprob")
    assert req["logprobs"] is True and req["top_logprobs"] == 20
    top = [NS(token="B", logprob=-0.2), NS(token=" A", logprob=-1.7), NS(token="b", logprob=-3.0)]
    resp = NS(model="m", choices=[NS(message=NS(content="B"), logprobs=NS(content=[NS(top_logprobs=top)]))])
    assert oai.read_response(resp, "logprob")["score"] == pytest.approx(1.5)
    only_a = NS(model="m", choices=[NS(message=NS(content="A"), logprobs=NS(content=[NS(top_logprobs=[NS(token="A", logprob=-0.1)])]))])
    assert oai.read_response(only_a, "logprob")["score"] == pytest.approx(-5.0)      # B outside the top-k
    none = NS(model="m", choices=[NS(message=NS(content=""), logprobs=None)])
    assert oai.read_response(none, "logprob")["score"] is None


def test_local_judge_score_is_logit_difference():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return {"A": [3], " A": [4], "B": [5], " B": [6]}.get(s, [1, 2])
        def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
            assert messages[0]["role"] == "system" and "single letter" in messages[1]["content"]
            return {"input_ids": torch.arange(1, 9).unsqueeze(0)}
    class Model:
        device = "cpu"
        def __call__(self, input_ids, attention_mask):
            logits = torch.zeros(1, input_ids.shape[1], 10)
            logits[0, -1, 5] = 2.0                                  # "B"
            logits[0, -1, 3] = 0.5                                  # "A"
            return NS(logits=logits)
    a, b = LOC.letter_ids(Tok())
    assert (a, b) == ([3, 4], [5, 6])
    s = LOC.score_unit(Tok(), Model(), UNSAFE, 1, a, b, 6000)
    expect = float(torch.logsumexp(torch.tensor([2.0, 0.0]), 0) - torch.logsumexp(torch.tensor([0.5, 0.0]), 0))
    assert s == pytest.approx(expect)


def test_guard_eval_reads_scores_and_treats_missing_as_blocked(tmp_path):
    traces = [record_to_trace(dict(UNSAFE, id=1)), record_to_trace(dict(SAFE, id=2)), record_to_trace(dict(SAFE, id=3))]
    rows = [{"id": 1, "step": 0, "score": 10.0, "guard": "g", "mode": "causal"},
            {"id": 1, "step": 1, "score": 90.0, "guard": "g", "mode": "causal"},
            {"id": 2, "step": 0, "score": None, "guard": "g", "mode": "causal"}]
    p = tmp_path / "scores.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows))
    scores, info = GE.record_scores(p, traces)
    assert scores[0] == 90.0 and np.isinf(scores[1]) and np.isnan(scores[2])        # max; no score; no row
    assert info["rows"] == 3 and info["rows_without_score"] == 1 and info["records_scored"] == 2


def test_guard_eval_end_to_end_controls_fbr():
    rng = np.random.default_rng(0)
    traces = []
    for i in range(240):
        rec = dict(UNSAFE if i % 2 else SAFE, id=i)
        rec["contents"] = [[dict(m) for m in rec["contents"][0]]]
        rec["contents"][0][0]["content"] = f"prompt {i // 6}"
        traces.append(record_to_trace(rec, ["Web", "Program", "Finance"][i % 3]))
    scores = np.array([rng.normal(1.5 * t.label, 1.0) for t in traces])
    r = GE.evaluate_guard("g", scores, traces, reps=200, seed=0)
    assert r["random"]["fbr@0.10"]["mean"] <= 0.10 + 0.01
    assert r["random"]["block@0.10"]["mean"] > 0.3 and r["auroc_all"] > 0.75
    assert set(r["recal"]) == {"Finance", "Program", "Web"}
    assert r["recal"]["Web"]["fbr_in"]["mean"] <= 0.10 + 0.02


def test_mondrian_and_recal_split():
    rng = np.random.default_rng(1)
    n = 4000
    cat = rng.choice(["a", "b"], n); y = rng.integers(0, 2, n)
    s = rng.normal(np.where(cat == "b", 2.0, 0.0) + 1.0 * y, 1.0)   # category b scores higher overall
    half = n // 2
    m = E.evaluate(s[:half], y[:half], s[half:], y[half:], c_cal=cat[:half], c_test=cat[half:])
    assert m["fbr_cat:b@0.10"] > 0.15 and m["fbr_cat:a@0.10"] < 0.05         # pooled: unequal across categories
    assert abs(m["fbr_mond_cat:a@0.10"] - 0.10) < 0.04 and abs(m["fbr_mond_cat:b@0.10"] - 0.10) < 0.04
    traces = [record_to_trace(dict(SAFE if i % 2 else UNSAFE, id=i), "Web" if i < 40 else "Program") for i in range(100)]
    tr, cal, te = E.split_recal(traces, rng, "Web")
    assert all(traces[i].category == "Program" for i in tr) and len(tr) == 60
    assert all(traces[i].category == "Web" for i in np.concatenate([cal, te])) and len(cal) + len(te) == 40
    assert not set(cal) & set(te)


def test_step_targets_from_attacker_tools():
    attackers = {"SmartLockGrantAccess", "GmailReadEmail"}             # the user's own tool is also on the list
    bad = record_to_trace(UNSAFE); bad.attack_type = "injection"
    good = record_to_trace(SAFE); good.attack_type = "injection"
    assert S.step_targets(bad, attackers).tolist() == [S.SAFE, S.VIOL]  # step 0 skipped although listed
    assert S.step_targets(good, attackers).tolist() == [S.SAFE, S.SAFE]
    assert S.step_targets(bad, {"OtherTool"}) is None                   # unsafe label, no tool evidence
    good.attack_type = "unintended"
    assert S.step_targets(good, attackers) is None


def test_step_level_guarantee_with_one_step_per_trace():
    rng = np.random.default_rng(2)
    fbr = []
    for _ in range(300):
        def draw(n):
            scores, labels = [], []
            for _ in range(n):
                k = int(rng.integers(2, 6)); eff = rng.normal(0, 1)
                y = np.zeros(k, dtype=int)
                if rng.random() < 0.5:
                    y[-1] = S.VIOL
                scores.append(eff + 0.3 * rng.normal(size=k) + 3.0 * (y == S.VIOL)); labels.append(y)
            return scores, labels
        s_cal, y_cal = draw(80); s_te, y_te = draw(80)
        m = S.evaluate_steps(s_cal, y_cal, s_te, y_te, rng, alphas=(0.10,))
        fbr.append(m["fbr@0.10"])
    assert np.mean(fbr) <= 0.10 + 0.01
    assert m["block_at@0.10"] > 0.5 and m["n_cal"] == 80
