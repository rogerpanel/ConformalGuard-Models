"""Tests for the closed-loop study and for LLM judges on multi-agent sessions.
No network, no API key, no model download: scripted agents, a hashing text encoder and stand-in clients."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from conformalguard.mas import closed_loop as CL
from conformalguard.mas import guard as GD
from conformalguard.mas import judge as J
from conformalguard.mas import labels as L
from conformalguard.mas import success as SU
from conformalguard.mas.sim import GUARD_NOTICE, ScriptedPolicy, run_session
from conformalguard.mas.world import make_world

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
judge_claude = pytest.importorskip("mas_judge_claude")


def sessions(n=40, start=0, **kw):
    return [run_session(make_world(s), ScriptedPolicy(), **kw) for s in range(start, start + n)]


class BlockTool:
    """A guard that blocks every proposal of one tool, to test the loop mechanics."""
    def __init__(self, tool): self.tool = tool
    def decide(self, session):
        ev = session["events"][-1]
        return ev["type"] == "action" and ev.get("tool") == self.tool, 1.0, 0.001


def test_open_loop_task_success_is_decided_by_code():
    safe = unsafe = 0
    for s in sessions(60):
        assert not s["closed_loop"]
        for o in SU.task_outcomes(s):
            assert o["success"] and o["success_lenient"] and o["blocked_actions"] == 0      # the scripted team always delivers
            assert o["violation_executed"] == (not o["safe_to_complete"])                # and leaks exactly when asked to
            safe += o["safe_to_complete"]; unsafe += not o["safe_to_complete"]
    assert safe > 20 and unsafe > 5
    s = sessions(1)[0]
    s["events"] = [e for e in s["events"] if not (e["type"] == "action" and e.get("tool") in L.DESTINATION_ARG | {"post_to_channel": 0})]
    assert not any(o["success"] for o in SU.task_outcomes(s))                            # no delivery, no success


def test_a_blocked_action_is_not_carried_out_and_the_agent_is_told():
    seen = []
    class Spy(ScriptedPolicy):
        def act(self, system, turns):
            seen.append(turns[-1]["content"]); return super().act(system, turns)
    hit = 0
    for seed in range(30):
        s = run_session(make_world(seed), Spy(), guard=BlockTool("send_email"))
        assert s["closed_loop"]
        ran = SU.executed(s)
        for i, e in enumerate(s["events"]):
            if e["type"] == "action":
                assert {"blocked", "guard_score", "guard_seconds"} <= set(e)
                if e.get("tool") == "send_email":
                    hit += 1
                    assert e["blocked"] and i not in ran
                    nxt = s["events"][i + 1] if i + 1 < len(s["events"]) else None
                    assert not (nxt and nxt["type"] == "return" and nxt["agent"] == e["agent"])   # no tool result
        for o in SU.task_outcomes(s):
            if o["tool"] == "send_email":
                assert not o["success"] and not o["violation_executed"] and o["blocked_actions"] >= 1
    assert hit > 5 and any(GUARD_NOTICE in t for t in seen)


def test_repeated_blocks_escalate_and_stop_only_that_thread():
    class Stubborn:                                            # proposes the same tool call for ever
        def act(self, system, turns):
            return {"kind": "tool", "tool": "read_inbox", "args": {}}
    s = run_session(make_world(5), Stubborn(), guard=BlockTool("read_inbox"), max_reproposals=2)
    acts = [e for e in s["events"] if e["type"] == "action"]
    by_thread = {}
    for e in acts:
        by_thread.setdefault(e["thread"], []).append(e)
    assert len(by_thread) == s["n_threads"]
    for evs in by_thread.values():
        assert len(evs) == 3 and all(e["blocked"] for e in evs) and evs[-1].get("escalated") and not evs[0].get("escalated")


def test_trained_guard_runs_in_the_loop_and_the_report_adds_up():
    train = sessions(120)
    g = GD.train_guard(train, GD.HashEncoder(), alpha=0.10, epochs=4, d=16, layers=2, max_len=128)
    assert g.info["n_train"] + g.info["n_cal"] == 120 and np.isfinite(g.threshold)
    s = sessions(1, start=500)[0]
    assert len(g.step_scores(s)) == sum(e["type"] == "action" for e in s["events"])
    blocked, score, secs = g.decide(s)
    assert blocked == (score > g.threshold) and secs >= 0
    a, b = sessions(12, start=700), sessions(12, start=700, guard=g)
    res, md = CL.report(a, b, {**g.info, "agent_model": "scripted"})
    A, B, P = res["without_guard"], res["with_guard"], res["paired"]
    assert A["sessions"] == B["sessions"] == P["paired_workspaces"] == 12 and A["blocked_actions"] == 0
    assert A["safe_success"] == A["safe_tasks"] and A["tasks"] == B["tasks"]
    assert P["both"] + P["only_without_guard"] + P["only_with_guard"] + P["neither"] == A["safe_tasks"]
    assert B["safe_tasks_with_block_success"] <= B["safe_tasks_with_block"] <= B["safe_tasks"]
    assert "guard_seconds_median" in B and "still succeeded" in md and "Corollary 1" in md


def test_judge_sees_the_policy_the_metadata_and_nothing_after_the_action():
    s = next(x for x in sessions(60) if any(y["y"] == L.VIOL for y in L.label_session(x)) and x["n_threads"] > 1)
    k = next(x["event"] for x in L.label_session(s) if x["y"] == L.VIOL)
    before, proposed = J.transcript(s, k)
    assert len(before) == k and proposed.startswith(f"[task {s['events'][k]['thread']}] PROPOSED ACTION by")
    assert "[destination: OUTSIDE the organisation]" in proposed
    assert any("[document classification: RESTRICTED]" in line for line in before)
    later = [e for e in s["events"][k + 1:] if e["type"] == "return"]
    prompt = J.user_prompt(s, k, J.RATING_FORMAT)
    assert all(e["text"] not in prompt for e in later if len(e["text"]) > 30)      # nothing from after the action
    assert "RESTRICTED must not be sent" in J.INSTRUCTIONS and "communicator: lookup_contact" in J.INSTRUCTIONS
    assert J.units(s) == [i for i, e in enumerate(s["events"]) if e["type"] == "action"]
    for word in ("thought", "reasoning", "step by step", "scratchpad"):
        assert word not in (J.INSTRUCTIONS + prompt + json.dumps(judge_claude.SCHEMA)).lower()


def test_judge_evaluation_controls_the_session_budget(tmp_path):
    ss = sessions(160)
    rng = np.random.default_rng(0)
    rows = {}
    for s in ss:                                                # an informative synthetic judge
        labs = L.label_session(s)
        for x in labs:
            rows[(s["seed"], x["event"])] = {"seed": s["seed"], "event": x["event"], "guard": "fake", "seconds": 0.5,
                                             "score": float(rng.normal() + 4.0 * (x["y"] == L.VIOL))}
    del rows[(ss[0]["seed"], J.units(ss[0])[0])]                 # one unscored step counts as blocked
    sc = [J.session_scores(s, rows) for s in ss]
    assert np.isinf(sc[0][0]) and J.session_scores(make_and_run(9999), rows) is None
    r = J.evaluate(ss, sc, reps=200)
    assert r["fbr_session@0.10"]["mean"] <= 0.10 + 0.02 and r["fbr@0.10"]["mean"] <= 0.10 + 0.02
    assert r["block@0.10"]["mean"] > 0.9 and r["unscored_steps"] == 1 and r["n_sessions"] == 160
    path = tmp_path / "scores.jsonl"
    path.write_text("".join(json.dumps(v) + "\n" for v in rows.values()))
    assert len(J.read_scores(path)) == len(rows) and J.read_scores(tmp_path / "missing.jsonl") == {}


def make_and_run(seed):
    return run_session(make_world(seed), ScriptedPolicy())


def test_claude_judge_request_and_response():
    s = sessions(1)[0]
    k = J.units(s)[-1]
    req = judge_claude.build_request(s, k, "claude-opus-5-5", "low", fallbacks=True)
    assert req["betas"] == [judge_claude.FALLBACK_BETA] and req["fallbacks"] == "default" and req["system"] == J.INSTRUCTIONS
    assert req["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": judge_claude.SCHEMA}}
    assert "thinking" not in req and "temperature" not in req
    blocks = req["messages"][0]["content"]
    assert blocks[-2]["cache_control"] == {"type": "ephemeral"} and sum("cache_control" in b for b in blocks) == 1
    assert "PROPOSED ACTION" in blocks[-1]["text"] and not any("PROPOSED ACTION" in b["text"] for b in blocks[:-1])
    assert "".join(b["text"] for b in blocks) == J.user_prompt(s, k, J.RATING_FORMAT).replace("\n</log>", "\n</log>")   # same text as one prompt
    earlier = judge_claude.build_request(s, J.units(s)[-2], "m", "low", fallbacks=False)["messages"][0]["content"]
    n = len(earlier) - 1                                          # the earlier step's log blocks ...
    assert [b["text"] for b in blocks[:n]] == [b["text"] for b in earlier[:n]] and n < len(blocks) - 1   # ... are a block-aligned prefix of this one
    assert "betas" not in judge_claude.build_request(s, k, "m", "low", fallbacks=False)
    usage = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=900, cache_creation_input_tokens=None)
    ok = NS(stop_reason="end_turn", model="claude-opus-5-5", usage=usage, content=[NS(type="thinking", thinking=""), NS(type="text", text='{"risk": 140}')])
    row = judge_claude.read_response(ok)
    assert row["score"] == 100.0 and row["cache_read_input_tokens"] == 900 and row["cache_creation_input_tokens"] == 0
    refused = NS(stop_reason="refusal", model="claude-opus-4-8", usage=usage, content=[], stop_details=NS(category="cyber"))
    row = judge_claude.read_response(refused)
    assert row["score"] is None and row["refusal_category"] == "cyber"
    assert judge_claude.read_response(NS(stop_reason="end_turn", model="m", usage=usage, content=[NS(type="text", text="no json")]))["score"] is None


def test_openai_compatible_judge_request_and_response():
    judge_openai = pytest.importorskip("mas_judge_openai")
    s = sessions(1)[0]
    req = judge_openai.build_request(s, J.units(s)[-1], "deepseek/deepseek-chat")
    assert req["model"] == "deepseek/deepseek-chat" and req["messages"][0] == {"role": "system", "content": J.INSTRUCTIONS}
    assert "PROPOSED ACTION" in req["messages"][1]["content"] and J.RATING_FORMAT in req["messages"][1]["content"]
    resp = NS(choices=[NS(message=NS(content="85", refusal=None))], model="deepseek/deepseek-chat", usage=NS(prompt_tokens=1200, completion_tokens=2))
    row = judge_openai.read_response(resp)
    assert row == {"score": 85.0, "served_by": "deepseek/deepseek-chat", "input_tokens": 1200, "output_tokens": 2, "raw": "85"}
    resp.choices[0].message = NS(content="I am not sure", refusal=None)
    assert judge_openai.read_response(resp)["score"] is None
    resp.choices[0].message = NS(content="Let's examine the policy.\n1. RESTRICTED documents must not be sent", refusal=None)
    assert judge_openai.read_response(resp)["score"] is None          # list numbering is not a rating
    assert J.leading_rating("100\n\nThe proposed action ...") == 100.0 and J.leading_rating(" 80 - because") == 80.0
    assert J.leading_rating("250") is None and J.leading_rating("0.5") is None and J.leading_rating(None) is None
    rows = {(1, 0): {"score": 1.0, "raw": "1. RESTRICTED documents"}, (1, 2): {"score": 90.0, "raw": "90"}}
    assert list(J.session_scores({"seed": 1, "events": [{"type": "action"}, {"type": "return"}, {"type": "action"}]}, rows)) == [float("inf"), 90.0]


def test_session_samples_are_nested_and_reproducible():
    ss = sessions(40)
    a, b = J.sample_sessions(ss, 10), J.sample_sessions(list(reversed(ss)), 25)
    assert [s["seed"] for s in a] == [s["seed"] for s in b[:10]]            # nested, and independent of input order
    assert len(J.sample_sessions(ss, 0)) == len(J.sample_sessions(ss, 999)) == 40


def test_channel_names_with_and_without_a_hash_are_the_same_destination():
    """Live agents write the channel as '#name'; a first version of the checker then counted every post as a failure."""
    assert L.destination("post_to_channel", {"channel": "#ops-updates"}) == L.destination("post_to_channel", {"channel": "ops-updates"}) == "#ops-updates"
    w = make_world(0)
    assert w.call("post_to_channel", {"channel": "#ops-updates", "text": "x"})[1] == {"destination": "#ops-updates"}
    hit = 0
    for seed in range(60):
        s = run_session(make_world(seed), ScriptedPolicy())
        for e in s["events"]:
            if e["type"] == "action" and e.get("tool") == "post_to_channel":
                e["args"]["channel"] = "#" + e["args"]["channel"]; hit += 1
        assert all(o["success"] for o in SU.task_outcomes(s) if o["tool"] == "post_to_channel")
    assert hit > 3


def test_closed_loop_report_compares_only_workspaces_run_in_both_arms():
    a = sessions(10, start=900)
    g = BlockTool("upload_file")
    b = sessions(6, start=900, guard=g)
    res, md = CL.report(a, b, {"alpha": 0.1, "model": "graph", "agent_model": "scripted"})
    assert res["without_guard"]["sessions"] == res["with_guard"]["sessions"] == res["paired"]["paired_workspaces"] == 6
    assert "6 run in both arms" in md and "(10 were generated without the guard and 6 with it)" in md
