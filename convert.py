"""Simulated session -> execution-graph trace, node metadata and step labels.

Nodes (the four types of the paper's schema)
  message   a user request; an agent's hand-over message or final note
  agent     one agent invocation (text: its role; agents are not asked for any commentary)
  tool      a proposed tool call (text: tool name and arguments)
  memory    what a tool returned

Edges, all pointing forward in time
  replies    user request -> the coordinator invocation that answers it; agent -> the message it emits
  invokes    agent -> the tool call it proposes
  writes     tool call -> the memory entry it produced
  reads      memory entry -> the next invocation of the agent that received it
  delegates  hand-over message -> the first invocation of the receiving agent
  next       agent -> the next invocation of the same agent instance
  cites      the node where an identifier first appeared -> a later proposed action
             that contains the same identifier.  Identifiers are record references,
             e-mail addresses and URLs, found by string matching; this is the
             provenance edge of the schema.  It uses only nodes that precede the
             action, so a runtime monitor can add it when the action is proposed.

Node metadata (appended to the text features; the same for every scorer)
  role of the acting agent (4), task-thread slot (3, optional), classification of
  the resource a memory entry came from (3: public / internal / restricted), and
  whether a proposed tool call names an external or an internal destination (2).
  Classification is resource metadata held by the monitor; agents never see it.

A decision step is one agent action, as in the R-Judge experiment.
"""

from __future__ import annotations

import json
import re

import numpy as np

from conformalguard.mas import labels as L
from conformalguard.mas.world import is_external
from conformalguard.rjudge.data import AGENT, MEMORY, MESSAGE, REL, TOOL, Trace

ROLES = ("coordinator", "researcher", "communicator", "admin")
CLASSES = ("public", "internal", "restricted")
MAX_THREADS = 3
META_DIM = len(ROLES) + MAX_THREADS + len(CLASSES) + 2
_R0, _T0, _C0, _D0 = 0, len(ROLES), len(ROLES) + MAX_THREADS, len(ROLES) + MAX_THREADS + len(CLASSES)
IDENT = re.compile(r"REF-\d{6}|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|https?://[^\s\"'<>)\]]+")


def identifiers(text: str) -> set[str]:
    return {m.rstrip(".,;") for m in IDENT.findall(text or "")}


def session_to_trace(session: dict, sid: int, thread_feature: bool = True, cites: bool = True):
    """Returns (trace, metadata array (n_nodes, META_DIM), step targets, per-step info)."""
    labs = L.label_session(session)
    tr = Trace(id=sid, label=int(any(x["y"] == L.VIOL for x in labs)),
               category=str(session.get("agent_model", "unknown")),
               attack_type="concurrent" if session["n_threads"] > 1 else "single",
               group=str(session.get("seed", sid)))
    meta: list[np.ndarray] = []
    origin: dict[str, int] = {}                       # identifier -> node where it first appeared
    inbox: dict[str, list[tuple[int, str]]] = {}      # agent name -> pending (node, relation)
    last_agent: dict[str, int] = {}
    unread: dict[str, list[int]] = {}
    last_tool: dict[str, int] = {}

    def add(ntype: int, text: str, thread: int, role: str | None = None) -> int:
        n = tr._add(ntype, text)
        m = np.zeros(META_DIM, dtype=np.float32)
        if role in ROLES:
            m[_R0 + ROLES.index(role)] = 1.0
        if thread_feature:
            m[_T0 + min(int(thread), MAX_THREADS - 1)] = 1.0
        meta.append(m)
        return n

    def note(n: int) -> None:
        for tok in identifiers(tr.node_text[n]):
            origin.setdefault(tok, n)

    for ev in session["events"]:
        th = ev["thread"]
        if ev["type"] == "user":
            u = add(MESSAGE, ev["text"], th)
            note(u)
            inbox.setdefault(f"coordinator#{th}", []).append((u, "replies"))
        elif ev["type"] == "action":
            name, role = ev["agent"], ev["role"]
            a = add(AGENT, f"{role} agent", th, role)
            for src, rel in inbox.pop(name, []):
                tr._edge(src, a, rel)
            if name in last_agent:
                tr._edge(last_agent[name], a, "next")
            for mem in unread.pop(name, []):
                tr._edge(mem, a, "reads")
            last_agent[name] = a
            if ev["kind"] == "tool":
                n = add(TOOL, f"{ev.get('tool', '')} {json.dumps(ev.get('args') or {}, ensure_ascii=False)}", th, role)
                tr._edge(a, n, "invokes")
                dest = L.destination(ev.get("tool", ""), ev.get("args") or {})
                if dest:
                    meta[n][_D0 + (0 if is_external(dest) else 1)] = 1.0
                last_tool[name] = n
            else:
                to = (ev.get("handover") or "user").split("#")[0]
                n = add(MESSAGE, f"to {to}: {ev.get('content', '')}", th, role)
                tr._edge(a, n, "replies")
            if cites:
                for src in sorted({origin[t] for t in identifiers(tr.node_text[n]) if t in origin and origin[t] < a}):
                    tr.edges.append((src, n, REL["cites"]))
            note(a); note(n)
            if ev.get("handover"):
                inbox.setdefault(ev["handover"], []).append((n, "delegates"))
            tr.steps.append((a, n))
        elif ev["type"] == "return":
            name = ev["agent"]
            mem = add(MEMORY, ev.get("text", ""), th, name.split("#")[0])
            cls = (ev.get("meta") or {}).get("classification")
            if cls in CLASSES:
                meta[mem][_C0 + CLASSES.index(cls)] = 1.0
            if name in last_tool:
                tr._edge(last_tool[name], mem, "writes")
            unread.setdefault(name, []).append(mem)
            note(mem)
    targets = np.array([x["y"] for x in labs], dtype=np.int64)
    assert len(targets) == len(tr.steps)
    info = [{"thread": x["thread"], "kinds": x["kinds"]} for x in labs]
    return tr, np.stack(meta), targets, info


def load_sessions(paths) -> list[dict]:
    """Read sessions from one or more .jsonl or .jsonl.gz files (one session per line), in file order."""
    import gzip
    out = []
    for p in ([paths] if isinstance(paths, (str, bytes)) or hasattr(paths, "open") else paths):
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as fh:
            out += [json.loads(line) for line in fh if line.strip()]
    return out


def summary(sessions: list[dict]) -> dict:
    """Counts worth reporting with every result."""
    from collections import Counter
    c: Counter = Counter()
    kinds: Counter = Counter()
    for s in sessions:
        labs = L.label_session(s)
        viol = [x for x in labs if x["y"] == L.VIOL]
        c["sessions"] += 1
        c["sessions_with_violation"] += bool(viol)
        c["concurrent_sessions"] += s["n_threads"] > 1
        c["steps"] += len(labs)
        c["safe_steps"] += sum(x["y"] == L.SAFE for x in labs)
        c["violating_steps"] += len(viol)
        c["tasks"] += s["n_threads"]
        c["tasks_with_injection"] += sum(bool(t.get("injected")) for t in s["tasks"])
        c["refusals"] += int(s.get("refusals", 0))
        for x in viol:
            kinds[x["kinds"][0]] += 1
    return {**dict(c), "violations_by_kind": dict(kinds),
            "sim_versions": dict(Counter(str(s.get("sim_version", 1)) for s in sessions)),
            "by_agent_model": dict(Counter(str(s.get("agent_model", "unknown")) for s in sessions))}
