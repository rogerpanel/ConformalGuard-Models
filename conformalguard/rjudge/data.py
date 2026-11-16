"""R-Judge records -> execution-graph traces.

R-Judge (Yuan et al., Findings of EMNLP 2024; https://github.com/Lordog/R-Judge)
contains human-labelled records of an agent interacting with a user and an
environment.  Each record has a binary label (0 safe / 1 unsafe) agreed by its
annotators, an application category and an attack type.  The data are *not*
redistributed here: clone the repository and point ``load_rjudge`` at it.

Conversion to the execution-graph schema of the paper (Definition 1)
--------------------------------------------------------------------
node types   agent invocation  (text: the agent's thought)
             tool call         (text: the action string, when it is a tool call)
             memory entry      (text: the environment's return value)
             message           (text: a user message, or an agent reply / final answer)

relations    invokes   agent      -> tool call it proposes
             writes    tool/agent -> memory entry (the observation it produced)
             reads     memory     -> the next agent invocation that sees it
             replies   message    -> agent invocation that answers it,
                       agent      -> the message it emits
             next      agent      -> the next invocation of the same agent

Edges are oriented by information flow (earlier node -> later node), so a
node's representation after message passing depends only on its ancestors and
a guard that scores step k never sees the environment's response to step k.

Two deviations from the paper's schema, both forced by the data: R-Judge
records have a single agent, so ``delegates`` and ``cites`` never occur, and a
seventh relation ``next`` links consecutive invocations of that agent.

A *decision step* is one agent message: the proposed action is its tool call
or its reply.  R-Judge labels whole records, not steps, so the unit of
labelling, calibration and evaluation in this experiment is the record.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NODE_TYPES = ("agent", "tool", "memory", "message")
RELATIONS = ("invokes", "reads", "writes", "delegates", "replies", "cites", "next")
AGENT, TOOL, MEMORY, MESSAGE = range(4)
REL = {r: i for i, r in enumerate(RELATIONS)}
MAX_NODE_CHARS = 2000

_TOOL_CALL = re.compile(
    r"""^\s*(?:
        [A-Za-z][A-Za-z0-9_.]*\s*(?:\{|\(|\n\s*Action\ Input)   # Name{...}  Name(...)  Name\nAction Input
      | [A-Z][A-Za-z0-9_]+\s*:\s                                  # TerminalExecute: ...
      | [\{\[]                                                    # raw JSON arguments
      | "[a-z_]+"\s*:                                             # "command": ...
      | [A-Z]{2,}\s*<                                             # SET <Reminder>: ...
      | [a-z_]+\s*:\s*[\{\[]                                      # write_to_file: {...}
      | ```                                                       # fenced shell / code to execute
      | <                                                         # <Settings>-<Volume> ...
      | The\ action\ is\ \{                                       # GUI action record
    )""",
    re.X,
)
# Short imperative device / GUI operations ("click <x>", "Open <Gmail>", "turn on the ...").
_UI_VERB = re.compile(r"^\s*(click|open|type|search|delete|press|tap|scroll|swipe|select|turn|"
                      r"activate|set|vim|bash|at)\b", re.I)


def action_kind(action: str | None) -> str:
    """'tool' if the action string is a tool invocation, 'message' if it is text."""
    if not action or not action.strip():
        return "none"
    a = action.strip()
    if a.lower().startswith("final answer"):
        return "message"
    if _TOOL_CALL.match(a) or (len(a) < 200 and _UI_VERB.match(a)):
        return "tool"
    return "message"


@dataclass
class Trace:
    id: int
    label: int                      # 1 = unsafe (R-Judge human label)
    category: str                   # Application / Finance / IoT / Program / Web
    attack_type: str                # injection / unintended
    group: str                      # hash of the first user prompt (template id)
    node_text: list[str] = field(default_factory=list)
    node_type: list[int] = field(default_factory=list)
    edges: list[tuple[int, int, int]] = field(default_factory=list)   # (src, dst, relation)
    steps: list[tuple[int, int]] = field(default_factory=list)        # (agent node, action node)

    @property
    def n_nodes(self) -> int:
        return len(self.node_text)

    def _add(self, ntype: int, text: str | None) -> int:
        self.node_text.append((text or "").strip()[:MAX_NODE_CHARS])
        self.node_type.append(ntype)
        return len(self.node_text) - 1

    def _edge(self, src: int, dst: int, rel: str) -> None:
        assert src < dst, "edges must point forward in time"
        self.edges.append((src, dst, REL[rel]))


def _as_text(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def record_to_trace(rec: dict, category: str = "unknown") -> Trace:
    first_user = next((_as_text(m.get("content")) for turn in rec["contents"] for m in turn
                       if m.get("role") == "user" and m.get("content")), f"id-{rec['id']}")
    group = hashlib.sha1(re.sub(r"\s+", " ", first_user.lower()).encode()).hexdigest()[:12]
    tr = Trace(id=int(rec["id"]), label=int(rec["label"]), category=category,
               attack_type=rec.get("attack_type", "unknown"), group=group)

    profile = _as_text(rec.get("profile"))
    last_user = last_agent = last_action = last_reply = None
    last_action_is_tool = False
    unread_memory: list[int] = []

    for turn in rec["contents"]:
        for m in turn:
            role = m.get("role")
            if role == "user":
                text = _as_text(m.get("content"))
                if last_user is None and profile:
                    text = f"{profile}\n{text}"
                u = tr._add(MESSAGE, text)
                if last_reply is not None:
                    tr._edge(last_reply, u, "replies")
                last_user = u
            elif role == "agent":
                a = tr._add(AGENT, _as_text(m.get("thought")))
                if last_user is not None:
                    tr._edge(last_user, a, "replies")
                if last_agent is not None:
                    tr._edge(last_agent, a, "next")
                for mem in unread_memory:
                    tr._edge(mem, a, "reads")
                unread_memory = []
                last_agent = a
                action = _as_text(m.get("action"))
                kind = action_kind(action)
                if kind == "tool":
                    n = tr._add(TOOL, action)
                    tr._edge(a, n, "invokes")
                    last_action_is_tool = True
                else:                                   # reply, final answer, or empty action
                    n = tr._add(MESSAGE, action)
                    tr._edge(a, n, "replies")
                    last_action_is_tool = False
                    last_reply = n
                last_action = n
                tr.steps.append((a, n))
            elif role == "environment":
                text = _as_text(m.get("content"))
                if not text.strip():
                    continue
                mem = tr._add(MEMORY, text)
                src = last_action if (last_action_is_tool and last_action is not None) else last_agent
                if src is not None:
                    tr._edge(src, mem, "writes")
                unread_memory.append(mem)
    return tr


def load_rjudge(root: str | Path) -> list[Trace]:
    """Load every record under ``<root>/data/<Category>/*.json``, sorted by id."""
    root = Path(root)
    files = sorted((root / "data").glob("*/*.json"))
    if not files:
        raise FileNotFoundError(
            f"no R-Judge data under {root}/data. Run: git clone https://github.com/Lordog/R-Judge {root}")
    traces = []
    for f in files:
        for rec in json.loads(f.read_text(encoding="utf-8")):
            tr = record_to_trace(rec, category=f.parent.name)
            if tr.steps:
                traces.append(tr)
    traces.sort(key=lambda t: t.id)
    return traces


def rjudge_commit(root: str | Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_published_judge(root: str | Path, model: str = "Meta-Llama-3.1-8B-Instruct") -> dict[int, int]:
    """Per-record predictions of an LLM judge as released by the R-Judge authors.

    Returns {record id: pred} with pred in {1 unsafe, 0 safe, -1 unparseable}.
    These are the R-Judge authors' own runs, shipped in their repository under
    ``results/<model>/<attack_type>/results.json``; nothing is re-run here.
    """
    out: dict[int, int] = {}
    for f in sorted((Path(root) / "results" / model).glob("*/results.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            out[int(r["id"])] = int(r["pred"])
    return out


def summary(traces: list[Trace]) -> dict:
    lab = np.array([t.label for t in traces])
    kinds = np.concatenate([np.asarray(t.node_type) for t in traces])
    rels = np.concatenate([np.asarray([e[2] for e in t.edges] or [0]) for t in traces])
    return {
        "records": len(traces), "unsafe": int(lab.sum()), "safe": int((1 - lab).sum()),
        "prompt_groups": len({t.group for t in traces}),
        "decision_steps": int(sum(len(t.steps) for t in traces)),
        "nodes": int(kinds.size),
        "nodes_by_type": {NODE_TYPES[i]: int((kinds == i).sum()) for i in range(4)},
        "edges_by_relation": {RELATIONS[i]: int((rels == i).sum()) for i in range(len(RELATIONS))},
        "by_category": {c: int(sum(t.category == c for t in traces))
                        for c in sorted({t.category for t in traces})},
        "by_attack_type": {a: int(sum(t.attack_type == a for t in traces))
                           for a in sorted({t.attack_type for t in traces})},
    }
