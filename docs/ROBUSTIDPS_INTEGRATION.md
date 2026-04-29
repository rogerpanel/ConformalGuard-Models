# Integration with RobustIDPS.ai

ConformalGuard ships as the **agent-layer safety module** of the broader
RobustIDPS.ai platform. This document describes the wiring.

## Topology

```
                          ┌──────────────────────────┐
                          │  Multi-agent application │  (AutoGen / LangGraph / …)
                          └────────────┬─────────────┘
                                       │ instrumentation hooks
                                       ▼
                       ┌─────────────────────────────────┐
                       │  conformalguard.deployment      │
                       │  StreamProcessor + Runtime      │
                       └────────────┬───────┬────────────┘
                                    │       │
                       /trace/event │       │ /trace/step
                                    ▼       ▼
                  ┌──────────────────────────────────────────┐
                  │  RobustIDPS.ai backend (FastAPI / Flask) │
                  │  - 60+ REST endpoints                    │
                  │  - WebSocket: agent.safety.events        │
                  │  - Celery: async investigation jobs      │
                  │  - PostgreSQL: trace store + audit log   │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │  Frontend SPA (React 18 PWA)             │
                  │  - SOC analyst review queue              │
                  │  - Alert causality / investigation pages │
                  │  - LLM Attack Surfaces dashboards        │
                  └──────────────────────────────────────────┘
```

## RobustIDPS plug-in

ConformalGuard exposes itself to RobustIDPS through the FastAPI server in
`conformalguard.deployment.server`. The companion repository
[github.com/rogerpanel/robustidps.ai](https://github.com/rogerpanel/robustidps.ai)
loads it via the entry point `integrated_ai_ids.plugins.conformalguard`,
mirroring the convention already used for the SurrogateIDS, MambaShield,
and FedGTD models.

Wiring at the platform level:

```python
# integrated_ai_ids/plugins/conformalguard.py
from conformalguard.deployment.guard import ConformalGuardRuntime
from conformalguard.deployment.routing import AnalystRouter
from conformalguard.deployment.stream_processor import StreamProcessor

runtime = ConformalGuardRuntime.load("artifacts/conformalguard/model.pt")
processor = StreamProcessor(runtime, AnalystRouter())

def on_agent_event(event):           # called from RobustIDPS WebSocket bus
    processor.ingest(event)

def on_step(t, proposed_action_id, top_violation_id):
    rec = processor.step(t, proposed_action_id, top_violation_id)
    return rec.decision
```

## Topic mapping

| RobustIDPS WebSocket topic           | ConformalGuard endpoint     |
|--------------------------------------|-----------------------------|
| `agent.invocation.started`           | `POST /trace/event` (kind=agent)   |
| `agent.tool.invoked`                 | `POST /trace/event` (kind=tool, +invoke edge) |
| `agent.memory.write`                 | `POST /trace/event` (kind=memory, +write edge) |
| `agent.message.sent`                 | `POST /trace/event` (kind=message, +reply edge) |
| `agent.step.candidate`               | `POST /trace/step`          |

## Cross-page surfaces

| RobustIDPS page                         | Role of ConformalGuard                                  |
|-----------------------------------------|---------------------------------------------------------|
| Live Monitoring (AI Command Center)     | Verdict stream + per-step latency / α_eff               |
| Multi-Agent Attacks (LLM Attack Surfaces)| Adversarial regression suite using the same encoder    |
| Auto-Investigation (SOC Intelligence)   | Routed REVIEW items become investigation tickets         |
| MITRE ATT&CK Mapper                     | 11 violation classes mapped onto agent-relevant ATT&CK |
| OWASP LLM Top-10 Scorecard              | Aligns to LLM01-Prompt Injection, LLM05-Improper Tool, … |
| AI Data & Models                        | Lists the AgentChain-26 dataset and DyGFormer-HGT model |

## Production performance (paper Sec. 6.3)

Over a 30-day window with 1.84 M agent steps:

| Metric             | Value      |
|--------------------|-----------:|
| Precision          | 97.1 %     |
| Recall             | 94.6 %     |
| Avg. latency       | 24.7 ms    |
| Auto-block share   | 71.3 %     |
| Review-queue share | 17.5 %     |
| Allow share        | 11.2 %     |

These numbers are tracked in real time via the `/queue` endpoint and the
`agent.safety.events` WebSocket topic.
