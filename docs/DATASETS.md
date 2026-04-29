# Datasets

## Primary benchmark — AgentChain-26

| Attribute      | Value                                                          |
|----------------|----------------------------------------------------------------|
| Size           | 12,400 multi-agent execution traces                            |
| Trace length   | 5–50 steps (mean 19.4)                                         |
| Platform mix   | AutoGen 30 % / MetaGPT 30 % / LangGraph 25 % / CrewAI 15 %     |
| Labels         | `safe` + 11 critical violation classes                         |
| Splits         | 80 / 10 / 10 train / val / test (stratified by platform+label) |
| Distribution   | JSONL — one trace per line; see `AgentChainTrace.to_dict`      |
| Curated DOI    | (TBA — pending Zenodo deposit)                                 |
| Mirror         | `github.com/rogerpanel/CV/releases/tag/agentchain26-v1`        |

When the curated archive is not present locally, the loader transparently
falls back to a deterministic procedural generator
(`conformalguard.data.synthetic`) that reproduces the marginal statistics of
the benchmark. This makes every test in `tests/` runnable on a fresh machine
without any download.

### Eleven critical violation classes

| Id | Name                       | Severity  |
|---:|----------------------------|-----------|
| 0  | safe                       | low       |
| 1  | pii_exfiltration           | critical  |
| 2  | prompt_injection           | critical  |
| 3  | unauthorized_tool_call     | critical  |
| 4  | memory_poisoning           | critical  |
| 5  | capability_escalation      | critical  |
| 6  | trust_delegation_abuse     | high      |
| 7  | data_exfiltration_via_tool | critical  |
| 8  | jailbreak_persona          | high      |
| 9  | rag_poisoning              | high      |
| 10 | orchestrator_compromise    | critical  |
| 11 | consensus_manipulation     | high      |

## Out-of-distribution / cross-stack evaluation

The OOD-generalization study (paper Sec. 5.3) draws from these external
corpora. We do not redistribute them; URLs and licenses are tracked in
`conformalguard.data.external.EXTERNAL_DATASETS`.

| Dataset           | Use                                       | Source                                                     |
|-------------------|-------------------------------------------|------------------------------------------------------------|
| ToolEmu           | Sandbox agent traces (LM-emulated)        | https://github.com/ryoungj/ToolEmu                         |
| AgentBench        | Cross-platform agent task suite           | https://github.com/THUDM/AgentBench                        |
| TrustLLM          | Safety alignment evaluation prompts       | https://github.com/HowieHwong/TrustLLM                     |
| InjecAgent        | Indirect prompt-injection corpus          | https://github.com/uiuc-kang-lab/InjecAgent                |
| HarmBench         | Red-team harmful behaviors                | https://github.com/centerforaisafety/HarmBench             |
| AdvBench          | Universal harmful prompts (jailbreak)     | https://github.com/llm-attacks/llm-attacks                 |
| BeaverTails       | Safety preference / classification        | https://huggingface.co/datasets/PKU-Alignment/BeaverTails  |

## RobustIDPS network co-evaluation

Network-IDPS-side evaluation reuses datasets from the upstream RobustIDPS.ai
release. They are **not** required to reproduce the AgentChain-26 results,
but enable the *cross-stack* experiments (Sec. 6.4).

| Dataset             | Use                                  | Source                                                       |
|---------------------|--------------------------------------|--------------------------------------------------------------|
| RobustIDPS PCAPs    | Crafted / drift / OOD network flows  | https://doi.org/10.5281/zenodo.19129512                      |
| PQC Traffic         | Post-quantum handshake captures      | https://doi.org/10.34740/kaggle/dsv/15424420                 |
| CICIDS2017          | 15 attack classes baseline           | https://www.unb.ca/cic/datasets/ids-2017.html                |
| CIC-IoT-2023        | 33 IoT attack types                  | https://www.unb.ca/cic/datasets/iotdataset-2023.html         |
| UNSW-NB15           | 9 attack categories, 49 features     | https://research.unsw.edu.au/projects/unsw-nb15-dataset      |
| NSL-KDD             | 4 attack categories                  | https://www.unb.ca/cic/datasets/nsl.html                     |
| CIC-DDoS-2019       | Volumetric / app-layer DDoS          | https://www.unb.ca/cic/datasets/ddos-2019.html               |

## Materialization / preparation

```bash
# Generates the synthetic AgentChain-26 manifest under
# ~/.cache/conformalguard/agentchain26/{train,val,test}.jsonl
python scripts/download_datasets.py

# Or programmatically:
python - <<'PY'
from conformalguard.data.agentchain26 import AgentChain26
AgentChain26(split="train").save_jsonl("agentchain26/train.jsonl", n=9920)
AgentChain26(split="val").save_jsonl("agentchain26/val.jsonl",   n=1240)
AgentChain26(split="test").save_jsonl("agentchain26/test.jsonl", n=1240)
PY
```
