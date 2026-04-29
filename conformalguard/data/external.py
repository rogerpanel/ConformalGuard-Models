"""Registry of external corpora used for cross-distribution evaluation.

These datasets are consumed by the OOD generalization study (paper Sec. 5.3)
and by `scripts/download_datasets.py`. We do not ship the data; we provide
canonical download URLs and licensing notes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExternalDataset:
    name: str
    purpose: str
    url: str
    license: str
    citation: str
    expected_files: tuple[str, ...] = ()


EXTERNAL_DATASETS: list[ExternalDataset] = [
    ExternalDataset(
        name="ToolEmu",
        purpose="Out-of-distribution agent traces for OOD generalization study",
        url="https://github.com/ryoungj/ToolEmu",
        license="Apache-2.0",
        citation="Ruan et al., Identifying the Risks of LM Agents with an LM-Emulated Sandbox, 2024",
    ),
    ExternalDataset(
        name="AgentBench",
        purpose="Cross-platform agent task suite",
        url="https://github.com/THUDM/AgentBench",
        license="Apache-2.0",
        citation="Liu et al., AgentBench: Evaluating LLMs as Agents, ICLR 2024",
    ),
    ExternalDataset(
        name="TrustLLM",
        purpose="Safety alignment evaluation prompts",
        url="https://github.com/HowieHwong/TrustLLM",
        license="MIT",
        citation="Sun et al., TrustLLM, 2024",
    ),
    ExternalDataset(
        name="InjecAgent",
        purpose="Indirect prompt-injection corpus",
        url="https://github.com/uiuc-kang-lab/InjecAgent",
        license="Apache-2.0",
        citation="Zhan et al., InjecAgent, ACL 2024",
    ),
    ExternalDataset(
        name="HarmBench",
        purpose="Red-team harmful behavior prompts",
        url="https://github.com/centerforaisafety/HarmBench",
        license="MIT",
        citation="Mazeika et al., HarmBench, ICML 2024",
    ),
    ExternalDataset(
        name="AdvBench",
        purpose="Universal harmful prompt suite for jailbreak evaluation",
        url="https://github.com/llm-attacks/llm-attacks",
        license="MIT",
        citation="Zou et al., Universal and Transferable Adversarial Attacks on Aligned LLMs, 2023",
    ),
    ExternalDataset(
        name="BeaverTails",
        purpose="Safety preference + classification dataset",
        url="https://huggingface.co/datasets/PKU-Alignment/BeaverTails",
        license="CC-BY-NC-4.0",
        citation="Ji et al., BeaverTails, NeurIPS 2023",
    ),
    ExternalDataset(
        name="RobustIDPS-PCAPs",
        purpose="Network-IDPS co-evaluation (cross-stack)",
        url="https://doi.org/10.5281/zenodo.19129512",
        license="CC-BY-4.0",
        citation="Anaedevha and Trofimov, RobustIDPS.ai, 2026",
    ),
    ExternalDataset(
        name="PQC-Traffic",
        purpose="Post-quantum handshake captures used by PQC-IDS multi-agent baseline",
        url="https://doi.org/10.34740/kaggle/dsv/15424420",
        license="CC-BY-4.0",
        citation="Anaedevha and Trofimov, PQC Traffic Lab, 2026",
    ),
    ExternalDataset(
        name="CICIDS2017",
        purpose="Auxiliary network traffic baseline used by RobustIDPS plug-in tests",
        url="https://www.unb.ca/cic/datasets/ids-2017.html",
        license="academic-use",
        citation="Sharafaldin et al., CICIDS2017, 2018",
    ),
    ExternalDataset(
        name="UNSW-NB15",
        purpose="Auxiliary network IDS evaluation",
        url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        license="academic-use",
        citation="Moustafa and Slay, UNSW-NB15, 2015",
    ),
]


def find(name: str) -> ExternalDataset:
    for d in EXTERNAL_DATASETS:
        if d.name.lower() == name.lower():
            return d
    raise KeyError(name)
