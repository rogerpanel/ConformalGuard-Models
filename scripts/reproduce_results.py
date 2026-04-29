"""One-shot reproduction of every experiment reported in the paper.

Pipeline:
    1.  Materialize AgentChain-26 (curated or synthetic).
    2.  Train DyGFormer-HGT + head.
    3.  Calibrate the conformal head.
    4.  Evaluate against the 7 baselines (Table 2).
    5.  Run the streaming Gibbs-Candès experiment (Sec. 5.4).

Run:
    python scripts/reproduce_results.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT.parent)


def main():
    py = sys.executable
    _run([py, str(ROOT / "download_datasets.py")])
    _run([py, str(ROOT / "train.py"),     "--config", "configs/encoder.yaml"])
    _run([py, str(ROOT / "calibrate.py"), "--config", "configs/conformal.yaml"])
    _run([py, str(ROOT / "evaluate.py"),  "--config", "configs/default.yaml"])
    _run([py, str(ROOT / "run_inference.py"),
          "--config", "configs/adaptive.yaml"])
    print("\nAll done. See runs/conformalguard/{report,online}.json")


if __name__ == "__main__":
    main()
