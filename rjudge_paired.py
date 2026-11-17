"""Paired per-split comparison of two scorers from an existing results.json.

    python scripts/rjudge_paired.py runs/rjudge_st/results.json seq step block@0.10 [random|grouped]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.rjudge.experiment import paired  # noqa: E402

path, a, b, metric = sys.argv[1:5]
protocol = sys.argv[5] if len(sys.argv) > 5 else "random"
rows = json.loads(Path(path).read_text())["per_split"][protocol]
print(json.dumps({"protocol": protocol, "a": a, "b": b, "metric": metric, **paired(rows, a, b, metric)}, indent=1))
