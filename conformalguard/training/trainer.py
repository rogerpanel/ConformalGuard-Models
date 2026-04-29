"""Training loop for the DyGFormer-HGT encoder + linear safety head.

Hyper-parameters from the paper (Sec. 5):
    optimizer      : AdamW
    lr             : 5e-4
    weight_decay   : 0.01
    epochs         : 30
    schedule       : cosine annealing
    batch_size     : 64 traces (variable-size graphs)
    grad_clip      : 1.0
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from conformalguard.conformal.nonconformity import SafetyHead
from conformalguard.data.agentchain26 import AgentChainTrace
from conformalguard.encoder.dygformer_hgt import DyGFormerHGT, EncoderConfig
from conformalguard.training.losses import safety_loss
from conformalguard.violations.classes import VIOLATION_CLASSES


@dataclass
class TrainerConfig:
    lr: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 30
    batch_size: int = 64
    grad_clip: float = 1.0
    coverage_weight: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    log_every: int = 50
    output_dir: str = "runs/conformalguard"
    encoder: EncoderConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = EncoderConfig()


class Trainer:
    def __init__(self, cfg: TrainerConfig | None = None):
        self.cfg = cfg or TrainerConfig()
        torch.manual_seed(self.cfg.seed)
        self.encoder = DyGFormerHGT(self.cfg.encoder).to(self.cfg.device)
        self.head = SafetyHead(
            d_model=self.cfg.encoder.d_model,
            vocab_size=len(VIOLATION_CLASSES),
        ).to(self.cfg.device)
        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.opt, T_max=self.cfg.epochs)
        self.tau_hat: float = 1.5  # EMA threshold estimate

    # --------------------------------------------------------------------
    def _embed_batch(self, traces: list[AgentChainTrace]) -> tuple[torch.Tensor, torch.Tensor]:
        embs = []
        for t in traces:
            embs.append(self.encoder(self.encoder.tensorize(t.graph)))
        h = torch.stack(embs)
        y = torch.tensor([t.label for t in traces], dtype=torch.long, device=h.device)
        return h, y

    # --------------------------------------------------------------------
    def fit(self, train_traces: Iterable[AgentChainTrace],
            val_traces: Iterable[AgentChainTrace] | None = None) -> dict:
        train_traces = list(train_traces)
        val_traces = list(val_traces) if val_traces is not None else []
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        history = []

        for epoch in range(self.cfg.epochs):
            self.encoder.train(); self.head.train()
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()
            for i in range(0, len(train_traces), self.cfg.batch_size):
                batch = train_traces[i:i + self.cfg.batch_size]
                h, y = self._embed_batch(batch)
                logits = self.head(h)
                loss = safety_loss(
                    logits, y,
                    coverage_weight=self.cfg.coverage_weight,
                    threshold_estimate=self.tau_hat,
                )
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.head.parameters()),
                    self.cfg.grad_clip,
                )
                self.opt.step()
                # Update τ̂ (EMA over true-label nonconformity).
                with torch.no_grad():
                    log_p = torch.log_softmax(logits, dim=-1)
                    s_true = -log_p.gather(-1, y.view(-1, 1)).squeeze(-1)
                    tau_batch = float(s_true.quantile(0.95).item())
                    self.tau_hat = 0.95 * self.tau_hat + 0.05 * tau_batch
                epoch_loss += float(loss.item())
                n_batches += 1
                if (i // self.cfg.batch_size) % self.cfg.log_every == 0:
                    print(f"epoch {epoch:02d}  step {i:05d}  "
                          f"loss {loss.item():.4f}  τ̂ {self.tau_hat:.3f}")
            self.scheduler.step()

            val = self._eval(val_traces) if val_traces else {}
            row = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, n_batches),
                "tau_hat": self.tau_hat,
                "lr": self.opt.param_groups[0]["lr"],
                "epoch_time_s": time.time() - t0,
                **{f"val_{k}": v for k, v in val.items()},
            }
            history.append(row)
            self._save_checkpoint(out_dir / f"epoch_{epoch:02d}.pt")
            with (out_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2)
            print(json.dumps(row))

        self._save_checkpoint(out_dir / "model.pt")
        return {"history": history, "config": asdict(self.cfg)}

    # --------------------------------------------------------------------
    @torch.no_grad()
    def _eval(self, traces: list[AgentChainTrace]) -> dict:
        if not traces:
            return {}
        self.encoder.eval(); self.head.eval()
        h, y = self._embed_batch(traces)
        logits = self.head(h)
        loss = safety_loss(logits, y).item()
        acc = float((logits.argmax(-1) == y).float().mean().item())
        return {"loss": loss, "acc": acc}

    def _save_checkpoint(self, path: Path) -> None:
        torch.save({
            "encoder": self.encoder.state_dict(),
            "head":    self.head.state_dict(),
            "cfg":     asdict(self.cfg),
            "tau_hat": self.tau_hat,
        }, path)
