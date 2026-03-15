"""
ContrastiveTrainer
==================
Training loop for the BaristaContrastiveModel using InfoNCE loss.

Design notes
------------
* **Gradient accumulation** is supported so that larger effective batch
  sizes (important for InfoNCE — more in-batch negatives = better signal)
  can be used even on small GPUs.
* **Cosine LR schedule with warm-up** is used: LR increases linearly for
  the first ``warmup_steps`` then decays following a cosine curve.
* **Validation** computes mean InfoNCE loss on a held-out split and also
  reports Recall@1 (the fraction of orders for which the correct recipe
  is the nearest neighbour) as an interpretable downstream metric.
* **Checkpointing** saves the best model weights (by validation loss) to
  disk.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from ..models.contrastive import BaristaContrastiveModel
from .dataset import ContrastiveOrderDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for a contrastive training run."""

    # Optimisation
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    accumulation_steps: int = 2     # effective batch = batch_size × accumulation_steps

    # Learning-rate schedule
    warmup_steps: int = 100

    # Data split
    val_fraction: float = 0.1
    num_workers: int = 0            # Set > 0 only if data loading is a bottleneck

    # Checkpointing
    output_dir: str = "checkpoints"
    save_best: bool = True

    # Logging
    log_every_n_steps: int = 10


class ContrastiveTrainer:
    """Manages the full training and validation loop.

    Parameters
    ----------
    model:
        Fully constructed ``BaristaContrastiveModel``.
    dataset:
        ``ContrastiveOrderDataset`` built from the training CSV.
    tokenizer:
        The ``OrderTokenizer`` used to batch-encode order strings.
    config:
        ``TrainingConfig`` with all hyperparameters.
    device:
        ``torch.device`` (defaults to CUDA if available, else CPU).
    """

    def __init__(
        self,
        model: BaristaContrastiveModel,
        dataset: ContrastiveOrderDataset,
        tokenizer,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        collate = functools.partial(ContrastiveOrderDataset.collate_fn, tokenizer=tokenizer)

        # Train / validation split
        val_size = max(1, int(len(dataset) * config.val_fraction))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate,
            drop_last=True,  # Consistent batch size is crucial for InfoNCE
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate,
        )

        # Optimizer — separate LR for backbone vs. projection / recipe encoder
        backbone_params = list(model.order_encoder.backbone.parameters())
        other_params = [
            p for p in model.parameters()
            if not any(p is bp for bp in backbone_params)
        ]
        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": config.learning_rate * 0.1},
                {"params": other_params,   "lr": config.learning_rate},
            ],
            weight_decay=config.weight_decay,
        )

        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - config.warmup_steps
        )

        self._global_step = 0
        self._best_val_loss = float("inf")
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Warmup ────────────────────────────────────────────────────────────

    def _warmup_lr(self) -> None:
        """Linearly scale LR from 0 during the first ``warmup_steps``."""
        if self._global_step < self.config.warmup_steps:
            scale = self._global_step / max(1, self.config.warmup_steps)
            for i, pg in enumerate(self.optimizer.param_groups):
                base_lr = self.config.learning_rate if i == 1 else self.config.learning_rate * 0.1
                pg["lr"] = base_lr * scale

    # ── Training step ─────────────────────────────────────────────────────

    def _train_step(self, batch: dict) -> float:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(
            order_input_ids=batch["order_input_ids"],
            order_attention_mask=batch["order_attention_mask"],
            recipe_token_ids=batch["recipe_token_ids"],
        )
        loss = out["loss"] / self.config.accumulation_steps
        loss.backward()
        return loss.item() * self.config.accumulation_steps

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            loss_val = self._train_step(batch)
            total_loss += loss_val

            # Gradient accumulation flush
            if (step + 1) % self.config.accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self._warmup_lr()
                self.optimizer.step()
                if self._global_step >= self.config.warmup_steps:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self._global_step += 1

            if self._global_step % self.config.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[1]["lr"]
                temp = self.model.loss_fn.temperature.item()
                logger.info(
                    f"Epoch {epoch+1} | step {self._global_step} "
                    f"| loss {loss_val:.4f} | lr {lr:.2e} | τ {temp:.4f}"
                )

        return total_loss / len(self.train_loader)

    # ── Validation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Return (mean_val_loss, recall_at_1)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(
                order_input_ids=batch["order_input_ids"],
                order_attention_mask=batch["order_attention_mask"],
                recipe_token_ids=batch["recipe_token_ids"],
            )
            total_loss += out["loss"].item()

            # Recall@1: is the most similar recipe the correct one?
            sims = out["order_embs"] @ out["recipe_embs"].T  # (B, B)
            preds = sims.argmax(dim=1)                       # (B,)
            labels = torch.arange(len(preds), device=self.device)
            correct += (preds == labels).sum().item()
            total += len(preds)

        mean_loss = total_loss / len(self.val_loader)
        recall_at_1 = correct / max(1, total)
        return mean_loss, recall_at_1

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _maybe_save(self, val_loss: float) -> None:
        if self.config.save_best and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            path = Path(self.config.output_dir) / "best_model.pt"
            torch.save(self.model.state_dict(), path)
            logger.info(f"  Saved best model to {path} (val_loss={val_loss:.4f})")

    # ── Main entry point ──────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop for ``config.epochs`` epochs."""
        logger.info(
            f"Training on {self.device} | "
            f"{len(self.train_loader.dataset)} train / "
            f"{len(self.val_loader.dataset)} val samples"
        )
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss, recall = self._validate()
            self._maybe_save(val_loss)
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"recall@1={recall:.3f}"
            )
        logger.info("Training complete.")
