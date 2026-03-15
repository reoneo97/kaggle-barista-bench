"""
train_contrastive.py
====================
End-to-end script that trains the BaristaContrastiveModel using InfoNCE
loss and attention-based recipe representations, then optionally runs
evaluation against the existing loss_fn metric for comparison.

Usage
-----
    python -m barista_bench.train_contrastive \\
        --train_csv data/train.csv \\
        --epochs 10 \\
        --batch_size 32 \\
        --backbone distilbert-base-uncased \\
        --output_dir checkpoints/

Architecture recap
------------------
                 ┌─────────────────────────────────────────────────┐
  "I'd like an  │  OrderEncoder                                    │
  oat milk      │  DistilBERT backbone → mean pool → linear proj  │──→  order_emb (D)
  grande latte" └─────────────────────────────────────────────────┘
                           ↕  InfoNCE contrastive loss
                 ┌─────────────────────────────────────────────────┐
  [CLS] Latte   │  RecipeEncoder                                   │
  Grande        │  token embed + type embed + pos embed            │
  Oat Milk      │  → N × Multi-head Self-Attention layers         │──→  recipe_emb (D)
                └─────────────────────────────────────────────────┘

Both encoders are jointly optimised so that (order, recipe) positive pairs
are closest in the shared embedding space, while all other in-batch pairs
(negatives) are pushed apart.
"""

import argparse
import logging

import pandas as pd

from barista_bench.models.contrastive import BaristaContrastiveModel
from barista_bench.models.order_encoder import OrderEncoder, OrderTokenizer
from barista_bench.models.recipe_encoder import RecipeEncoder
from barista_bench.train.dataset import ContrastiveOrderDataset
from barista_bench.train.trainer import ContrastiveTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BaristaContrastiveModel with InfoNCE")
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--num_attn_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--backbone", default="distilbert-base-uncased",
                   help="HuggingFace model name for OrderEncoder backbone")
    p.add_argument("--freeze_backbone", action="store_true", default=True)
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.train_csv)

    tokenizer = OrderTokenizer(model_name=args.backbone, max_length=128)

    dataset = ContrastiveOrderDataset(df, tokenizer)
    logging.info(f"Dataset size: {len(dataset)} (order, recipe) pairs")

    # ── Model ─────────────────────────────────────────────────────────────
    order_enc = OrderEncoder(
        model_name=args.backbone,
        embedding_dim=args.embedding_dim,
        freeze_backbone=args.freeze_backbone,
    )
    recipe_enc = RecipeEncoder(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_attn_layers,
    )
    model = BaristaContrastiveModel(
        order_encoder=order_enc,
        recipe_encoder=recipe_enc,
        temperature=args.temperature,
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {trainable:,}")

    # ── Training ──────────────────────────────────────────────────────────
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    trainer = ContrastiveTrainer(model, dataset, tokenizer, config)
    trainer.train()


if __name__ == "__main__":
    main()
