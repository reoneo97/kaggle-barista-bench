"""
ContrastiveOrderDataset
========================
Converts the Kaggle Barista Bench training CSV into (order text, recipe
token-ID sequence) positive pairs suitable for InfoNCE training.

Each row in the CSV may contain multiple ordered items.  We expand each
row into one sample per item, so that every (order_text, single_item)
pair is treated as a positive.

This multi-instance expansion is intentional:
  - The same order text will appear multiple times if the customer ordered
    several items.  The InfoNCE loss will then pull the order embedding
    towards *all* ordered items' recipe embeddings, which is the correct
    multi-label behaviour without needing explicit multi-label loss heads.
  - In-batch negatives are other items from *other* orders, so there is
    minimal false-negative risk (ordering a Latte does not make Cappuccino
    a true negative — but they come from different rows, so are treated as
    negatives with only small probability of collision).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..models.recipe_encoder import recipe_to_token_ids
from ..schema import Order, OrderList


class ContrastiveOrderDataset(Dataset):
    """Pairs of (order_text, recipe_token_ids) for contrastive training.

    Parameters
    ----------
    df:
        DataFrame with at minimum columns ``order`` (str) and
        ``expected_json`` (JSON string parseable as an ``OrderList``).
    tokenizer:
        An ``OrderTokenizer`` (or any callable) that maps a list of
        strings to ``{"input_ids": Tensor, "attention_mask": Tensor}``.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.samples: list[tuple[str, list[int]]] = []  # (order_text, token_ids)
        self._build(df)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            order_text: str = row["order"]
            try:
                order_list = OrderList.model_validate_json(row["expected_json"])
            except Exception:
                continue  # Skip malformed rows

            for item in order_list.items:
                token_ids = self._item_to_token_ids(item)
                self.samples.append((order_text, token_ids))

    @staticmethod
    def _item_to_token_ids(item: Order) -> list[int]:
        return recipe_to_token_ids(
            item_name=item.name,
            size=item.size,
            modifiers=item.modifiers,
        )

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        order_text, recipe_token_ids = self.samples[idx]
        return {
            "order_text": order_text,
            "recipe_token_ids": torch.tensor(recipe_token_ids, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], tokenizer: Any) -> dict[str, torch.Tensor]:
        """Custom collate that tokenises order texts on-the-fly.

        Typically wrapped with ``functools.partial`` when passed to
        ``DataLoader(collate_fn=...)``.
        """
        order_texts = [b["order_text"] for b in batch]
        recipe_token_ids = torch.stack([b["recipe_token_ids"] for b in batch])  # (B, L)

        encoded_orders = tokenizer(order_texts)  # dict with input_ids, attention_mask

        return {
            "order_input_ids": encoded_orders["input_ids"],
            "order_attention_mask": encoded_orders["attention_mask"],
            "recipe_token_ids": recipe_token_ids,
        }
