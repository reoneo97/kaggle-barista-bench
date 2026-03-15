"""
OrderEncoder
============
Encodes a raw natural-language customer order string into a fixed-size
embedding that lives in the same space as RecipeEncoder outputs.

Architecture
------------
1. A pretrained HuggingFace transformer (default: distilbert-base-uncased)
   is used as the text backbone.  Only the final hidden states are used.
2. Mean pooling over non-padding tokens produces a sentence-level vector.
3. A linear projection maps that vector to ``embedding_dim``.

Why a pretrained backbone?
--------------------------
Customer orders contain rich natural-language variation ("lemme get",
"can I have", "I'd like", etc.) that a randomly-initialised model would
need many samples to learn.  Starting from a pretrained language model
provides strong priors for free; we only fine-tune the projection head
(or optionally the full backbone) via InfoNCE.

The projection head ensures the order embedding lives in the same
``embedding_dim``-dimensional space as the RecipeEncoder output, which is
a prerequisite for computing cosine similarity between the two.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class OrderEncoder(nn.Module):
    """Encodes free-text orders into a fixed-size embedding.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the text backbone.
    embedding_dim:
        Output embedding dimension — **must match** RecipeEncoder's.
    max_length:
        Maximum tokenisation length (longer orders are truncated).
    freeze_backbone:
        If ``True``, only the projection head is trained; the pretrained
        transformer weights are frozen.  Useful when training data is
        scarce (< 1 000 samples).  Set to ``False`` for full fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 128,
        max_length: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        backbone_dim = self.backbone.config.hidden_size

        # Project backbone output → shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, embedding_dim),
        )

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling over the sequence dimension."""
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)           # (B, d)
        counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : LongTensor (B, L)
        attention_mask : LongTensor (B, L)

        Returns
        -------
        Tensor (B, embedding_dim)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state          # (B, L, backbone_dim)
        pooled = self._mean_pool(hidden, attention_mask)  # (B, backbone_dim)
        return self.projection(pooled)              # (B, embedding_dim)


# ── Tokenisation helper ───────────────────────────────────────────────────────

class OrderTokenizer:
    """Thin wrapper around AutoTokenizer for batch-encoding order strings."""

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, orders: list[str]) -> dict[str, torch.Tensor]:
        """Tokenise a list of order strings, return input_ids & attention_mask."""
        return self.tokenizer(
            orders,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
