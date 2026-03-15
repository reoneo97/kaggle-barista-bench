"""
RecipeEncoder
=============
Encodes a structured recipe (item + optional size + modifiers) into a
fixed-size embedding using learned token embeddings and multi-head
self-attention.

Architecture
------------
Input  :  token IDs   [CLS, item, size?, mod1, mod2, ...]  (shape B × L)
          token types  (item / size / modifier / special)   (shape B × L)

1. Token embedding   : vocab_size  → d_model
2. Token-type embed  : 4 types     → d_model  (added to token embeddings)
3. N × TransformerEncoderLayer  (multi-head self-attention + FFN)
4. Take [CLS] position output    → recipe embedding  (shape B × d_model)
5. Linear projection             → embedding_dim

Why attention?
--------------
A simple sum-of-embeddings cannot capture that *"No Whip" on a Mocha
cancels the default Whip*, or that *Oat Milk replacing Whole Milk costs
extra*.  Self-attention lets the modifier tokens condition on the item
token (and each other) so the model learns these interactions from data.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .menu_vocab import (
    CLS_ID,
    MAX_RECIPE_LEN,
    PAD_ID,
    VOCAB_SIZE,
    token_type,
)


class RecipeEncoder(nn.Module):
    """Attention-based encoder that maps a structured recipe to an embedding.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the output recipe embedding (and d_model inside
        the transformer).
    num_heads:
        Number of attention heads in each TransformerEncoderLayer.
    num_layers:
        Number of stacked TransformerEncoderLayer blocks.
    ffn_dim:
        Hidden size of the position-wise feed-forward network.
    dropout:
        Dropout probability applied inside the transformer layers.
    """

    NUM_TOKEN_TYPES = 4  # item, size, modifier, special

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # ── Embeddings ────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_ID)
        self.type_emb = nn.Embedding(self.NUM_TOKEN_TYPES, embedding_dim)
        self.pos_emb = nn.Embedding(MAX_RECIPE_LEN, embedding_dim)

        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # ── Transformer encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for embedding and linear layers."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    @staticmethod
    def _build_type_ids(token_ids: torch.Tensor) -> torch.Tensor:
        """Construct a token-type tensor from raw token IDs (CPU-safe)."""
        type_ids = torch.zeros_like(token_ids)
        for b in range(token_ids.size(0)):
            for t in range(token_ids.size(1)):
                type_ids[b, t] = token_type(token_ids[b, t].item())
        return type_ids

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : LongTensor of shape (B, L)
            Each row is a recipe token sequence beginning with CLS_ID,
            padded to the same length with PAD_ID.

        Returns
        -------
        Tensor of shape (B, embedding_dim)
            The [CLS] position output — the recipe representation.
        """
        B, L = token_ids.shape
        device = token_ids.device

        # Token type IDs derived from vocabulary membership
        type_ids = self._build_type_ids(token_ids).to(device)

        # Positional IDs  0, 1, 2, …, L-1
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Combined embedding
        x = self.token_emb(token_ids) + self.type_emb(type_ids) + self.pos_emb(pos_ids)
        x = self.emb_dropout(self.emb_norm(x))

        # Padding mask — True where the token is padding (transformer ignores those)
        padding_mask = token_ids == PAD_ID  # (B, L), True → ignore

        # Self-attention over all recipe tokens
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, L, d)

        # [CLS] token is always at position 0
        cls_out = x[:, 0, :]  # (B, d)
        return cls_out


# ── Recipe → token-ID sequence conversion ────────────────────────────────────

from .menu_vocab import SIZES, token_to_id  # noqa: E402  (avoid circular at top)


def recipe_to_token_ids(
    item_name: str,
    size: str | None,
    modifiers: list[str],
    max_len: int = MAX_RECIPE_LEN,
) -> list[int]:
    """Convert a structured recipe into a padded token-ID sequence.

    Sequence layout:
        [CLS]  item  [size]  mod_1  mod_2  ...  [PAD] × k

    Parameters
    ----------
    item_name:
        Canonical item name from the menu (must be in ``ALL_ITEMS``).
    size:
        Size string or ``None`` for food items.
    modifiers:
        List of modifier strings (each must be in ``MODIFIERS``).
    max_len:
        Total sequence length (padded or truncated to this).

    Returns
    -------
    list[int]
        Token IDs of length ``max_len``.
    """
    ids: list[int] = [CLS_ID, token_to_id(item_name)]

    if size is not None and size in SIZES:
        ids.append(token_to_id(size))

    for mod in modifiers:
        ids.append(token_to_id(mod))
        if len(ids) >= max_len:
            break

    # Pad to max_len
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids
