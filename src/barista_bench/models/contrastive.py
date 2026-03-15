"""
InfoNCE loss and BaristaContrastiveModel
=========================================

InfoNCE (Noise-Contrastive Estimation with mutual information maximisation)
trains the two encoders to embed matching (order, recipe) pairs close
together while pushing non-matching pairs apart.

Loss formulation
----------------
Given a batch of B positive (order_i, recipe_i) pairs:

    sim_matrix[i, j] = cosine_similarity(order_emb_i, recipe_emb_j)
                       / temperature

    L_order   = CrossEntropy(sim_matrix,       labels=diag)   # rows
    L_recipe  = CrossEntropy(sim_matrix.T,     labels=diag)   # columns
    L_InfoNCE = (L_order + L_recipe) / 2

Every off-diagonal entry in the batch acts as an **in-batch negative**.
With a batch size of 64 we get 63 negatives per anchor for free —
no explicit negative mining is required, unlike triplet loss.

Why InfoNCE over alternatives?
-------------------------------
* **vs. Cross-entropy on item classification**: Multi-label structure is
  hard to handle; InfoNCE works naturally with arbitrary positive sets.
* **vs. Triplet loss**: InfoNCE is softer — it considers ALL negatives
  simultaneously, giving a more informative gradient signal.
* **vs. BPR / hinge loss**: Temperature τ gives a single, interpretable
  hyperparameter to control the sharpness of the similarity distribution.

BaristaContrastiveModel
-----------------------
Wraps OrderEncoder + RecipeEncoder and exposes a single forward pass that
returns L2-normalised embeddings ready for cosine similarity comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .order_encoder import OrderEncoder
from .recipe_encoder import RecipeEncoder


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (CLIP-style) contrastive loss.

    Parameters
    ----------
    temperature:
        Scaling factor τ for the logits.  Smaller values make the
        distribution sharper and the loss more discriminative.
        Typical range: 0.05 – 0.20.  Learnable by default.
    learnable_temperature:
        If ``True``, τ is treated as a trainable scalar (initialised to
        the provided ``temperature`` value).  This lets the model adapt
        the sharpness during training.
    """

    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = True) -> None:
        super().__init__()
        if learnable_temperature:
            self.log_temp = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer("log_temp", torch.tensor(temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(
        self,
        order_embs: torch.Tensor,
        recipe_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        order_embs  : (B, D)  L2-normalised order embeddings
        recipe_embs : (B, D)  L2-normalised recipe embeddings

        Returns
        -------
        Scalar loss tensor.

        Notes
        -----
        Assumes the i-th order and the i-th recipe are a positive pair.
        All B² − B off-diagonal pairs are treated as negatives.
        """
        B = order_embs.size(0)
        labels = torch.arange(B, device=order_embs.device)

        # Cosine similarities (already L2-normed) scaled by temperature
        logits = (order_embs @ recipe_embs.T) / self.temperature  # (B, B)

        # Symmetric loss: each order finds its recipe, each recipe finds its order
        loss_o2r = F.cross_entropy(logits, labels)
        loss_r2o = F.cross_entropy(logits.T, labels)

        return (loss_o2r + loss_r2o) / 2.0


class BaristaContrastiveModel(nn.Module):
    """End-to-end model combining OrderEncoder and RecipeEncoder.

    The model produces L2-normalised embeddings in a shared latent space.
    At inference time, cosine similarity (or dot product, since embeddings
    are normalised) is used to retrieve the most likely recipe(s) for an
    unseen order.

    Parameters
    ----------
    order_encoder:
        Pre-constructed OrderEncoder instance.
    recipe_encoder:
        Pre-constructed RecipeEncoder instance.
    temperature:
        Initial InfoNCE temperature (can be learnable).
    """

    def __init__(
        self,
        order_encoder: OrderEncoder,
        recipe_encoder: RecipeEncoder,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.order_encoder = order_encoder
        self.recipe_encoder = recipe_encoder
        self.loss_fn = InfoNCELoss(temperature=temperature, learnable_temperature=True)

    def encode_orders(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return L2-normalised order embeddings (B, D)."""
        emb = self.order_encoder(input_ids, attention_mask)
        return F.normalize(emb, dim=-1)

    def encode_recipes(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised recipe embeddings (B, D)."""
        emb = self.recipe_encoder(token_ids)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        order_input_ids: torch.Tensor,
        order_attention_mask: torch.Tensor,
        recipe_token_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss for a batch of (order, recipe) positive pairs.

        Parameters
        ----------
        order_input_ids : (B, L_text)
        order_attention_mask : (B, L_text)
        recipe_token_ids : (B, L_recipe)

        Returns
        -------
        dict with keys:
          ``loss``         – scalar InfoNCE loss
          ``order_embs``   – (B, D) normalised order embeddings
          ``recipe_embs``  – (B, D) normalised recipe embeddings
          ``temperature``  – current temperature value (scalar)
        """
        order_embs = self.encode_orders(order_input_ids, order_attention_mask)
        recipe_embs = self.encode_recipes(recipe_token_ids)
        loss = self.loss_fn(order_embs, recipe_embs)

        return {
            "loss": loss,
            "order_embs": order_embs,
            "recipe_embs": recipe_embs,
            "temperature": self.loss_fn.temperature,
        }

    @torch.no_grad()
    def similarity_matrix(
        self,
        order_input_ids: torch.Tensor,
        order_attention_mask: torch.Tensor,
        recipe_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return a (B_orders, B_recipes) cosine-similarity matrix for retrieval."""
        order_embs = self.encode_orders(order_input_ids, order_attention_mask)
        recipe_embs = self.encode_recipes(recipe_token_ids)
        return order_embs @ recipe_embs.T
