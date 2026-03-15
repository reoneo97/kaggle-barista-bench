"""
Inference pipeline for the trained BaristaContrastiveModel
===========================================================

At inference time we:
1. Pre-compute embeddings for **every possible canonical recipe** in the
   menu (item × size × modifier powerset would be huge — instead we use
   the ground-truth recipes seen in training plus a set of canonical
   base recipes: each item × each valid size × no modifiers).
2. For each new order text, encode it with the OrderEncoder.
3. Retrieve the top-K nearest recipes by cosine similarity.
4. Aggregate retrieved items into an ``OrderList`` and compute the price.

This is fundamentally different from the LLM approach:
  LLM approach  → generate free-form JSON via next-token prediction
  This approach → retrieve structured recipes via embedding similarity

The retrieval approach is:
  + Faster at inference (no LLM API call)
  + Fully constrained (can only output valid menu items)
  + Interpretable (we know exactly which embeddings are close)
  - Requires training data and a trained model
  - Cannot handle truly novel items outside the menu vocabulary
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..models.contrastive import BaristaContrastiveModel
from ..models.menu_vocab import ALL_ITEMS, DRINK_ITEMS, FOOD_ITEMS, MODIFIERS, SIZES
from ..models.order_encoder import OrderTokenizer
from ..models.recipe_encoder import recipe_to_token_ids
from ..schema import Order, OrderList

# ── Prices (must mirror menu.md) ─────────────────────────────────────────────
_BASE_PRICES: dict[str, float] = {
    "Espresso": 3.00, "Americano": 3.50, "Drip Coffee": 2.50,
    "Latte": 4.50, "Cappuccino": 4.50, "Flat White": 4.75,
    "Mocha": 5.00, "Caramel Macchiato": 5.25, "Cold Brew": 4.25,
    "Iced Coffee": 3.00, "Frappe (Coffee)": 5.50, "Frappe (Mocha)": 5.75,
    "Strawberry Smoothie": 6.00, "Chai Latte": 4.75, "Matcha Latte": 5.25,
    "Earl Grey Tea": 3.00, "Green Tea": 3.00, "Hot Chocolate": 4.00,
    "Butter Croissant": 3.50, "Blueberry Muffin": 3.75, "Bagel": 2.50,
    "Avocado Toast": 7.00, "Bacon Gouda Sandwich": 5.50,
}
_SIZE_ADJUSTMENTS: dict[str, float] = {
    "Short": -0.50, "Tall": 0.00, "Grande": 0.50, "Venti": 1.00, "Trenta": 1.50,
}
_MODIFIER_PRICES: dict[str, float] = {
    "Oat Milk": 0.80, "Almond Milk": 0.60, "Soy Milk": 0.60,
    "Coconut Milk": 0.70, "Breve": 0.80, "Skim Milk": 0.00,
    "Vanilla Syrup": 0.50, "Caramel Syrup": 0.50, "Hazelnut Syrup": 0.50,
    "Peppermint Syrup": 0.50, "Sugar Free Vanilla": 0.50, "Classic Syrup": 0.00,
    "Extra Shot": 1.00, "Whip Cream": 0.50, "No Whip": 0.00,
    "Cold Foam": 1.25, "Caramel Drizzle": 0.50, "Extra Hot": 0.00,
    "Light Ice": 0.00, "No Ice": 0.00,
}


@dataclass
class CanonicalRecipe:
    """A single menu entry with its token IDs pre-computed."""
    item_name: str
    size: Optional[str]
    modifiers: list[str]
    token_ids: list[int]

    def to_order(self, quantity: int = 1) -> Order:
        return Order(name=self.item_name, size=self.size, quantity=quantity,
                     modifiers=self.modifiers)

    def unit_price(self) -> float:
        price = _BASE_PRICES.get(self.item_name, 0.0)
        if self.size:
            price += _SIZE_ADJUSTMENTS.get(self.size, 0.0)
        for mod in self.modifiers:
            price += _MODIFIER_PRICES.get(mod, 0.0)
        return round(price, 2)


def build_canonical_recipe_index() -> list[CanonicalRecipe]:
    """Generate the base recipe catalogue: each item × each valid size, no modifiers.

    This is the retrieval index.  If a training set is available the index
    can be extended with real observed recipes to cover common modifier
    combinations.
    """
    recipes: list[CanonicalRecipe] = []

    for item in DRINK_ITEMS:
        for size in SIZES:
            # Skip Trenta for hot drinks (cold-only)
            if size == "Trenta" and item not in (
                "Cold Brew", "Iced Coffee", "Frappe (Coffee)",
                "Frappe (Mocha)", "Strawberry Smoothie",
            ):
                continue
            token_ids = recipe_to_token_ids(item, size, modifiers=[])
            recipes.append(CanonicalRecipe(item, size, [], token_ids))

    for item in FOOD_ITEMS:
        token_ids = recipe_to_token_ids(item, size=None, modifiers=[])
        recipes.append(CanonicalRecipe(item, None, [], token_ids))

    return recipes


class BaristaRetriever:
    """Retrieves the most likely ordered items for a natural-language order.

    Parameters
    ----------
    model:
        Trained ``BaristaContrastiveModel`` (in eval mode).
    tokenizer:
        ``OrderTokenizer`` matching the one used during training.
    recipe_index:
        List of ``CanonicalRecipe`` objects to retrieve from.
    top_k:
        Maximum number of items to retrieve per order.
    threshold:
        Minimum cosine similarity to include a recipe (avoids including
        irrelevant items for short orders).
    device:
        Inference device.
    """

    def __init__(
        self,
        model: BaristaContrastiveModel,
        tokenizer: OrderTokenizer,
        recipe_index: Optional[list[CanonicalRecipe]] = None,
        top_k: int = 5,
        threshold: float = 0.3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.index = recipe_index or build_canonical_recipe_index()
        self._recipe_embs = self._precompute_index_embeddings()

    @torch.no_grad()
    def _precompute_index_embeddings(self) -> torch.Tensor:
        """Encode all canonical recipes once and cache the result."""
        token_id_matrix = torch.tensor(
            [r.token_ids for r in self.index], dtype=torch.long
        ).to(self.device)
        return self.model.encode_recipes(token_id_matrix)  # (N, D)

    @torch.no_grad()
    def retrieve(self, order_text: str) -> OrderList:
        """Parse a single free-text order into a structured ``OrderList``.

        Parameters
        ----------
        order_text:
            Raw customer order string.

        Returns
        -------
        ``OrderList`` with retrieved items and computed total price.
        """
        encoded = self.tokenizer([order_text])
        order_emb = self.model.encode_orders(
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )  # (1, D)

        sims = (order_emb @ self._recipe_embs.T).squeeze(0)  # (N,)
        topk_vals, topk_idxs = sims.topk(min(self.top_k, len(self.index)))

        items: list[Order] = []
        for sim, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
            if sim < self.threshold:
                break
            recipe = self.index[idx]
            items.append(recipe.to_order(quantity=1))

        if not items:
            # Fall back to the single highest-similarity recipe
            best_idx = sims.argmax().item()
            items = [self.index[best_idx].to_order(quantity=1)]

        total_price = round(
            sum(self.index[i].unit_price() for i in topk_idxs.tolist()[: len(items)]), 2
        )
        return OrderList(items=items, total_price=total_price)
