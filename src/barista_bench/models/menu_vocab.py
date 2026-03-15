"""
Defines the complete menu vocabulary used to tokenise recipes for the
attention-based RecipeEncoder.

A "recipe token" is one atomic menu concept: an item name, a size, or a
modifier.  Special tokens ([PAD] and [CLS]) are prepended so that the
encoder can follow the standard transformer convention:

    [CLS]  item_token  size_token  mod_token_1  mod_token_2  ...

The [CLS] position's output is used as the recipe embedding.
"""

from __future__ import annotations

# ── Menu items ──────────────────────────────────────────────────────────────
DRINK_ITEMS: list[str] = [
    "Espresso",
    "Americano",
    "Drip Coffee",
    "Latte",
    "Cappuccino",
    "Flat White",
    "Mocha",
    "Caramel Macchiato",
    "Cold Brew",
    "Iced Coffee",
    "Frappe (Coffee)",
    "Frappe (Mocha)",
    "Strawberry Smoothie",
    "Chai Latte",
    "Matcha Latte",
    "Earl Grey Tea",
    "Green Tea",
    "Hot Chocolate",
]

FOOD_ITEMS: list[str] = [
    "Butter Croissant",
    "Blueberry Muffin",
    "Bagel",
    "Avocado Toast",
    "Bacon Gouda Sandwich",
]

ALL_ITEMS: list[str] = DRINK_ITEMS + FOOD_ITEMS

# ── Sizes ────────────────────────────────────────────────────────────────────
SIZES: list[str] = ["Short", "Tall", "Grande", "Venti", "Trenta"]

# ── Modifiers ────────────────────────────────────────────────────────────────
MODIFIERS: list[str] = [
    # Milks
    "Oat Milk",
    "Almond Milk",
    "Soy Milk",
    "Coconut Milk",
    "Breve",
    "Skim Milk",
    # Syrups
    "Vanilla Syrup",
    "Caramel Syrup",
    "Hazelnut Syrup",
    "Peppermint Syrup",
    "Sugar Free Vanilla",
    "Classic Syrup",
    # Toppings / texture
    "Extra Shot",
    "Whip Cream",
    "No Whip",
    "Cold Foam",
    "Caramel Drizzle",
    "Extra Hot",
    "Light Ice",
    "No Ice",
]

# ── Token-type IDs (used for type embeddings in RecipeEncoder) ───────────────
TOKEN_TYPE_ITEM = 0
TOKEN_TYPE_SIZE = 1
TOKEN_TYPE_MODIFIER = 2
TOKEN_TYPE_SPECIAL = 3   # [CLS] / [PAD]

# ── Vocabulary ───────────────────────────────────────────────────────────────
_SPECIAL_TOKENS = ["[PAD]", "[CLS]"]

VOCAB: list[str] = _SPECIAL_TOKENS + ALL_ITEMS + SIZES + MODIFIERS

PAD_ID: int = VOCAB.index("[PAD]")
CLS_ID: int = VOCAB.index("[CLS]")

_TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}


def token_to_id(token: str) -> int:
    """Return the integer ID for a vocabulary token."""
    if token not in _TOKEN_TO_ID:
        raise KeyError(f"Token '{token}' not in menu vocabulary.")
    return _TOKEN_TO_ID[token]


def token_type(token_id: int) -> int:
    """Return the semantic type (item / size / modifier / special) for a token."""
    tok = VOCAB[token_id]
    if tok in _SPECIAL_TOKENS:
        return TOKEN_TYPE_SPECIAL
    if tok in ALL_ITEMS:
        return TOKEN_TYPE_ITEM
    if tok in SIZES:
        return TOKEN_TYPE_SIZE
    return TOKEN_TYPE_MODIFIER


VOCAB_SIZE: int = len(VOCAB)
# Maximum recipe sequence length: [CLS] + 1 item + 1 size + all modifiers
MAX_RECIPE_LEN: int = 2 + len(MODIFIERS)
