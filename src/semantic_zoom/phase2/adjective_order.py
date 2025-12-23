"""Adjective ordering extraction and normalization (NSM-41 / GitHub #7).

Classifies adjectives into the canonical English ordering slots:
    opinion > size > age > shape > color > origin > material > purpose

Detects non-canonical orderings and normalizes them while preserving
the original positions for downstream processing.
"""

from semantic_zoom.models import AdjectiveChain, AdjectiveSlot, Token


# =============================================================================
# Adjective Category Lexicons
# =============================================================================

OPINION_ADJECTIVES = frozenset({
    # Positive opinions
    "lovely", "beautiful", "amazing", "wonderful", "excellent", "perfect",
    "nice", "good", "great", "fantastic", "marvelous", "delightful",
    "pleasant", "charming", "gorgeous", "stunning", "magnificent",
    # Negative opinions
    "ugly", "horrible", "terrible", "awful", "dreadful", "nasty",
    "bad", "poor", "disgusting", "hideous", "ghastly",
    # Neutral evaluative
    "interesting", "unusual", "strange", "ordinary", "common",
})

SIZE_ADJECTIVES = frozenset({
    "big", "small", "large", "tiny", "huge", "enormous", "massive",
    "gigantic", "little", "tall", "short", "long", "wide", "narrow",
    "thick", "thin", "slim", "fat", "broad", "compact", "miniature",
    "vast", "immense", "colossal", "petite",
})

AGE_ADJECTIVES = frozenset({
    "old", "young", "ancient", "new", "modern", "antique", "vintage",
    "contemporary", "old-fashioned", "youthful", "elderly", "mature",
    "fresh", "recent", "prehistoric", "medieval", "Victorian",
})

SHAPE_ADJECTIVES = frozenset({
    "round", "square", "flat", "circular", "rectangular", "triangular",
    "oval", "curved", "straight", "angular", "spherical", "cylindrical",
    "conical", "wavy", "crooked", "pointed", "hollow",
})

COLOR_ADJECTIVES = frozenset({
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "beige", "tan",
    "gold", "silver", "bronze", "crimson", "scarlet", "azure",
    "turquoise", "violet", "indigo", "maroon", "navy", "olive",
    "cream", "ivory", "ebony", "amber", "coral", "teal", "magenta",
    "cyan", "khaki", "lavender", "mauve", "burgundy",
})

# Origin adjectives are typically proper adjectives (nationality/place)
ORIGIN_ADJECTIVES = frozenset({
    "american", "french", "chinese", "japanese", "italian", "german",
    "british", "english", "spanish", "mexican", "indian", "russian",
    "canadian", "australian", "african", "asian", "european", "latin",
    "mediterranean", "scandinavian", "eastern", "western", "northern",
    "southern", "lunar", "solar", "martian", "tropical", "arctic",
})

# Material adjectives often end in -en or describe what something is made of
MATERIAL_ADJECTIVES = frozenset({
    "wooden", "metal", "metallic", "silk", "silky", "cotton", "plastic",
    "golden", "silver", "bronze", "copper", "steel", "iron", "glass",
    "leather", "wool", "woolen", "velvet", "satin", "linen", "denim",
    "rubber", "ceramic", "porcelain", "marble", "stone", "brick",
    "concrete", "paper", "cardboard", "aluminum", "bamboo", "wicker",
})

# Purpose adjectives (often gerunds used attributively)
PURPOSE_ADJECTIVES = frozenset({
    "sleeping", "wedding", "running", "cooking", "swimming", "dining",
    "writing", "reading", "walking", "hiking", "fishing", "hunting",
    "racing", "training", "working", "cleaning", "washing", "shopping",
})


def classify_adjective_slot(token: Token) -> AdjectiveSlot:
    """Classify an adjective into its canonical ordering slot.

    Uses lexicon lookup with fallback to morphological heuristics.

    Args:
        token: Adjective token to classify

    Returns:
        AdjectiveSlot classification
    """
    lemma = token.lemma.lower()
    text = token.text.lower()

    # Check each category lexicon
    if lemma in OPINION_ADJECTIVES or text in OPINION_ADJECTIVES:
        return AdjectiveSlot.OPINION

    if lemma in SIZE_ADJECTIVES or text in SIZE_ADJECTIVES:
        return AdjectiveSlot.SIZE

    if lemma in AGE_ADJECTIVES or text in AGE_ADJECTIVES:
        return AdjectiveSlot.AGE

    if lemma in SHAPE_ADJECTIVES or text in SHAPE_ADJECTIVES:
        return AdjectiveSlot.SHAPE

    if lemma in COLOR_ADJECTIVES or text in COLOR_ADJECTIVES:
        return AdjectiveSlot.COLOR

    if lemma in ORIGIN_ADJECTIVES or text in ORIGIN_ADJECTIVES:
        return AdjectiveSlot.ORIGIN

    if lemma in MATERIAL_ADJECTIVES or text in MATERIAL_ADJECTIVES:
        return AdjectiveSlot.MATERIAL

    if lemma in PURPOSE_ADJECTIVES or text in PURPOSE_ADJECTIVES:
        return AdjectiveSlot.PURPOSE

    # Morphological heuristics for unknown words

    # VBG tag (gerund) used as modifier -> PURPOSE
    if token.tag == "VBG" and token.dep == "amod":
        return AdjectiveSlot.PURPOSE

    # Words ending in -en often material (wooden, golden)
    if text.endswith("en") and len(text) > 3:
        return AdjectiveSlot.MATERIAL

    # Capitalized adjectives often origin (French, American)
    if token.text[0].isupper() and token.pos == "ADJ":
        return AdjectiveSlot.ORIGIN

    # Default to OPINION (closest to determiner, first in canonical order)
    return AdjectiveSlot.OPINION


def extract_adjective_chains(tokens: list[Token]) -> list[AdjectiveChain]:
    """Extract all adjective chains from a token list.

    Groups adjectives by the noun they modify using dependency relations.

    Args:
        tokens: Full token list for sentence

    Returns:
        List of AdjectiveChain objects
    """
    # Group adjectives by their head noun
    noun_to_adjs: dict[int, list[Token]] = {}

    for token in tokens:
        # Look for adjectives modifying nouns
        if token.pos == "ADJ" or (token.tag == "VBG" and token.dep == "amod"):
            if token.dep in ("amod", "acomp"):
                head_idx = token.head_idx
                # Verify head is a noun
                if 0 <= head_idx < len(tokens):
                    head = tokens[head_idx]
                    if head.pos in ("NOUN", "PROPN"):
                        if head_idx not in noun_to_adjs:
                            noun_to_adjs[head_idx] = []
                        noun_to_adjs[head_idx].append(token)

    # Build chains
    chains = []
    for noun_idx, adjs in noun_to_adjs.items():
        # Sort adjectives by their position (left to right)
        adjs.sort(key=lambda t: t.idx)

        # Classify each adjective
        for adj in adjs:
            adj.adj_slot = classify_adjective_slot(adj)

        chain = AdjectiveChain(
            noun_idx=noun_idx,
            adjectives=adjs,
        )
        chains.append(chain)

    return chains


def is_canonical_order(chain: AdjectiveChain) -> bool:
    """Check if adjectives are in canonical English order.

    Canonical order: opinion > size > age > shape > color > origin > material > purpose

    Args:
        chain: AdjectiveChain to check

    Returns:
        True if order is canonical (or chain has 0-1 adjectives)
    """
    if len(chain.adjectives) <= 1:
        return True

    # Get slot values (lower = earlier in canonical order)
    slots = [adj.adj_slot.value if adj.adj_slot else 0 for adj in chain.adjectives]

    # Check if slots are monotonically increasing (or equal)
    for i in range(len(slots) - 1):
        if slots[i] > slots[i + 1]:
            return False

    return True


def normalize_chain(chain: AdjectiveChain) -> AdjectiveChain:
    """Normalize an adjective chain to canonical order.

    Reorders adjectives to canonical order while preserving the original
    positions in the adj_original_pos field of each token.

    Args:
        chain: AdjectiveChain to normalize

    Returns:
        The same chain with canonical_order and is_canonical updated
    """
    if len(chain.adjectives) <= 1:
        chain.is_canonical = True
        chain.canonical_order = [adj.idx for adj in chain.adjectives]
        return chain

    # Record original positions
    for adj in chain.adjectives:
        adj.adj_original_pos = adj.idx

    # Check if already canonical
    chain.is_canonical = is_canonical_order(chain)

    # Sort adjectives by their slot value for canonical order
    sorted_adjs = sorted(
        chain.adjectives,
        key=lambda adj: adj.adj_slot.value if adj.adj_slot else 0
    )

    # canonical_order lists the original indices in canonical slot order
    chain.canonical_order = [adj.idx for adj in sorted_adjs]

    return chain


def classify_tokens_adjectives(tokens: list[Token]) -> list[AdjectiveChain]:
    """Extract and classify all adjective chains in a token list.

    Main entry point for adjective ordering analysis.

    Args:
        tokens: Full token list

    Returns:
        List of normalized AdjectiveChain objects
    """
    chains = extract_adjective_chains(tokens)

    for chain in chains:
        normalize_chain(chain)

    return chains
