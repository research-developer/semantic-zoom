"""Adverb tier assignment (NSM-42 / GitHub #8).

Classifies adverbs into semantic tiers:
- MANNER: how (quickly, carefully)
- PLACE: where (here, there, everywhere)
- FREQUENCY: how often (always, never, sometimes)
- TIME: when (now, yesterday, soon)
- PURPOSE: why (therefore, consequently)
- SENTENCE: sentence-level modifiers (frankly, unfortunately)
- DEGREE: intensifiers attached to other modifiers (very, extremely)

Canonical order: Manner > Place > Frequency > Time > Purpose
"""

from semantic_zoom.models import AdverbTier, Token


# =============================================================================
# Adverb Category Lexicons
# =============================================================================

MANNER_ADVERBS = frozenset({
    "quickly", "slowly", "carefully", "carelessly", "loudly", "quietly",
    "softly", "hard", "fast", "well", "badly", "easily", "difficultly",
    "beautifully", "gracefully", "awkwardly", "clumsily", "gently",
    "roughly", "harshly", "smoothly", "suddenly", "gradually",
    "deliberately", "accidentally", "automatically", "manually",
})

PLACE_ADVERBS = frozenset({
    "here", "there", "everywhere", "somewhere", "nowhere", "anywhere",
    "outside", "inside", "upstairs", "downstairs", "abroad", "overseas",
    "nearby", "far", "away", "home", "underground", "overhead",
    "backwards", "forwards", "sideways", "upward", "downward",
})

FREQUENCY_ADVERBS = frozenset({
    "always", "never", "often", "sometimes", "rarely", "seldom",
    "usually", "frequently", "occasionally", "constantly", "continuously",
    "regularly", "irregularly", "daily", "weekly", "monthly", "yearly",
    "annually", "normally", "generally", "typically", "ever",
})

TIME_ADVERBS = frozenset({
    "now", "then", "yesterday", "today", "tomorrow", "soon", "later",
    "already", "still", "yet", "lately", "recently", "immediately",
    "eventually", "finally", "before", "after", "afterwards", "meanwhile",
    "early", "late", "previously", "formerly", "currently", "presently",
})

PURPOSE_ADVERBS = frozenset({
    "therefore", "consequently", "hence", "thus", "accordingly",
    "so", "why",  # Note: "so" can be degree too, context matters
})

SENTENCE_ADVERBS = frozenset({
    "frankly", "honestly", "unfortunately", "fortunately", "surprisingly",
    "obviously", "clearly", "apparently", "evidently", "certainly",
    "probably", "possibly", "hopefully", "supposedly", "allegedly",
    "admittedly", "undoubtedly", "interestingly", "curiously",
    "naturally", "personally", "seriously", "literally",
})

DEGREE_ADVERBS = frozenset({
    "very", "extremely", "quite", "rather", "too", "fairly", "highly",
    "incredibly", "remarkably", "absolutely", "completely", "totally",
    "entirely", "utterly", "perfectly", "somewhat", "slightly",
    "barely", "hardly", "scarcely", "almost", "nearly", "really",
    "truly", "awfully", "terribly", "enormously", "immensely",
})


def classify_adverb_tier(
    token: Token,
    is_sentence_initial: bool = False,
) -> AdverbTier | None:
    """Classify an adverb into its semantic tier.

    Args:
        token: Adverb token to classify
        is_sentence_initial: Whether this adverb is at sentence start

    Returns:
        AdverbTier classification or None if not an adverb
    """
    if token.pos != "ADV":
        return None

    lemma = token.lemma.lower()
    text = token.text.lower()

    # Check degree first (they modify other modifiers)
    if lemma in DEGREE_ADVERBS or text in DEGREE_ADVERBS:
        return AdverbTier.DEGREE

    # Sentence adverbs are evaluative and often sentence-initial
    if lemma in SENTENCE_ADVERBS or text in SENTENCE_ADVERBS:
        # Only mark as SENTENCE if it's sentence-initial or modifies ROOT
        if is_sentence_initial:
            return AdverbTier.SENTENCE
        # Otherwise fall through to check other categories
        # (some words like "clearly" can be manner too)

    # Check specific categories
    if lemma in MANNER_ADVERBS or text in MANNER_ADVERBS:
        return AdverbTier.MANNER

    if lemma in PLACE_ADVERBS or text in PLACE_ADVERBS:
        return AdverbTier.PLACE

    if lemma in FREQUENCY_ADVERBS or text in FREQUENCY_ADVERBS:
        return AdverbTier.FREQUENCY

    if lemma in TIME_ADVERBS or text in TIME_ADVERBS:
        return AdverbTier.TIME

    if lemma in PURPOSE_ADVERBS or text in PURPOSE_ADVERBS:
        return AdverbTier.PURPOSE

    # Heuristic: -ly adverbs not in other categories are usually manner
    if text.endswith("ly") and len(text) > 3:
        return AdverbTier.MANNER

    # Default to MANNER for unknown adverbs
    return AdverbTier.MANNER


def get_degree_attachment(token: Token, tokens: list[Token]) -> int | None:
    """Find what word a degree adverb modifies.

    Degree adverbs like "very" typically attach to the next adjective
    or adverb in the sequence.

    Args:
        token: Degree adverb token
        tokens: Full token list

    Returns:
        Index of the modified word, or None if not found
    """
    if token.adv_tier != AdverbTier.DEGREE:
        tier = classify_adverb_tier(token)
        if tier != AdverbTier.DEGREE:
            return None

    # The head of a degree adverb is typically what it modifies
    head_idx = token.head_idx
    if 0 <= head_idx < len(tokens):
        head = tokens[head_idx]
        # Degree adverbs modify adjectives or other adverbs
        if head.pos in ("ADJ", "ADV"):
            return head_idx

    # Fallback: look for adjacent adjective or adverb
    next_idx = token.idx + 1
    if next_idx < len(tokens):
        next_token = tokens[next_idx]
        if next_token.pos in ("ADJ", "ADV"):
            return next_idx

    return None


def classify_tokens_adverbs(tokens: list[Token]) -> list[Token]:
    """Classify all adverbs in a token list.

    Updates adv_tier and adv_attachment fields on adverb tokens.

    Args:
        tokens: Full token list

    Returns:
        The same list with adverb classifications added
    """
    for i, token in enumerate(tokens):
        if token.pos != "ADV":
            continue

        # Check if sentence-initial
        is_sentence_initial = i == 0 or (
            i > 0 and tokens[i - 1].pos == "PUNCT"
        )

        # Classify tier
        token.adv_tier = classify_adverb_tier(token, is_sentence_initial)

        # For degree adverbs, find attachment
        if token.adv_tier == AdverbTier.DEGREE:
            token.adv_attachment = get_degree_attachment(token, tokens)

    return tokens
