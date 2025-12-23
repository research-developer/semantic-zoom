"""Verb tense and aspect classification (NSM-40 / GitHub #6).

Classifies verbs into:
- Tense: PAST, PRESENT, FUTURE, INFINITIVE
- Aspect: SIMPLE, PROGRESSIVE, PERFECT, PERFECT_PROGRESSIVE

Handles compound verb forms by analyzing auxiliary chains:
- "has walked" -> PRESENT PERFECT
- "will be walking" -> FUTURE PROGRESSIVE
- "had been walking" -> PAST PERFECT_PROGRESSIVE
"""

from semantic_zoom.models import Aspect, Tense, Token, VerbCompound


# Penn Treebank tags to tense mapping for simple verbs
TAG_TO_TENSE: dict[str, Tense] = {
    "VBD": Tense.PAST,      # Past tense: walked
    "VBZ": Tense.PRESENT,   # 3rd person singular present: walks
    "VBP": Tense.PRESENT,   # Non-3rd person present: walk
    "VB": Tense.INFINITIVE,  # Base form: walk, to walk
    "VBG": Tense.PRESENT,   # Gerund/present participle: walking (default)
    "VBN": Tense.PAST,      # Past participle: walked (default without aux)
}


def classify_tense(token: Token) -> Tense | None:
    """Classify tense from a single verb token.

    This handles simple (non-compound) verb forms. For compound forms,
    use analyze_verb_compound() which considers auxiliaries.

    Args:
        token: Token to classify

    Returns:
        Tense classification or None if not a verb
    """
    # Only classify verbs
    if token.pos not in ("VERB", "AUX"):
        return None

    # Modals don't have intrinsic tense
    if token.tag == "MD":
        return None

    return TAG_TO_TENSE.get(token.tag)


def classify_aspect(token: Token) -> Aspect | None:
    """Classify aspect from a single verb token.

    Simple verb forms have SIMPLE aspect. For progressive/perfect,
    use analyze_verb_compound() with the full token list.

    Args:
        token: Token to classify

    Returns:
        Aspect classification or None if not a verb
    """
    if token.pos not in ("VERB", "AUX"):
        return None

    # VBG alone suggests progressive (but needs context)
    if token.tag == "VBG":
        return Aspect.PROGRESSIVE

    # Simple aspect for finite forms
    return Aspect.SIMPLE


def _find_auxiliary_chain(tokens: list[Token], main_verb_idx: int) -> list[int]:
    """Find indices of auxiliaries forming a compound with the main verb.

    Walks backward through the token list to find auxiliaries that
    connect to the main verb through the dependency chain.

    Args:
        tokens: Full token list
        main_verb_idx: Index of the main verb

    Returns:
        List of auxiliary indices in order (first aux to last)
    """
    aux_indices: list[int] = []

    # Look for tokens before the main verb that are auxiliaries
    for i in range(main_verb_idx):
        token = tokens[i]
        if token.pos == "AUX" or token.tag == "MD":
            # Check if this aux connects to the main verb chain
            # Simple heuristic: aux before main verb is part of compound
            aux_indices.append(i)

    return aux_indices


def _analyze_aux_chain(tokens: list[Token], aux_indices: list[int]) -> tuple[Tense, Aspect]:
    """Determine tense and aspect from auxiliary chain.

    Rules:
    - "will/shall" -> FUTURE
    - "have/has" + VBN -> PERFECT
    - "had" + VBN -> PAST PERFECT
    - "be" + VBG -> PROGRESSIVE
    - Combinations: "have been VBG" -> PERFECT_PROGRESSIVE

    Args:
        tokens: Full token list
        aux_indices: Indices of auxiliaries in order

    Returns:
        Tuple of (tense, aspect)
    """
    tense = Tense.PRESENT  # Default
    aspect = Aspect.SIMPLE  # Default

    has_will = False
    has_have = False
    has_be_progressive = False
    have_tense: Tense | None = None

    for idx in aux_indices:
        aux = tokens[idx]
        lemma = aux.lemma.lower()

        # Modal "will/shall" -> future
        if aux.tag == "MD" and lemma in ("will", "shall", "'ll"):
            has_will = True

        # "have/has/had" -> perfect aspect
        elif lemma == "have":
            has_have = True
            # Determine tense from have auxiliary
            if aux.tag == "VBD":  # "had"
                have_tense = Tense.PAST
            elif aux.tag in ("VBZ", "VBP"):  # "has/have"
                have_tense = Tense.PRESENT
            elif aux.tag == "VB":  # "have" after modal
                have_tense = None  # Tense from modal

        # "be" forms before VBG -> progressive
        elif lemma == "be":
            has_be_progressive = True
            # If no have, get tense from be
            if not has_have and not has_will:
                if aux.tag == "VBD":  # "was/were"
                    tense = Tense.PAST
                elif aux.tag in ("VBZ", "VBP"):  # "is/am/are"
                    tense = Tense.PRESENT

    # Determine final tense
    if has_will:
        tense = Tense.FUTURE
    elif has_have and have_tense:
        tense = have_tense

    # Determine final aspect
    if has_have and has_be_progressive:
        aspect = Aspect.PERFECT_PROGRESSIVE
    elif has_have:
        aspect = Aspect.PERFECT
    elif has_be_progressive:
        aspect = Aspect.PROGRESSIVE
    else:
        aspect = Aspect.SIMPLE

    return tense, aspect


def analyze_verb_compound(tokens: list[Token], main_verb_idx: int) -> VerbCompound:
    """Analyze a compound verb form to determine tense and aspect.

    Args:
        tokens: Full token list for the sentence
        main_verb_idx: Index of the main (lexical) verb

    Returns:
        VerbCompound with tense, aspect, and auxiliary indices
    """
    main_verb = tokens[main_verb_idx]

    # Find auxiliary chain
    aux_indices = _find_auxiliary_chain(tokens, main_verb_idx)

    if not aux_indices:
        # No auxiliaries - simple form
        tense = classify_tense(main_verb)
        aspect = Aspect.SIMPLE if main_verb.tag != "VBG" else Aspect.PROGRESSIVE

        return VerbCompound(
            main_verb_idx=main_verb_idx,
            auxiliary_indices=[],
            tense=tense,
            aspect=aspect,
        )

    # Analyze the auxiliary chain
    tense, aspect = _analyze_aux_chain(tokens, aux_indices)

    return VerbCompound(
        main_verb_idx=main_verb_idx,
        auxiliary_indices=aux_indices,
        tense=tense,
        aspect=aspect,
    )


def classify_token_tense(token: Token, tokens: list[Token] | None = None) -> Token:
    """Classify tense/aspect and update the token.

    Args:
        token: Token to classify
        tokens: Optional full token list for compound analysis

    Returns:
        The same token with tense and aspect fields populated
    """
    if tokens and token.pos == "VERB":
        # Try compound analysis
        compound = analyze_verb_compound(tokens, token.idx)
        token.tense = compound.tense
        token.aspect = compound.aspect
    else:
        # Simple classification
        token.tense = classify_tense(token)
        token.aspect = classify_aspect(token)

    return token
