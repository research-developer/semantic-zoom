"""Noun/pronoun person classification (NSM-39 / GitHub #5).

Classifies nouns and pronouns into person categories:
- FIRST: I, me, we, us, my, our, etc.
- SECOND: you, your, yourself, etc.
- THIRD: he, she, it, they, proper nouns, etc.
- NONE: Common nouns without person marking

Also detects generic constructions like "one must" or "people say".
"""

from semantic_zoom.models import Person, Token


# First person pronouns and possessives
FIRST_PERSON_LEMMAS = frozenset({
    "i", "me", "myself", "we", "us", "ourselves",
    "my", "mine", "our", "ours",
})

# Second person pronouns and possessives
SECOND_PERSON_LEMMAS = frozenset({
    "you", "yourself", "yourselves",
    "your", "yours",
})

# Third person pronouns and possessives
THIRD_PERSON_LEMMAS = frozenset({
    "he", "him", "himself", "his",
    "she", "her", "herself", "hers",
    "it", "itself", "its",
    "they", "them", "themselves", "their", "theirs",
    "one",  # Generic "one" is grammatically third person
    "who", "whom", "whose", "which", "that",  # Relative pronouns
})

# Generic construction markers (when used as subject)
GENERIC_MARKERS = frozenset({
    "one",     # "One must be careful"
    "people",  # "People say..."
    "someone", # "Someone could..."
    "anyone",  # "Anyone can..."
    "everyone",  # "Everyone knows..."
    "somebody",
    "anybody",
    "everybody",
})


def classify_person(token: Token) -> Person:
    """Classify a token's grammatical person.

    Args:
        token: Token with POS and lemma information

    Returns:
        Person classification (FIRST, SECOND, THIRD, or NONE)
    """
    # Only classify pronouns, nouns, and proper nouns
    if token.pos not in ("PRON", "NOUN", "PROPN"):
        return Person.NONE

    lemma = token.lemma.lower()

    # Check pronoun categories
    if token.pos == "PRON":
        if lemma in FIRST_PERSON_LEMMAS:
            return Person.FIRST
        if lemma in SECOND_PERSON_LEMMAS:
            return Person.SECOND
        if lemma in THIRD_PERSON_LEMMAS:
            return Person.THIRD
        # Unknown pronoun defaults to third person
        return Person.THIRD

    # Proper nouns are always third person
    if token.pos == "PROPN":
        return Person.THIRD

    # Common nouns don't have inherent person marking
    return Person.NONE


def is_generic_construction(token: Token) -> bool:
    """Check if a token is part of a generic construction.

    Generic constructions use nouns/pronouns in a non-specific way:
    - "One must be careful" (generic "one")
    - "People say that..." (generic "people" as subject)

    Args:
        token: Token to check

    Returns:
        True if this is a generic construction marker
    """
    lemma = token.lemma.lower()

    # Check if it's a known generic marker
    if lemma not in GENERIC_MARKERS:
        return False

    # "One" is always generic when used as a pronoun
    if lemma == "one" and token.pos == "PRON":
        return True

    # For nouns like "people", they're only generic when used as subject
    # (e.g., "People say..." vs "the people in the room")
    if token.pos == "NOUN" and token.dep in ("nsubj", "nsubjpass"):
        return True

    return False


def classify_token_person(token: Token) -> Token:
    """Classify person and update the token in place.

    Args:
        token: Token to classify

    Returns:
        The same token with person and generic fields populated
    """
    token.person = classify_person(token)
    token.generic = is_generic_construction(token)
    return token


def classify_tokens_person(tokens: list[Token]) -> list[Token]:
    """Classify person for all tokens in a list.

    Args:
        tokens: List of tokens to classify

    Returns:
        The same list with person classifications added
    """
    for token in tokens:
        classify_token_person(token)
    return tokens
