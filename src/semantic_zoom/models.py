"""Shared data models for Semantic Zoom pipeline.

This module defines the Token interface and classification enums used
across phases. Phase 1 produces tokens with POS tags; Phase 2 adds
grammatical classifications.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# Phase 2: Grammatical Classification Enums (NSM-39 through NSM-42)
# =============================================================================


class Person(Enum):
    """Noun/pronoun person classification (NSM-39)."""

    FIRST = auto()  # I, we, me, us
    SECOND = auto()  # you
    THIRD = auto()  # he, she, it, they, proper nouns
    NONE = auto()  # Non-personal nouns without person marking


class Tense(Enum):
    """Verb tense classification (NSM-40)."""

    PAST = auto()
    PRESENT = auto()
    FUTURE = auto()
    INFINITIVE = auto()  # Non-finite forms


class Aspect(Enum):
    """Verb aspect classification (NSM-40)."""

    SIMPLE = auto()  # walks, walked
    PROGRESSIVE = auto()  # is walking
    PERFECT = auto()  # has walked
    PERFECT_PROGRESSIVE = auto()  # has been walking


class AdjectiveSlot(Enum):
    """Adjective ordering slots following canonical English order (NSM-41).

    Order: opinion > size > age > shape > color > origin > material > purpose
    """

    OPINION = 1  # lovely, ugly, amazing
    SIZE = 2  # big, small, tiny
    AGE = 3  # old, young, ancient
    SHAPE = 4  # round, square, flat
    COLOR = 5  # red, blue, green
    ORIGIN = 6  # American, French, lunar
    MATERIAL = 7  # wooden, metal, silk
    PURPOSE = 8  # sleeping (bag), wedding (dress)


class AdverbTier(Enum):
    """Adverb tier classification (NSM-42).

    Canonical order: Manner > Place > Frequency > Time > Purpose
    """

    MANNER = auto()  # quickly, carefully
    PLACE = auto()  # here, there, everywhere
    FREQUENCY = auto()  # always, often, never
    TIME = auto()  # now, yesterday, soon
    PURPOSE = auto()  # therefore, consequently
    SENTENCE = auto()  # Sentence-level adverbs (frankly, unfortunately)
    DEGREE = auto()  # Modifying adverbs (very, extremely)


# =============================================================================
# Token Dataclass - Shared between Phase 1 and Phase 2
# =============================================================================


@dataclass
class Token:
    """Token with POS and grammatical classification.

    Phase 1 populates: text, lemma, pos, tag, dep, head_idx, idx
    Phase 2 adds: person, generic, tense, aspect, adj_slot, adj_original_pos,
                  adv_tier, adv_attachment
    """

    # Core token info (Phase 1: NSM-35)
    text: str
    lemma: str
    idx: int  # Token index in sentence

    # POS tagging (Phase 1: NSM-36)
    pos: str  # Universal POS tag (NOUN, VERB, ADJ, etc.)
    tag: str  # Fine-grained tag (VBD, NN, JJ, etc.)

    # Dependency parsing (Phase 1: NSM-37)
    dep: str  # Dependency relation (nsubj, dobj, etc.)
    head_idx: int  # Index of head token

    # Noun person classification (Phase 2: NSM-39)
    person: Optional[Person] = None
    generic: bool = False  # Generic construction ("One must be careful")

    # Verb tense/aspect (Phase 2: NSM-40)
    tense: Optional[Tense] = None
    aspect: Optional[Aspect] = None

    # Adjective ordering (Phase 2: NSM-41)
    adj_slot: Optional[AdjectiveSlot] = None
    adj_original_pos: Optional[int] = None  # Original position if reordered

    # Adverb tier (Phase 2: NSM-42)
    adv_tier: Optional[AdverbTier] = None
    adv_attachment: Optional[int] = None  # Index of word being modified (for DEGREE)


@dataclass
class AdjectiveChain:
    """A chain of adjectives modifying the same noun (NSM-41).

    Tracks original vs. canonical ordering.
    """

    noun_idx: int  # Index of the noun being modified
    adjectives: list[Token] = field(default_factory=list)  # In original order
    canonical_order: list[int] = field(default_factory=list)  # Indices in canonical order
    is_canonical: bool = True  # True if original order matches canonical


@dataclass
class VerbCompound:
    """Compound verb form tracking auxiliaries (NSM-40).

    Examples: "has been walking", "will have finished"
    """

    main_verb_idx: int
    auxiliary_indices: list[int] = field(default_factory=list)
    tense: Optional[Tense] = None
    aspect: Optional[Aspect] = None
