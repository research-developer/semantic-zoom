"""Discourse adverb → inter-frame relation mapping (NSM-45).

Marks discourse adverbs as INTER_FRAME_MORPHISM with relation types:
CONTRAST, CONSEQUENCE, ADDITION, CONCESSION, SEQUENCE, EXEMPLIFICATION.

Frames act as nodes, discourse adverbs as edges connecting them.
"""
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class DiscourseRelation(Enum):
    """Types of inter-frame discourse relations.

    These represent the semantic relationship between adjacent frames
    as signaled by discourse adverbs/connectives.
    """
    CONTRAST = auto()  # however, nevertheless, instead
    CONSEQUENCE = auto()  # therefore, thus, hence
    ADDITION = auto()  # moreover, furthermore, also
    CONCESSION = auto()  # although, though, still
    SEQUENCE = auto()  # then, next, finally
    EXEMPLIFICATION = auto()  # for example, namely


@dataclass
class DiscourseAdverb:
    """A discourse adverb identified in text.

    Attributes:
        text: The adverb/phrase as found in text
        start_char: Character offset where adverb starts
        end_char: Character offset where adverb ends
        relation: The discourse relation type
    """
    text: str
    start_char: int
    end_char: int
    relation: DiscourseRelation


@dataclass
class InterFrameMorphism:
    """An inter-frame morphism generated from a discourse adverb.

    Represents an edge between frames in the semantic graph.

    Attributes:
        source_frame: Reference to the source frame (preceding context)
        target_frame: Reference to the target frame (following context)
        edge_label: The discourse adverb/phrase (lowercase)
        relation: The discourse relation type
        strength: Confidence/strength of the relation (0.0 to 1.0)
    """
    source_frame: str
    target_frame: str
    edge_label: str
    relation: DiscourseRelation
    strength: float = 1.0


# Discourse adverb mappings by relation type
_CONTRAST_MARKERS: dict[str, float] = {
    "however": 1.0,
    "nevertheless": 0.95,
    "nonetheless": 0.95,
    "conversely": 0.9,
    "instead": 0.85,
    "rather": 0.7,
    "on the other hand": 1.0,
    "in contrast": 0.95,
    "on the contrary": 0.95,
    "but": 0.6,  # Lower confidence - often not discourse-level
}

_CONSEQUENCE_MARKERS: dict[str, float] = {
    "therefore": 1.0,
    "thus": 0.95,
    "hence": 0.95,
    "consequently": 1.0,
    "accordingly": 0.9,
    "as a result": 1.0,
    "so": 0.5,  # Lower - often not discourse-level
    "for this reason": 0.95,
}

_ADDITION_MARKERS: dict[str, float] = {
    "moreover": 1.0,
    "furthermore": 1.0,
    "additionally": 0.95,
    "also": 0.7,  # Often phrase-level, not discourse
    "besides": 0.85,
    "in addition": 1.0,
    "what is more": 0.9,
}

_CONCESSION_MARKERS: dict[str, float] = {
    "although": 0.95,
    "though": 0.8,
    "albeit": 0.95,
    "still": 0.7,
    "yet": 0.75,
    "even so": 0.9,
    "granted": 0.85,
}

_SEQUENCE_MARKERS: dict[str, float] = {
    "then": 0.8,
    "subsequently": 0.95,
    "afterwards": 0.9,
    "next": 0.85,
    "finally": 0.9,
    "meanwhile": 0.85,
    "first": 0.7,
    "second": 0.7,
    "third": 0.7,
    "lastly": 0.9,
    "afterward": 0.9,
}

_EXEMPLIFICATION_MARKERS: dict[str, float] = {
    "for example": 1.0,
    "for instance": 1.0,
    "namely": 0.95,
    "specifically": 0.85,
    "in particular": 0.9,
    "such as": 0.7,
    "e.g.": 0.95,
    "i.e.": 0.9,
}

# Combined mapping: marker → (relation, strength)
_ALL_MARKERS: dict[str, tuple[DiscourseRelation, float]] = {}
for markers, relation in [
    (_CONTRAST_MARKERS, DiscourseRelation.CONTRAST),
    (_CONSEQUENCE_MARKERS, DiscourseRelation.CONSEQUENCE),
    (_ADDITION_MARKERS, DiscourseRelation.ADDITION),
    (_CONCESSION_MARKERS, DiscourseRelation.CONCESSION),
    (_SEQUENCE_MARKERS, DiscourseRelation.SEQUENCE),
    (_EXEMPLIFICATION_MARKERS, DiscourseRelation.EXEMPLIFICATION),
]:
    for marker, strength in markers.items():
        _ALL_MARKERS[marker] = (relation, strength)


def identify_discourse_adverbs(text: str) -> list[DiscourseAdverb]:
    """Identify discourse adverbs/connectives in text.

    Args:
        text: Input text to analyze

    Returns:
        List of DiscourseAdverb objects in text order

    Examples:
        >>> result = identify_discourse_adverbs("However, we disagree.")
        >>> len(result)
        1
        >>> result[0].relation
        DiscourseRelation.CONTRAST
    """
    if not text:
        return []

    results: list[DiscourseAdverb] = []
    text_lower = text.lower()

    # Sort markers by length (longest first) for proper multi-word matching
    sorted_markers = sorted(_ALL_MARKERS.keys(), key=len, reverse=True)

    for marker in sorted_markers:
        # Use word boundary matching
        pattern = r'\b' + re.escape(marker) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start, end = match.start(), match.end()

            # Check for overlap with existing matches
            overlaps = any(
                existing.start_char <= start < existing.end_char or
                existing.start_char < end <= existing.end_char
                for existing in results
            )
            if overlaps:
                continue

            # Get original case from text
            original_text = text[start:end]
            relation, _ = _ALL_MARKERS[marker]

            results.append(DiscourseAdverb(
                text=original_text,
                start_char=start,
                end_char=end,
                relation=relation,
            ))

    # Sort by position
    results.sort(key=lambda x: x.start_char)
    return results


def map_to_inter_frame_relation(adverb: DiscourseAdverb) -> InterFrameMorphism:
    """Map a discourse adverb to an inter-frame morphism.

    Creates a morphism edge with:
    - source_frame: "PREVIOUS" (placeholder for preceding frame)
    - target_frame: "FOLLOWING" (placeholder for following frame)
    - edge_label: the discourse marker (lowercase)
    - relation: the discourse relation type
    - strength: confidence in the relation

    Args:
        adverb: A DiscourseAdverb object from identify_discourse_adverbs

    Returns:
        InterFrameMorphism representing the inter-frame relation

    Note:
        Frame references are placeholders. Phase 4 (Frame Integration)
        will resolve these to actual frame IDs.
    """
    marker_lower = adverb.text.lower()
    _, strength = _ALL_MARKERS.get(marker_lower, (adverb.relation, 0.8))

    return InterFrameMorphism(
        source_frame="PREVIOUS",
        target_frame="FOLLOWING",
        edge_label=marker_lower,
        relation=adverb.relation,
        strength=strength,
    )
