"""Focusing adverb identification and scope marking (NSM-44).

Identifies focusing adverbs (only, even, just, etc.) and marks them as
SCOPE_OPERATORs with scope target identification and possible bindings.
"""
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ScopeOperator(Enum):
    """Types of scope operators."""
    FOCUS = auto()  # Focusing adverbs
    QUANTIFIER = auto()  # Quantificational scope
    NEGATION = auto()  # Negative scope


@dataclass
class FocusingAdverb:
    """A focusing adverb identified in text.

    Attributes:
        text: The adverb text as found
        start_char: Character offset where adverb starts
        end_char: Character offset where adverb ends
        operator_type: Always SCOPE_OPERATOR.FOCUS
        invertible: Always False (focusing operations are non-reversible)
        focus_type: 'restrictive' (only, just) or 'additive' (even, also)
        scope_target: The identified scope target (if determinable)
    """
    text: str
    start_char: int
    end_char: int
    operator_type: ScopeOperator = ScopeOperator.FOCUS
    invertible: bool = False
    focus_type: str = "restrictive"
    scope_target: Optional[str] = None


@dataclass
class ScopeBinding:
    """A possible scope interpretation for a focusing adverb.

    Attributes:
        target: The constituent that the adverb scopes over
        confidence: Confidence score for this interpretation (0.0 to 1.0)
        position: 'subject', 'object', 'verb', 'adjunct'
    """
    target: str
    confidence: float
    position: str = "unknown"


# Focusing adverbs categorized by type
_RESTRICTIVE_ADVERBS = frozenset({
    "only", "just", "merely", "simply", "solely", "exclusively",
    "purely", "alone",
})

_ADDITIVE_ADVERBS = frozenset({
    "even", "also", "too", "as well",
})

_PARTICULARIZING_ADVERBS = frozenset({
    "especially", "particularly", "specifically", "notably",
    "chiefly", "mainly", "primarily", "principally",
})

_PRECISION_ADVERBS = frozenset({
    "exactly", "precisely", "just",  # "just" can be precision or restrictive
})

# Combined set for quick lookup
_ALL_FOCUSING_ADVERBS = (
    _RESTRICTIVE_ADVERBS | _ADDITIVE_ADVERBS |
    _PARTICULARIZING_ADVERBS | _PRECISION_ADVERBS
)


def _get_focus_type(adverb: str) -> str:
    """Determine focus type for an adverb."""
    adverb_lower = adverb.lower()
    if adverb_lower in _ADDITIVE_ADVERBS:
        return "additive"
    elif adverb_lower in _PARTICULARIZING_ADVERBS:
        return "particularizing"
    elif adverb_lower in _PRECISION_ADVERBS:
        return "precision"
    else:
        return "restrictive"


def identify_focusing_adverbs(text: str) -> list[FocusingAdverb]:
    """Identify focusing adverbs in text.

    Args:
        text: Input text to analyze

    Returns:
        List of FocusingAdverb objects with positions and types

    Examples:
        >>> result = identify_focusing_adverbs("I only want coffee.")
        >>> len(result)
        1
        >>> result[0].text
        'only'
        >>> result[0].focus_type
        'restrictive'
    """
    if not text:
        return []

    results: list[FocusingAdverb] = []
    text_lower = text.lower()

    # Build regex pattern for multi-word adverbs first, then single words
    # Sort by length (longest first) to match multi-word before parts
    sorted_adverbs = sorted(_ALL_FOCUSING_ADVERBS, key=len, reverse=True)

    for adverb in sorted_adverbs:
        # Use word boundary matching
        pattern = r'\b' + re.escape(adverb) + r'\b'
        for match in re.finditer(pattern, text_lower):
            # Check if this position overlaps with already found adverb
            start, end = match.start(), match.end()
            overlaps = any(
                existing.start_char <= start < existing.end_char or
                existing.start_char < end <= existing.end_char
                for existing in results
            )
            if overlaps:
                continue

            # Extract original case from text
            original_text = text[start:end]

            # Attempt to identify scope target (simple heuristic)
            scope_target = _identify_scope_target(text, start, end)

            results.append(FocusingAdverb(
                text=original_text,
                start_char=start,
                end_char=end,
                operator_type=ScopeOperator.FOCUS,
                invertible=False,
                focus_type=_get_focus_type(adverb),
                scope_target=scope_target,
            ))

    # Sort by position in text
    results.sort(key=lambda x: x.start_char)
    return results


def _identify_scope_target(text: str, adverb_start: int, adverb_end: int) -> Optional[str]:
    """Heuristically identify the scope target for a focusing adverb.

    Uses simple pattern matching. For robust scope resolution,
    this should integrate with Phase 2 dependency parsing.
    """
    # Get text after the adverb
    after_adverb = text[adverb_end:].strip()

    if not after_adverb:
        return None

    # Simple heuristic: first word or phrase after adverb
    # This is a placeholder - real implementation would use dependency parse
    words = after_adverb.split()
    if words:
        # If first word is capitalized (proper noun), likely the target
        first_word = words[0].rstrip(".,!?;:")
        return first_word

    return None


def get_scope_bindings(text: str) -> list[ScopeBinding]:
    """Get possible scope bindings for focusing adverbs in text.

    For ambiguous positions, enumerates multiple possible interpretations
    with confidence scores.

    Args:
        text: Input text with focusing adverb(s)

    Returns:
        List of ScopeBinding objects representing possible interpretations

    Examples:
        >>> bindings = get_scope_bindings("Only John passed.")
        >>> len(bindings)
        1
        >>> bindings[0].target
        'John'
    """
    adverbs = identify_focusing_adverbs(text)
    if not adverbs:
        return []

    bindings: list[ScopeBinding] = []

    for adverb in adverbs:
        # Get text after adverb for analysis
        after = text[adverb.end_char:].strip()
        before = text[:adverb.start_char].strip()

        # Heuristic scope binding analysis
        # Position 1: Sentence-initial "Only X ..." - clear subject scope
        if not before or before in {".", "!", "?"}:
            words = after.split()
            if words:
                target = words[0].rstrip(".,!?;:")
                bindings.append(ScopeBinding(
                    target=target,
                    confidence=0.9,
                    position="subject",
                ))
        else:
            # Mid-sentence position - potentially ambiguous
            words_after = after.split()
            words_before = before.split()

            # Primary interpretation: immediate following constituent
            if words_after:
                primary_target = words_after[0].rstrip(".,!?;:")
                bindings.append(ScopeBinding(
                    target=primary_target,
                    confidence=0.7,
                    position="object" if len(words_before) > 1 else "verb",
                ))

            # Secondary interpretations for ambiguous positions
            # This is simplified - full implementation needs parse tree
            if len(words_after) > 2:
                # Possible PP attachment scope
                for i, word in enumerate(words_after[1:], 1):
                    if word.lower() in {"on", "in", "at", "to", "for"}:
                        # Prepositional phrase could be scope
                        pp_target = " ".join(words_after[i:i+3])
                        bindings.append(ScopeBinding(
                            target=pp_target,
                            confidence=0.3,
                            position="adjunct",
                        ))
                        break

    return bindings
