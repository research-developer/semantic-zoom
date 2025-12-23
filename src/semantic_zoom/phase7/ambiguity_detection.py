"""Ambiguity detection (NSM-58).

This module provides ambiguity detection that:
- Identifies structural ambiguities (PP-attachment, coordination, pronoun, quantifier, negation scope)
- Enumerates multiple parse interpretations with confidence
- Lists possible antecedents for pronoun ambiguity
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Set
import spacy

# Load spaCy model
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    _nlp = spacy.load("en_core_web_sm")


class AmbiguityType(Enum):
    """Types of structural ambiguity."""
    PP_ATTACHMENT = auto()
    COORDINATION = auto()
    PRONOUN = auto()
    QUANTIFIER_SCOPE = auto()
    NEGATION_SCOPE = auto()


@dataclass
class Interpretation:
    """A possible interpretation of an ambiguous structure.

    Attributes:
        description: Human-readable description of this interpretation
        confidence: Confidence score (0.0 to 1.0)
        attachment_point: For PP-attachment, the token being attached to
    """
    description: str
    confidence: float
    attachment_point: Optional[str] = None


@dataclass
class Antecedent:
    """A possible antecedent for a pronoun.

    Attributes:
        text: The antecedent text
        token_id: Token ID in the parse
        confidence: Confidence score (0.0 to 1.0)
    """
    text: str
    token_id: int
    confidence: float


@dataclass
class Ambiguity:
    """A detected ambiguity in the text.

    Attributes:
        ambiguity_type: Type of ambiguity
        span: Character span (start, end) of ambiguous region
        text: The ambiguous text
        interpretations: List of possible interpretations
        possible_antecedents: For pronoun ambiguity, list of possible antecedents
    """
    ambiguity_type: AmbiguityType
    span: Tuple[int, int]
    text: str
    interpretations: List[Interpretation] = field(default_factory=list)
    possible_antecedents: Optional[List[Antecedent]] = None


@dataclass
class AmbiguityResult:
    """Result of ambiguity detection.

    Attributes:
        text: The original text
        ambiguities: List of detected ambiguities
    """
    text: str
    ambiguities: List[Ambiguity] = field(default_factory=list)


def detect_ambiguities(text: str) -> AmbiguityResult:
    """Detect structural ambiguities in text.

    Args:
        text: Text to analyze

    Returns:
        AmbiguityResult with detected ambiguities
    """
    doc = _nlp(text)
    ambiguities: List[Ambiguity] = []

    # Detect PP-attachment ambiguities
    ambiguities.extend(_detect_pp_attachment(doc))

    # Detect coordination ambiguities
    ambiguities.extend(_detect_coordination(doc))

    # Detect pronoun ambiguities
    ambiguities.extend(_detect_pronoun_ambiguity(doc))

    # Detect quantifier scope ambiguities
    ambiguities.extend(_detect_quantifier_scope(doc))

    # Detect negation scope ambiguities
    ambiguities.extend(_detect_negation_scope(doc))

    return AmbiguityResult(text=text, ambiguities=ambiguities)


def _detect_pp_attachment(doc) -> List[Ambiguity]:
    """Detect prepositional phrase attachment ambiguities.

    Classic example: "I saw the man with the telescope"
    - PP "with the telescope" could attach to verb "saw" or noun "man"
    """
    ambiguities = []

    for token in doc:
        # Look for prepositions
        if token.pos_ == "ADP" and token.dep_ == "prep":
            # Get the PP span
            pp_tokens = list(token.subtree)
            if not pp_tokens:
                continue

            pp_start = min(t.idx for t in pp_tokens)
            pp_end = max(t.idx + len(t.text) for t in pp_tokens)
            pp_text = doc.text[pp_start:pp_end]

            # Check for potential attachment ambiguity
            head = token.head
            potential_attachments = []

            # Current attachment point
            potential_attachments.append((head.text, head.pos_))

            # Look for other potential attachment points
            # If attached to noun, check if there's a verb that could take it
            # If attached to verb, check if there's a noun that could take it
            if head.pos_ == "NOUN":
                # Look for verb ancestor
                for ancestor in head.ancestors:
                    if ancestor.pos_ == "VERB":
                        potential_attachments.append((ancestor.text, ancestor.pos_))
                        break
            elif head.pos_ == "VERB":
                # Look for noun object/complement
                for child in head.children:
                    if child.pos_ == "NOUN" and child.i < token.i:
                        potential_attachments.append((child.text, child.pos_))
                        break

            # Only report as ambiguous if multiple attachment points
            if len(potential_attachments) >= 2:
                interpretations = []
                confidence = 1.0 / len(potential_attachments)

                for attach_text, attach_pos in potential_attachments:
                    if attach_pos == "VERB":
                        desc = f"PP '{pp_text}' attaches to verb '{attach_text}'"
                    else:
                        desc = f"PP '{pp_text}' attaches to noun '{attach_text}'"
                    interpretations.append(Interpretation(
                        description=desc,
                        confidence=confidence,
                        attachment_point=attach_text
                    ))

                ambiguities.append(Ambiguity(
                    ambiguity_type=AmbiguityType.PP_ATTACHMENT,
                    span=(pp_start, pp_end),
                    text=pp_text,
                    interpretations=interpretations
                ))

    return ambiguities


def _detect_coordination(doc) -> List[Ambiguity]:
    """Detect coordination scope ambiguities.

    Example: "Old men and women"
    - "Old" could modify just "men" or "men and women"
    """
    ambiguities = []

    for token in doc:
        # Look for coordinating conjunctions
        if token.pos_ == "CCONJ" and token.dep_ == "cc":
            # Get the coordinated elements
            head = token.head
            conj_children = [c for c in head.head.children if c.dep_ == "conj"] if head.head else []

            if not conj_children:
                conj_children = [c for c in head.children if c.dep_ == "conj"]

            if conj_children:
                # Check if there's a modifier that could scope differently
                modifiers = []
                for child in head.children:
                    if child.dep_ in ("amod", "advmod") and child.i < head.i:
                        modifiers.append(child)

                if modifiers:
                    # There's a pre-modifier that could scope over coordination
                    for mod in modifiers:
                        coord_text = doc[head.i:conj_children[-1].i + 1].text

                        interpretations = [
                            Interpretation(
                                description=f"'{mod.text}' modifies only '{head.text}'",
                                confidence=0.5
                            ),
                            Interpretation(
                                description=f"'{mod.text}' modifies '{coord_text}'",
                                confidence=0.5
                            )
                        ]

                        span_start = mod.idx
                        span_end = conj_children[-1].idx + len(conj_children[-1].text)

                        ambiguities.append(Ambiguity(
                            ambiguity_type=AmbiguityType.COORDINATION,
                            span=(span_start, span_end),
                            text=doc.text[span_start:span_end],
                            interpretations=interpretations
                        ))

    return ambiguities


def _detect_pronoun_ambiguity(doc) -> List[Ambiguity]:
    """Detect pronoun reference ambiguities.

    Example: "John told Bill that he was wrong"
    - "he" could refer to John or Bill
    """
    ambiguities = []

    # Collect potential antecedents (proper nouns and nouns)
    potential_antecedents = []
    for token in doc:
        if token.pos_ in ("PROPN", "NOUN") and token.dep_ not in ("compound",):
            potential_antecedents.append(token)

    # Look for pronouns
    for token in doc:
        if token.pos_ == "PRON" and token.text.lower() in (
            "he", "she", "it", "they", "him", "her", "them",
            "his", "hers", "its", "their", "theirs"
        ):
            # Find compatible antecedents (preceding the pronoun)
            compatible = []
            for ant in potential_antecedents:
                if ant.i < token.i:  # Antecedent must precede pronoun
                    # Simple gender/number compatibility check
                    if _is_compatible(token, ant):
                        compatible.append(ant)

            # Only ambiguous if multiple compatible antecedents
            if len(compatible) >= 2:
                confidence = 1.0 / len(compatible)
                antecedent_list = [
                    Antecedent(
                        text=ant.text,
                        token_id=ant.i,
                        confidence=confidence
                    )
                    for ant in compatible
                ]

                interpretations = [
                    Interpretation(
                        description=f"'{token.text}' refers to '{ant.text}'",
                        confidence=confidence
                    )
                    for ant in compatible
                ]

                ambiguities.append(Ambiguity(
                    ambiguity_type=AmbiguityType.PRONOUN,
                    span=(token.idx, token.idx + len(token.text)),
                    text=token.text,
                    interpretations=interpretations,
                    possible_antecedents=antecedent_list
                ))

    return ambiguities


def _is_compatible(pronoun, antecedent) -> bool:
    """Check if pronoun and antecedent are compatible."""
    # Simple heuristic - in real implementation would check gender/number
    pron_lower = pronoun.text.lower()

    # Singular pronouns
    singular = {"he", "she", "it", "him", "her", "his", "hers", "its"}
    # Plural pronouns
    plural = {"they", "them", "their", "theirs"}

    if pron_lower in singular:
        # Check if antecedent is likely singular
        return antecedent.tag_ not in ("NNS", "NNPS")
    elif pron_lower in plural:
        # Check if antecedent could be plural or group
        return True  # Be permissive

    return True


def _detect_quantifier_scope(doc) -> List[Ambiguity]:
    """Detect quantifier scope ambiguities.

    Example: "Every student read a book"
    - Surface scope: each student read their own book
    - Inverse scope: all students read the same book
    """
    ambiguities = []

    # Universal quantifiers
    universal = {"every", "each", "all"}
    # Existential quantifiers
    existential = {"a", "an", "some"}

    universal_tokens = []
    existential_tokens = []

    for token in doc:
        if token.text.lower() in universal and token.dep_ == "det":
            universal_tokens.append(token)
        elif token.text.lower() in existential and token.dep_ == "det":
            existential_tokens.append(token)

    # Check for scope interaction
    for univ in universal_tokens:
        for exist in existential_tokens:
            if univ.i < exist.i:  # Universal comes before existential
                # Get the noun phrases
                univ_np = univ.head.text if univ.head else univ.text
                exist_np = exist.head.text if exist.head else exist.text

                span_start = univ.idx
                span_end = exist.head.idx + len(exist.head.text) if exist.head else exist.idx + len(exist.text)

                interpretations = [
                    Interpretation(
                        description=f"Surface scope: each {univ_np} has their own {exist_np}",
                        confidence=0.6  # Surface scope often preferred
                    ),
                    Interpretation(
                        description=f"Inverse scope: all {univ_np}s share the same {exist_np}",
                        confidence=0.4
                    )
                ]

                ambiguities.append(Ambiguity(
                    ambiguity_type=AmbiguityType.QUANTIFIER_SCOPE,
                    span=(span_start, span_end),
                    text=doc.text[span_start:span_end],
                    interpretations=interpretations
                ))

    return ambiguities


def _detect_negation_scope(doc) -> List[Ambiguity]:
    """Detect negation scope ambiguities.

    Example: "John didn't leave because he was tired"
    - Narrow scope: he left, but not because of tiredness
    - Wide scope: he didn't leave (reason: tiredness)
    """
    ambiguities = []

    for token in doc:
        # Look for negation
        if token.dep_ == "neg" or token.text.lower() in ("not", "n't", "never"):
            head = token.head

            # Check for adverbial clause that could be in/out of negation scope
            for child in head.children:
                if child.dep_ in ("advcl", "prep") and child.i > token.i:
                    # There's a clause/PP after the negation
                    clause_tokens = list(child.subtree)
                    if clause_tokens:
                        clause_start = min(t.idx for t in clause_tokens)
                        clause_end = max(t.idx + len(t.text) for t in clause_tokens)
                        clause_text = doc.text[clause_start:clause_end]

                        span_start = token.idx
                        span_end = clause_end

                        interpretations = [
                            Interpretation(
                                description=f"Negation scopes over main clause only ('{clause_text}' is outside negation)",
                                confidence=0.5
                            ),
                            Interpretation(
                                description=f"Negation scopes over '{clause_text}' as well",
                                confidence=0.5
                            )
                        ]

                        ambiguities.append(Ambiguity(
                            ambiguity_type=AmbiguityType.NEGATION_SCOPE,
                            span=(span_start, span_end),
                            text=doc.text[span_start:span_end],
                            interpretations=interpretations
                        ))
                        break

            # Check for quantifier-negation interaction
            for other in doc:
                if other.text.lower() in ("all", "every", "each") and other.dep_ == "det":
                    if other.i < token.i:  # Quantifier before negation
                        np_text = other.head.text if other.head else other.text

                        interpretations = [
                            Interpretation(
                                description=f"Not all {np_text}s (some do)",
                                confidence=0.5
                            ),
                            Interpretation(
                                description=f"All {np_text}s don't (none do)",
                                confidence=0.5
                            )
                        ]

                        span_start = other.idx
                        span_end = token.idx + len(token.text)

                        ambiguities.append(Ambiguity(
                            ambiguity_type=AmbiguityType.NEGATION_SCOPE,
                            span=(span_start, span_end),
                            text=doc.text[span_start:span_end],
                            interpretations=interpretations
                        ))

    return ambiguities
