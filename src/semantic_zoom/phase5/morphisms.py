"""NSM-51: Morphism attachment (prepositions, adverbs).

Attaches morphisms at different levels:
- NODE: Prepositions modifying nouns
- EDGE: Prepositions modifying verbs/relations
- FRAME: Prepositions at frame level
- PROPOSITION: Prepositions at proposition level

Also handles:
- Focusing adverb scope operators (invertible=False)
- Tiered adverb morphisms
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid

from semantic_zoom.phase3 import (
    CategoricalSymbol,
    PrepositionMapping,
    SymbolState,
)
from semantic_zoom.phase5.edges import AdverbTier


class AttachmentLevel(Enum):
    """Levels at which morphisms can attach."""
    NODE = auto()        # Modifies a noun node
    EDGE = auto()        # Modifies a verb edge
    FRAME = auto()       # Modifies a frame instance
    PROPOSITION = auto() # Modifies entire proposition


@dataclass
class MorphismAttachment:
    """A morphism attached to a graph element.
    
    Attributes:
        attachment_id: Unique identifier
        target_id: ID of the node/edge/frame being modified
        level: Attachment level (NODE, EDGE, FRAME, PROPOSITION)
        symbol: The categorical symbol from Phase 3
        state: Symbol state (polarity, motion, inverse)
        original_text: The original preposition text
        is_dual_citizen: Whether preposition has multiple meanings
        saturated: Whether dual-citizenship resolved
    """
    attachment_id: str
    target_id: str
    level: AttachmentLevel
    symbol: CategoricalSymbol
    state: SymbolState
    original_text: str
    is_dual_citizen: bool = False
    saturated: bool = True


@dataclass
class FocusingAdverbAttachment:
    """A focusing adverb scope operator attachment.
    
    Focusing adverbs (only, even, just, merely) create scope
    operators that are NOT invertible.
    
    Attributes:
        attachment_id: Unique identifier
        target_id: ID of the focused element
        adverb: The focusing adverb text
        scope_start: Start of scope span
        scope_end: End of scope span
        invertible: Always False for focusing adverbs
    """
    attachment_id: str
    target_id: str
    adverb: str
    scope_start: int
    scope_end: int
    invertible: bool = False


@dataclass
class AdverbMorphismAttachment:
    """A general adverb morphism at a tier level.
    
    Attributes:
        attachment_id: Unique identifier
        target_id: ID of the edge being modified
        adverb: The adverb text
        tier: The adverb tier (MANNER, TEMPORAL, etc.)
    """
    attachment_id: str
    target_id: str
    adverb: str
    tier: AdverbTier


def _generate_attachment_id() -> str:
    """Generate a unique attachment identifier."""
    return f"attach_{uuid.uuid4().hex[:12]}"


def attach_preposition(
    prep_mapping: PrepositionMapping,
    target_id: str,
    level: AttachmentLevel,
) -> MorphismAttachment:
    """Attach a preposition morphism to a graph element.
    
    Args:
        prep_mapping: PrepositionMapping from Phase 3
        target_id: ID of the node/edge/frame to attach to
        level: The attachment level
        
    Returns:
        MorphismAttachment object
    """
    return MorphismAttachment(
        attachment_id=_generate_attachment_id(),
        target_id=target_id,
        level=level,
        symbol=prep_mapping.symbol,
        state=prep_mapping.state,
        original_text=prep_mapping.original,
        is_dual_citizen=prep_mapping.is_dual_citizen,
        saturated=prep_mapping.saturated,
    )


def attach_focusing_adverb(
    adverb: str,
    target_id: str,
    scope_start: int,
    scope_end: int,
) -> FocusingAdverbAttachment:
    """Attach a focusing adverb as a scope operator.
    
    Focusing adverbs create scope operators that highlight
    or restrict focus to certain elements. They are NOT
    invertible transformations.
    
    Args:
        adverb: The focusing adverb (only, even, just, etc.)
        target_id: ID of the focused element
        scope_start: Character offset where scope starts
        scope_end: Character offset where scope ends
        
    Returns:
        FocusingAdverbAttachment with invertible=False
    """
    return FocusingAdverbAttachment(
        attachment_id=_generate_attachment_id(),
        target_id=target_id,
        adverb=adverb,
        scope_start=scope_start,
        scope_end=scope_end,
        invertible=False,  # Always False for focusing adverbs
    )


def attach_adverb_morphism(
    adverb: str,
    target_id: str,
    tier: AdverbTier,
) -> AdverbMorphismAttachment:
    """Attach an adverb morphism at a specific tier.
    
    Args:
        adverb: The adverb text
        target_id: ID of the edge to modify
        tier: The adverb tier level
        
    Returns:
        AdverbMorphismAttachment object
    """
    return AdverbMorphismAttachment(
        attachment_id=_generate_attachment_id(),
        target_id=target_id,
        adverb=adverb,
        tier=tier,
    )
