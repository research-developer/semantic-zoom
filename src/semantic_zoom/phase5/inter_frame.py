"""NSM-52: Inter-frame linking.

Creates links between frames based on:
- Discourse adverbs (explicit links)
- Implicit relations (lower confidence)
- FrameNet frame-to-frame relations
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid

from semantic_zoom.phase3 import DiscourseRelation

# Try to import FrameNet for relation detection
try:
    from nltk.corpus import framenet as fn
    HAS_FRAMENET = True
except ImportError:
    HAS_FRAMENET = False


class LinkType(Enum):
    """Types of inter-frame links."""
    EXPLICIT = auto()   # Created from discourse markers
    IMPLICIT = auto()   # Inferred from context
    FRAMENET = auto()   # From FrameNet relations


@dataclass
class FrameLink:
    """A link between two frames in the semantic graph.
    
    Attributes:
        link_id: Unique identifier
        source_frame_id: ID of the source frame
        target_frame_id: ID of the target frame
        relation: Discourse relation type
        link_type: EXPLICIT, IMPLICIT, or FRAMENET
        confidence: Confidence score (0.0 to 1.0)
        discourse_marker: The marker that triggered the link (if explicit)
        evidence: Evidence for implicit links
        framenet_relation: FrameNet relation type (if FRAMENET)
        bidirectional: Whether link goes both ways
    """
    link_id: str
    source_frame_id: str
    target_frame_id: str
    relation: DiscourseRelation
    link_type: LinkType
    confidence: float
    discourse_marker: Optional[str] = None
    evidence: Optional[str] = None
    framenet_relation: Optional[str] = None
    bidirectional: bool = False


def _generate_link_id() -> str:
    """Generate a unique link identifier."""
    return f"link_{uuid.uuid4().hex[:12]}"


# Confidence scores for explicit discourse markers
_MARKER_CONFIDENCE = {
    "however": 0.95,
    "therefore": 0.95,
    "moreover": 0.90,
    "then": 0.85,
    "nevertheless": 0.95,
    "thus": 0.90,
    "consequently": 0.95,
    "furthermore": 0.90,
    "additionally": 0.85,
}


def create_explicit_link(
    source_frame_id: str,
    target_frame_id: str,
    discourse_marker: str,
    relation: DiscourseRelation,
    bidirectional: bool = False,
) -> FrameLink:
    """Create an explicit link from a discourse marker.
    
    Args:
        source_frame_id: ID of the source frame
        target_frame_id: ID of the target frame
        discourse_marker: The discourse marker text
        relation: The discourse relation type
        bidirectional: Whether link is bidirectional
        
    Returns:
        FrameLink with high confidence
    """
    # Get confidence from marker or default to 0.8
    confidence = _MARKER_CONFIDENCE.get(discourse_marker.lower(), 0.8)
    
    return FrameLink(
        link_id=_generate_link_id(),
        source_frame_id=source_frame_id,
        target_frame_id=target_frame_id,
        relation=relation,
        link_type=LinkType.EXPLICIT,
        confidence=confidence,
        discourse_marker=discourse_marker,
        bidirectional=bidirectional,
    )


def create_implicit_link(
    source_frame_id: str,
    target_frame_id: str,
    inferred_relation: DiscourseRelation,
    evidence: Optional[str] = None,
    confidence: float = 0.5,
) -> FrameLink:
    """Create an implicit link inferred from context.
    
    Implicit links have lower confidence than explicit ones.
    
    Args:
        source_frame_id: ID of the source frame
        target_frame_id: ID of the target frame
        inferred_relation: The inferred relation type
        evidence: Reason for the inference
        confidence: Confidence score (default 0.5)
        
    Returns:
        FrameLink with lower confidence
    """
    return FrameLink(
        link_id=_generate_link_id(),
        source_frame_id=source_frame_id,
        target_frame_id=target_frame_id,
        relation=inferred_relation,
        link_type=LinkType.IMPLICIT,
        confidence=min(confidence, 0.7),  # Cap implicit at 0.7
        evidence=evidence,
        bidirectional=False,
    )


def detect_frame_relations(
    frame_names: list[str],
    frame_ids: list[str],
) -> list[FrameLink]:
    """Detect FrameNet frame-to-frame relations.
    
    Uses FrameNet's relation types:
    - Inheritance: Child inherits from parent
    - Using: Frame uses another frame
    - Subframe: Frame is subframe of another
    - Precedes: Temporal ordering
    
    Args:
        frame_names: List of FrameNet frame names
        frame_ids: Corresponding frame IDs
        
    Returns:
        List of FrameLinks for detected relations
    """
    if not HAS_FRAMENET or len(frame_names) < 2:
        return []
    
    links = []
    
    # Map names to IDs
    name_to_id = dict(zip(frame_names, frame_ids))
    
    # Check each pair of frames
    for i, name1 in enumerate(frame_names):
        for j, name2 in enumerate(frame_names):
            if i >= j:
                continue
            
            try:
                frame1 = fn.frame(name1)
                frame2 = fn.frame(name2)
                
                # Check frame relations
                for rel in frame1.frameRelations:
                    rel_type = rel.type.name
                    
                    # Check if related to frame2
                    if hasattr(rel, 'superFrameName') and rel.superFrameName == name2:
                        links.append(FrameLink(
                            link_id=_generate_link_id(),
                            source_frame_id=name_to_id[name1],
                            target_frame_id=name_to_id[name2],
                            relation=DiscourseRelation.SEQUENCE,  # Default
                            link_type=LinkType.FRAMENET,
                            confidence=0.8,
                            framenet_relation=rel_type,
                        ))
                    elif hasattr(rel, 'subFrameName') and rel.subFrameName == name2:
                        links.append(FrameLink(
                            link_id=_generate_link_id(),
                            source_frame_id=name_to_id[name1],
                            target_frame_id=name_to_id[name2],
                            relation=DiscourseRelation.SEQUENCE,
                            link_type=LinkType.FRAMENET,
                            confidence=0.8,
                            framenet_relation=rel_type,
                        ))
            except Exception:
                # Frame lookup failed
                continue
    
    return links
