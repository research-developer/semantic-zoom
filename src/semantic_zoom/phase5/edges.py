"""NSM-50: Edge creation (verbs with modifiers).

Creates semantic graph edges from verb predicates with:
- Subject to object connections
- NULL node for intransitive verbs
- Tiered adverb stack attachment
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid

from semantic_zoom.phase5.nodes import SemanticNode, create_nodes_from_text

# Sentinel value for NULL/implicit object node
NULL_NODE_ID = "NULL_NODE"

# Lazy load spacy
_nlp = None


def _get_nlp():
    """Lazy load spacy model."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


class EdgeType(Enum):
    """Types of semantic edges."""
    ACTION = auto()      # Dynamic action verbs
    STATE = auto()       # Stative verbs
    RELATION = auto()    # Relational verbs
    PERCEPTION = auto()  # Perception verbs
    COGNITION = auto()   # Mental state verbs


class AdverbTier(Enum):
    """Tiers for adverb attachment.
    
    Adverbs attach at different semantic levels:
    - MANNER: How the action is performed (quickly, carefully)
    - TEMPORAL: When the action occurs (yesterday, now)
    - LOCATIVE: Where the action occurs (here, outside)
    - FREQUENCY: How often (always, sometimes)
    - DEGREE: Intensity modifiers (very, extremely)
    - EPISTEMIC: Speaker certainty (probably, definitely)
    """
    MANNER = auto()
    TEMPORAL = auto()
    LOCATIVE = auto()
    FREQUENCY = auto()
    DEGREE = auto()
    EPISTEMIC = auto()


@dataclass
class AdverbAttachment:
    """An adverb attached to an edge at a specific tier."""
    text: str
    tier: AdverbTier
    span: Optional[tuple[int, int]] = None


@dataclass
class SemanticEdge:
    """An edge in the semantic graph representing a verb predicate.
    
    Attributes:
        edge_id: Unique identifier for this edge
        source_id: Node ID of the subject
        target_id: Node ID of the object (NULL_NODE_ID if intransitive)
        verb: The verb lemma or text
        span: Character offsets of the verb in source text
        edge_type: Classification of the edge (ACTION, STATE, etc.)
        adverb_stack: List of adverbs attached at various tiers
        is_intransitive: True if verb has no direct object
        has_implicit_object: True if object is understood but not expressed
    """
    edge_id: str
    source_id: str
    target_id: str
    verb: str
    span: tuple[int, int]
    edge_type: EdgeType = EdgeType.ACTION
    adverb_stack: list[AdverbAttachment] = field(default_factory=list)
    is_intransitive: bool = False
    has_implicit_object: bool = False
    
    def __hash__(self):
        return hash(self.edge_id)
    
    def __eq__(self, other):
        if not isinstance(other, SemanticEdge):
            return False
        return self.edge_id == other.edge_id


def _generate_edge_id() -> str:
    """Generate a unique edge identifier."""
    return f"edge_{uuid.uuid4().hex[:12]}"


def _classify_edge_type(verb: str) -> EdgeType:
    """Classify edge type based on verb."""
    stative_verbs = {"be", "have", "know", "believe", "want", "need", "like", "love"}
    perception_verbs = {"see", "hear", "feel", "smell", "taste", "watch", "notice"}
    cognition_verbs = {"think", "understand", "remember", "forget", "realize", "consider"}
    relation_verbs = {"belong", "contain", "include", "resemble", "equal", "differ"}
    
    verb_lower = verb.lower()
    
    if verb_lower in stative_verbs:
        return EdgeType.STATE
    elif verb_lower in perception_verbs:
        return EdgeType.PERCEPTION
    elif verb_lower in cognition_verbs:
        return EdgeType.COGNITION
    elif verb_lower in relation_verbs:
        return EdgeType.RELATION
    else:
        return EdgeType.ACTION


def create_edge(
    verb: str,
    subject_node: SemanticNode,
    object_node: Optional[SemanticNode],
    span: tuple[int, int],
    adverbs: Optional[list[tuple[str, AdverbTier]]] = None,
    edge_type: Optional[EdgeType] = None,
    implicit_object: bool = False,
) -> SemanticEdge:
    """Create a semantic edge from a verb predicate.
    
    Args:
        verb: The verb text
        subject_node: The subject node
        object_node: The object node (None for intransitive)
        span: Character offsets of the verb
        adverbs: List of (adverb_text, tier) tuples
        edge_type: Optional explicit edge type
        implicit_object: True if object is implicit (understood)
        
    Returns:
        SemanticEdge connecting subject to object
    """
    # Determine target
    if object_node is None:
        target_id = NULL_NODE_ID
        is_intransitive = True
    else:
        target_id = object_node.node_id
        is_intransitive = False
    
    # Build adverb stack
    adverb_stack = []
    if adverbs:
        for adverb_text, tier in adverbs:
            adverb_stack.append(AdverbAttachment(
                text=adverb_text,
                tier=tier,
            ))
    
    # Determine edge type
    if edge_type is None:
        edge_type = _classify_edge_type(verb)
    
    return SemanticEdge(
        edge_id=_generate_edge_id(),
        source_id=subject_node.node_id,
        target_id=target_id,
        verb=verb,
        span=span,
        edge_type=edge_type,
        adverb_stack=adverb_stack,
        is_intransitive=is_intransitive,
        has_implicit_object=implicit_object,
    )


def create_edges_from_text(text: str) -> list[SemanticEdge]:
    """Extract all verb edges from text using NLP.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of SemanticEdge objects for all verb predicates
    """
    nlp = _get_nlp()
    doc = nlp(text)
    
    # First create nodes
    nodes = create_nodes_from_text(text)
    node_by_span = {(n.span[0], n.span[1]): n for n in nodes}
    
    edges = []
    
    for token in doc:
        if token.pos_ == "VERB":
            # Find subject and object
            subject_node = None
            object_node = None
            adverbs = []
            
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    # Find corresponding node
                    for span, node in node_by_span.items():
                        if span[0] <= child.idx < span[1]:
                            subject_node = node
                            break
                elif child.dep_ in ("dobj", "obj"):
                    for span, node in node_by_span.items():
                        if span[0] <= child.idx < span[1]:
                            object_node = node
                            break
                elif child.dep_ == "advmod":
                    # Classify adverb tier (simplified)
                    tier = _classify_adverb_tier(child.text)
                    adverbs.append((child.text, tier))
            
            if subject_node:
                edge = create_edge(
                    verb=token.text,
                    subject_node=subject_node,
                    object_node=object_node,
                    span=(token.idx, token.idx + len(token.text)),
                    adverbs=adverbs if adverbs else None,
                )
                edges.append(edge)
    
    return edges


def _classify_adverb_tier(adverb: str) -> AdverbTier:
    """Classify adverb into a tier based on common patterns."""
    adverb_lower = adverb.lower()
    
    temporal = {"yesterday", "today", "tomorrow", "now", "then", "soon", "later", "already"}
    locative = {"here", "there", "everywhere", "nowhere", "somewhere", "inside", "outside"}
    frequency = {"always", "never", "often", "sometimes", "rarely", "usually", "seldom"}
    degree = {"very", "extremely", "quite", "rather", "somewhat", "too", "enough"}
    epistemic = {"probably", "possibly", "certainly", "definitely", "maybe", "perhaps"}
    
    if adverb_lower in temporal:
        return AdverbTier.TEMPORAL
    elif adverb_lower in locative:
        return AdverbTier.LOCATIVE
    elif adverb_lower in frequency:
        return AdverbTier.FREQUENCY
    elif adverb_lower in degree:
        return AdverbTier.DEGREE
    elif adverb_lower in epistemic:
        return AdverbTier.EPISTEMIC
    else:
        return AdverbTier.MANNER  # Default to manner
