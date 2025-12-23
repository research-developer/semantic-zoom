"""NSM-49: Node creation (nouns with attributes).

Creates semantic graph nodes from noun phrases with:
- Word ID span and attributes
- Adjective vector attachment
- Pronoun antecedent links
- Proper noun entity types
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid

# Lazy load spacy for text processing
_nlp = None


def _get_nlp():
    """Lazy load spacy model."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, try downloading
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


class NodeType(Enum):
    """Types of semantic nodes."""
    NOUN = auto()
    PRONOUN = auto()
    PROPER_NOUN = auto()


@dataclass
class SemanticNode:
    """A node in the semantic graph representing a noun phrase.
    
    Attributes:
        node_id: Unique identifier for this node
        text: The full text of the noun phrase
        span: Character offsets (start, end) in source text
        node_type: NOUN, PRONOUN, or PROPER_NOUN
        head: The head noun of the phrase
        adjective_vector: Ordered list of adjectives modifying the noun
        attributes: Additional attributes dictionary
        entity_type: For proper nouns, the NER entity type (PERSON, ORG, GPE, etc.)
        antecedent_id: For pronouns, the node_id of the antecedent
        is_resolved: For pronouns, whether antecedent is resolved
    """
    node_id: str
    text: str
    span: tuple[int, int]
    node_type: NodeType
    head: Optional[str] = None
    adjective_vector: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    entity_type: Optional[str] = None
    antecedent_id: Optional[str] = None
    is_resolved: bool = False
    
    def __post_init__(self):
        if self.node_type == NodeType.PRONOUN:
            self.is_resolved = self.antecedent_id is not None
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, SemanticNode):
            return False
        return self.node_id == other.node_id


def _generate_node_id() -> str:
    """Generate a unique node identifier."""
    return f"node_{uuid.uuid4().hex[:12]}"


def _determine_node_type(pos: str) -> NodeType:
    """Determine node type from POS tag."""
    if pos == "PRON":
        return NodeType.PRONOUN
    elif pos == "PROPN":
        return NodeType.PROPER_NOUN
    else:
        return NodeType.NOUN


def create_node(
    text: str,
    span: tuple[int, int],
    pos: str,
    head_text: Optional[str] = None,
    adjectives: Optional[list[str]] = None,
    entity_type: Optional[str] = None,
    antecedent_id: Optional[str] = None,
    attributes: Optional[dict] = None,
) -> SemanticNode:
    """Create a semantic node from a noun phrase.
    
    Args:
        text: The full text of the noun phrase
        span: Character offsets (start, end) in source text
        pos: Part-of-speech tag (NOUN, PRON, PROPN)
        head_text: The head noun of the phrase
        adjectives: List of adjectives modifying the noun
        entity_type: NER entity type for proper nouns
        antecedent_id: Node ID of antecedent for pronouns
        attributes: Additional attributes dictionary
        
    Returns:
        SemanticNode with all properties set
    """
    node_type = _determine_node_type(pos)
    
    return SemanticNode(
        node_id=_generate_node_id(),
        text=text,
        span=span,
        node_type=node_type,
        head=head_text or text,
        adjective_vector=adjectives or [],
        attributes=attributes or {},
        entity_type=entity_type,
        antecedent_id=antecedent_id,
        is_resolved=(antecedent_id is not None) if node_type == NodeType.PRONOUN else False,
    )


def create_nodes_from_text(text: str) -> list[SemanticNode]:
    """Extract all noun nodes from text using NLP.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of SemanticNode objects for all noun phrases
    """
    nlp = _get_nlp()
    doc = nlp(text)
    
    nodes = []
    
    # Process noun chunks (noun phrases)
    for chunk in doc.noun_chunks:
        # Get head noun
        head = chunk.root
        
        # Collect adjectives
        adjectives = []
        for token in chunk:
            if token.pos_ == "ADJ":
                adjectives.append(token.text)
        
        # Determine POS from head
        pos = head.pos_
        
        # Get entity type if available
        entity_type = None
        if head.ent_type_:
            entity_type = head.ent_type_
        
        node = create_node(
            text=chunk.text,
            span=(chunk.start_char, chunk.end_char),
            pos=pos,
            head_text=head.text,
            adjectives=adjectives,
            entity_type=entity_type,
        )
        nodes.append(node)
    
    # Also process standalone pronouns not in chunks
    for token in doc:
        if token.pos_ == "PRON":
            # Check if already covered by a chunk
            covered = any(
                node.span[0] <= token.idx < node.span[1]
                for node in nodes
            )
            if not covered:
                node = create_node(
                    text=token.text,
                    span=(token.idx, token.idx + len(token.text)),
                    pos="PRON",
                )
                nodes.append(node)
    
    return nodes
