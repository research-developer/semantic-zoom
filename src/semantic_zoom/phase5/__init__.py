"""Phase 5: Graph Construction.

NSM-49: Node creation (nouns with attributes)
NSM-50: Edge creation (verbs with modifiers)
NSM-51: Morphism attachment (prepositions, adverbs)
NSM-52: Inter-frame linking
"""
from semantic_zoom.phase5.nodes import (
    NodeType,
    SemanticNode,
    create_node,
    create_nodes_from_text,
)
from semantic_zoom.phase5.edges import (
    AdverbAttachment,
    AdverbTier,
    EdgeType,
    SemanticEdge,
    NULL_NODE_ID,
    create_edge,
    create_edges_from_text,
)
from semantic_zoom.phase5.morphisms import (
    AttachmentLevel,
    MorphismAttachment,
    FocusingAdverbAttachment,
    AdverbMorphismAttachment,
    attach_preposition,
    attach_focusing_adverb,
    attach_adverb_morphism,
)
from semantic_zoom.phase5.inter_frame import (
    FrameLink,
    LinkType,
    create_explicit_link,
    create_implicit_link,
    detect_frame_relations,
)

__all__ = [
    # NSM-49: Nodes
    "NodeType",
    "SemanticNode",
    "create_node",
    "create_nodes_from_text",
    # NSM-50: Edges
    "AdverbAttachment",
    "AdverbTier",
    "EdgeType",
    "SemanticEdge",
    "NULL_NODE_ID",
    "create_edge",
    "create_edges_from_text",
    # NSM-51: Morphisms
    "AttachmentLevel",
    "MorphismAttachment",
    "FocusingAdverbAttachment",
    "AdverbMorphismAttachment",
    "attach_preposition",
    "attach_focusing_adverb",
    "attach_adverb_morphism",
    # NSM-52: Inter-frame
    "FrameLink",
    "LinkType",
    "create_explicit_link",
    "create_implicit_link",
    "detect_frame_relations",
]
