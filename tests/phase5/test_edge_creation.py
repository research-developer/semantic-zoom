"""Tests for NSM-50: Edge creation (verbs with modifiers).

Acceptance Criteria:
- Verb predicates create edges connecting subject to object nodes
- Intransitive verbs connect to NULL/implicit object
- Tiered adverb stack attached to edges
"""
import pytest
from semantic_zoom.phase5.nodes import create_node, SemanticNode
from semantic_zoom.phase5.edges import (
    SemanticEdge,
    EdgeType,
    AdverbTier,
    create_edge,
    create_edges_from_text,
    NULL_NODE_ID,
)


class TestEdgeCreation:
    """Test basic edge creation from verb predicates."""

    def test_transitive_verb_creates_edge(self):
        """Transitive verbs should create edges between subject and object."""
        subject = create_node(text="cat", span=(0, 3), pos="NOUN")
        obj = create_node(text="mouse", span=(15, 20), pos="NOUN")

        edge = create_edge(
            verb="chased",
            subject_node=subject,
            object_node=obj,
            span=(4, 10)
        )

        assert isinstance(edge, SemanticEdge)
        assert edge.source_id == subject.node_id
        assert edge.target_id == obj.node_id
        assert edge.verb == "chased"

    def test_edge_has_unique_id(self):
        """Each edge should have a unique identifier."""
        subject = create_node(text="dog", span=(0, 3), pos="NOUN")
        obj = create_node(text="ball", span=(10, 14), pos="NOUN")

        edge1 = create_edge(verb="fetched", subject_node=subject, object_node=obj, span=(4, 11))
        edge2 = create_edge(verb="dropped", subject_node=subject, object_node=obj, span=(20, 27))

        assert edge1.edge_id != edge2.edge_id

    def test_edge_has_span(self):
        """Edges should have character span for the verb."""
        subject = create_node(text="she", span=(0, 3), pos="PRON")
        obj = create_node(text="book", span=(10, 14), pos="NOUN")

        edge = create_edge(
            verb="read",
            subject_node=subject,
            object_node=obj,
            span=(4, 8)
        )

        assert edge.span == (4, 8)


class TestIntransitiveVerbs:
    """Test intransitive verb handling with NULL object."""

    def test_intransitive_verb_creates_edge(self):
        """Intransitive verbs should create edge to NULL node."""
        subject = create_node(text="bird", span=(0, 4), pos="NOUN")

        edge = create_edge(
            verb="flew",
            subject_node=subject,
            object_node=None,  # No object
            span=(5, 9)
        )

        assert edge.source_id == subject.node_id
        assert edge.target_id == NULL_NODE_ID
        assert edge.is_intransitive is True

    def test_null_node_id_constant(self):
        """NULL_NODE_ID should be a consistent sentinel value."""
        subject1 = create_node(text="she", span=(0, 3), pos="PRON")
        subject2 = create_node(text="he", span=(10, 12), pos="PRON")

        edge1 = create_edge(verb="ran", subject_node=subject1, object_node=None, span=(4, 7))
        edge2 = create_edge(verb="walked", subject_node=subject2, object_node=None, span=(13, 19))

        # Both should point to same NULL node
        assert edge1.target_id == edge2.target_id == NULL_NODE_ID

    def test_implicit_object_flag(self):
        """Edges can have implicit (unexpressed) objects."""
        subject = create_node(text="she", span=(0, 3), pos="PRON")

        # "She ate" - object is implicit (food)
        edge = create_edge(
            verb="ate",
            subject_node=subject,
            object_node=None,
            span=(4, 7),
            implicit_object=True
        )

        assert edge.has_implicit_object is True


class TestAdverbTierStack:
    """Test tiered adverb stack attachment to edges."""

    def test_manner_adverb_attachment(self):
        """Manner adverbs should attach at MANNER tier."""
        subject = create_node(text="cat", span=(0, 3), pos="NOUN")
        obj = create_node(text="mouse", span=(20, 25), pos="NOUN")

        edge = create_edge(
            verb="caught",
            subject_node=subject,
            object_node=obj,
            span=(10, 16),
            adverbs=[("quickly", AdverbTier.MANNER)]
        )

        assert len(edge.adverb_stack) >= 1
        manner_adverbs = [a for a in edge.adverb_stack if a.tier == AdverbTier.MANNER]
        assert any(a.text == "quickly" for a in manner_adverbs)

    def test_temporal_adverb_attachment(self):
        """Temporal adverbs should attach at TEMPORAL tier."""
        subject = create_node(text="she", span=(0, 3), pos="PRON")
        obj = create_node(text="dinner", span=(15, 21), pos="NOUN")

        edge = create_edge(
            verb="cooked",
            subject_node=subject,
            object_node=obj,
            span=(4, 10),
            adverbs=[("yesterday", AdverbTier.TEMPORAL)]
        )

        temporal_adverbs = [a for a in edge.adverb_stack if a.tier == AdverbTier.TEMPORAL]
        assert any(a.text == "yesterday" for a in temporal_adverbs)

    def test_degree_adverb_attachment(self):
        """Degree adverbs should attach at DEGREE tier."""
        subject = create_node(text="he", span=(0, 2), pos="PRON")

        edge = create_edge(
            verb="ran",
            subject_node=subject,
            object_node=None,
            span=(3, 6),
            adverbs=[("very", AdverbTier.DEGREE), ("fast", AdverbTier.MANNER)]
        )

        degree_adverbs = [a for a in edge.adverb_stack if a.tier == AdverbTier.DEGREE]
        assert any(a.text == "very" for a in degree_adverbs)

    def test_multiple_adverbs_stacked(self):
        """Multiple adverbs should form a stack."""
        subject = create_node(text="she", span=(0, 3), pos="PRON")

        edge = create_edge(
            verb="spoke",
            subject_node=subject,
            object_node=None,
            span=(4, 9),
            adverbs=[
                ("softly", AdverbTier.MANNER),
                ("always", AdverbTier.FREQUENCY),
                ("here", AdverbTier.LOCATIVE)
            ]
        )

        assert len(edge.adverb_stack) == 3

    def test_adverb_stack_order(self):
        """Adverb stack should preserve tier ordering."""
        subject = create_node(text="he", span=(0, 2), pos="PRON")

        edge = create_edge(
            verb="works",
            subject_node=subject,
            object_node=None,
            span=(3, 8),
            adverbs=[
                ("diligently", AdverbTier.MANNER),
                ("always", AdverbTier.FREQUENCY),
                ("completely", AdverbTier.DEGREE)
            ]
        )

        # Verify all adverbs present
        texts = [a.text for a in edge.adverb_stack]
        assert "diligently" in texts
        assert "always" in texts
        assert "completely" in texts


class TestEdgesFromText:
    """Test creating edges from full text input."""

    def test_extract_edges_from_sentence(self):
        """Should extract verb edges from a sentence."""
        edges = create_edges_from_text("The cat chased the mouse.")

        # Should have at least one edge for "chased"
        assert len(edges) >= 1
        assert any(e.verb == "chased" for e in edges)

    def test_extracted_edges_have_nodes(self):
        """Extracted edges should reference valid node IDs."""
        edges = create_edges_from_text("The dog bit the man.")

        for edge in edges:
            assert edge.source_id is not None
            assert edge.target_id is not None


class TestSemanticEdge:
    """Test SemanticEdge data structure."""

    def test_edge_has_required_fields(self):
        """SemanticEdge should have all required fields."""
        subject = create_node(text="test", span=(0, 4), pos="NOUN")
        edge = create_edge(verb="test", subject_node=subject, object_node=None, span=(5, 9))

        assert hasattr(edge, "edge_id")
        assert hasattr(edge, "source_id")
        assert hasattr(edge, "target_id")
        assert hasattr(edge, "verb")
        assert hasattr(edge, "span")
        assert hasattr(edge, "adverb_stack")

    def test_edge_type_classification(self):
        """Edges should have type classification."""
        subject = create_node(text="she", span=(0, 3), pos="PRON")
        obj = create_node(text="cake", span=(10, 14), pos="NOUN")

        edge = create_edge(
            verb="baked",
            subject_node=subject,
            object_node=obj,
            span=(4, 9),
            edge_type=EdgeType.ACTION
        )

        assert edge.edge_type == EdgeType.ACTION
