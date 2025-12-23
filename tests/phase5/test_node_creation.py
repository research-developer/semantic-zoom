"""Tests for NSM-49: Node creation (nouns with attributes).

Acceptance Criteria:
- Noun phrases create nodes with word ID span and attributes
- Adjective vector attached to nodes
- Pronouns marked with antecedent link if resolved
- Proper nouns marked with entity type
"""
import pytest
from semantic_zoom.phase5.nodes import (
    SemanticNode,
    NodeType,
    create_node,
    create_nodes_from_text,
)


class TestNodeCreation:
    """Test basic node creation from noun phrases."""

    def test_simple_noun_creates_node(self):
        """A simple noun should create a node."""
        node = create_node(
            text="cat",
            span=(0, 3),
            pos="NOUN"
        )

        assert isinstance(node, SemanticNode)
        assert node.text == "cat"
        assert node.span == (0, 3)
        assert node.node_type == NodeType.NOUN

    def test_node_has_unique_id(self):
        """Each node should have a unique identifier."""
        node1 = create_node(text="dog", span=(0, 3), pos="NOUN")
        node2 = create_node(text="cat", span=(4, 7), pos="NOUN")

        assert node1.node_id != node2.node_id
        assert node1.node_id is not None

    def test_noun_phrase_creates_node(self):
        """A noun phrase should create a single node with span."""
        node = create_node(
            text="the big red ball",
            span=(0, 16),
            pos="NOUN",
            head_text="ball"
        )

        assert node.text == "the big red ball"
        assert node.head == "ball"
        assert node.span == (0, 16)

    def test_node_has_attributes_dict(self):
        """Nodes should have an attributes dictionary."""
        node = create_node(text="house", span=(0, 5), pos="NOUN")

        assert hasattr(node, "attributes")
        assert isinstance(node.attributes, dict)


class TestAdjectiveVector:
    """Test adjective vector attachment to nodes."""

    def test_adjectives_attached_to_node(self):
        """Adjectives modifying a noun should be attached as vector."""
        node = create_node(
            text="big red ball",
            span=(0, 12),
            pos="NOUN",
            head_text="ball",
            adjectives=["big", "red"]
        )

        assert node.adjective_vector is not None
        assert "big" in node.adjective_vector
        assert "red" in node.adjective_vector

    def test_adjective_order_preserved(self):
        """Adjective order should be preserved in vector."""
        node = create_node(
            text="beautiful old Italian ceramic vase",
            span=(0, 34),
            pos="NOUN",
            head_text="vase",
            adjectives=["beautiful", "old", "Italian", "ceramic"]
        )

        # Order should match English adjective ordering rules
        assert node.adjective_vector == ["beautiful", "old", "Italian", "ceramic"]

    def test_empty_adjective_vector(self):
        """Nodes without adjectives should have empty vector."""
        node = create_node(text="house", span=(0, 5), pos="NOUN")

        assert node.adjective_vector == [] or node.adjective_vector is None


class TestPronounNodes:
    """Test pronoun handling with antecedent links."""

    def test_pronoun_creates_node(self):
        """Pronouns should create nodes with PRONOUN type."""
        node = create_node(
            text="she",
            span=(0, 3),
            pos="PRON"
        )

        assert node.node_type == NodeType.PRONOUN

    def test_pronoun_with_antecedent(self):
        """Resolved pronouns should link to antecedent."""
        # First create the antecedent node
        antecedent = create_node(
            text="Mary",
            span=(0, 4),
            pos="PROPN"
        )

        # Then create pronoun with antecedent reference
        pronoun = create_node(
            text="she",
            span=(10, 13),
            pos="PRON",
            antecedent_id=antecedent.node_id
        )

        assert pronoun.antecedent_id == antecedent.node_id
        assert pronoun.is_resolved is True

    def test_unresolved_pronoun(self):
        """Unresolved pronouns should be marked as such."""
        pronoun = create_node(
            text="it",
            span=(0, 2),
            pos="PRON",
            antecedent_id=None
        )

        assert pronoun.antecedent_id is None
        assert pronoun.is_resolved is False


class TestProperNounNodes:
    """Test proper noun handling with entity types."""

    def test_proper_noun_creates_node(self):
        """Proper nouns should create nodes with PROPER_NOUN type."""
        node = create_node(
            text="London",
            span=(0, 6),
            pos="PROPN"
        )

        assert node.node_type == NodeType.PROPER_NOUN

    def test_proper_noun_with_entity_type(self):
        """Proper nouns should have entity type annotation."""
        node = create_node(
            text="Microsoft",
            span=(0, 9),
            pos="PROPN",
            entity_type="ORG"
        )

        assert node.entity_type == "ORG"

    def test_person_entity_type(self):
        """Person names should have PERSON entity type."""
        node = create_node(
            text="John Smith",
            span=(0, 10),
            pos="PROPN",
            entity_type="PERSON"
        )

        assert node.entity_type == "PERSON"

    def test_location_entity_type(self):
        """Location names should have GPE/LOC entity type."""
        node = create_node(
            text="Paris",
            span=(0, 5),
            pos="PROPN",
            entity_type="GPE"
        )

        assert node.entity_type == "GPE"


class TestNodesFromText:
    """Test creating nodes from full text input."""

    def test_extract_nodes_from_sentence(self):
        """Should extract all noun nodes from a sentence."""
        nodes = create_nodes_from_text("The cat sat on the mat.")

        # Should have at least "cat" and "mat" nodes
        node_texts = [n.text.lower() for n in nodes]
        assert any("cat" in t for t in node_texts)
        assert any("mat" in t for t in node_texts)

    def test_nodes_have_correct_spans(self):
        """Extracted nodes should have correct character spans."""
        text = "The dog runs."
        nodes = create_nodes_from_text(text)

        for node in nodes:
            # Verify span matches text
            assert text[node.span[0]:node.span[1]].lower() == node.text.lower() or \
                   node.text.lower() in text[node.span[0]:node.span[1]].lower()


class TestSemanticNode:
    """Test SemanticNode data structure."""

    def test_node_has_required_fields(self):
        """SemanticNode should have all required fields."""
        node = create_node(text="test", span=(0, 4), pos="NOUN")

        assert hasattr(node, "node_id")
        assert hasattr(node, "text")
        assert hasattr(node, "span")
        assert hasattr(node, "node_type")
        assert hasattr(node, "attributes")
        assert hasattr(node, "adjective_vector")

    def test_node_equality(self):
        """Nodes with same ID should be equal."""
        node1 = create_node(text="test", span=(0, 4), pos="NOUN")
        # Create reference to same node
        node2 = node1

        assert node1 == node2

    def test_node_hashable(self):
        """Nodes should be hashable for use in sets/dicts."""
        node = create_node(text="test", span=(0, 4), pos="NOUN")

        # Should not raise
        node_set = {node}
        assert node in node_set
