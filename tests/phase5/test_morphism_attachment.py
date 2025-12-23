"""Tests for NSM-51: Morphism attachment (prepositions, adverbs).

Acceptance Criteria:
- Preposition symbols attached at NODE, EDGE, FRAME, or PROPOSITION level
- Focusing adverb scope operators attached with invertible=False
- Adverbs attached at appropriate tier level
"""
import pytest
from semantic_zoom.phase5.nodes import create_node
from semantic_zoom.phase5.edges import create_edge, AdverbTier
from semantic_zoom.phase5.morphisms import (
    AttachmentLevel,
    MorphismAttachment,
    attach_preposition,
    attach_focusing_adverb,
    attach_adverb_morphism,
)
from semantic_zoom.phase3 import (
    CategoricalSymbol,
    map_preposition,
)


class TestPrepositionMorphism:
    """Test preposition symbol attachment."""

    def test_attach_to_node(self):
        """Prepositions can attach at NODE level."""
        node = create_node(text="house", span=(0, 5), pos="NOUN")
        prep_mapping = map_preposition("in")

        attachment = attach_preposition(
            prep_mapping=prep_mapping,
            target_id=node.node_id,
            level=AttachmentLevel.NODE
        )

        assert attachment.level == AttachmentLevel.NODE
        assert attachment.target_id == node.node_id
        assert attachment.symbol == CategoricalSymbol.CONTAINMENT_IN

    def test_attach_to_edge(self):
        """Prepositions can attach at EDGE level."""
        subject = create_node(text="she", span=(0, 3), pos="PRON")
        obj = create_node(text="park", span=(15, 19), pos="NOUN")
        edge = create_edge(verb="walked", subject_node=subject, object_node=obj, span=(4, 10))

        prep_mapping = map_preposition("to")

        attachment = attach_preposition(
            prep_mapping=prep_mapping,
            target_id=edge.edge_id,
            level=AttachmentLevel.EDGE
        )

        assert attachment.level == AttachmentLevel.EDGE
        assert attachment.symbol == CategoricalSymbol.DIRECTIONAL_TO

    def test_attach_to_frame(self):
        """Prepositions can attach at FRAME level."""
        prep_mapping = map_preposition("during")

        attachment = attach_preposition(
            prep_mapping=prep_mapping,
            target_id="frame_001",
            level=AttachmentLevel.FRAME
        )

        assert attachment.level == AttachmentLevel.FRAME
        assert attachment.symbol == CategoricalSymbol.TEMPORAL_DURING

    def test_attach_to_proposition(self):
        """Prepositions can attach at PROPOSITION level."""
        prep_mapping = map_preposition("before")

        attachment = attach_preposition(
            prep_mapping=prep_mapping,
            target_id="prop_001",
            level=AttachmentLevel.PROPOSITION
        )

        assert attachment.level == AttachmentLevel.PROPOSITION

    def test_attachment_preserves_state(self):
        """Attachment should preserve preposition state."""
        prep_mapping = map_preposition("into")

        attachment = attach_preposition(
            prep_mapping=prep_mapping,
            target_id="node_001",
            level=AttachmentLevel.NODE
        )

        assert attachment.state.motion == "dynamic"

    def test_dual_citizen_attachment(self):
        """Dual-citizen prepositions should be marked unsaturated."""
        prep_mapping = map_preposition("at")  # at is dual-citizen

        attachment = attach_preposition(
            prep_mapping=prep_mapping,
            target_id="node_001",
            level=AttachmentLevel.NODE
        )

        assert attachment.is_dual_citizen is True
        assert attachment.saturated is False


class TestFocusingAdverb:
    """Test focusing adverb scope operator attachment."""

    def test_attach_focusing_adverb(self):
        """Focusing adverbs should attach as scope operators."""
        attachment = attach_focusing_adverb(
            adverb="only",
            target_id="node_001",
            scope_start=0,
            scope_end=10
        )

        assert attachment is not None
        assert attachment.adverb == "only"
        assert attachment.invertible is False  # Focusing adverbs not invertible

    def test_focusing_adverb_not_invertible(self):
        """Focusing adverb operators should be marked invertible=False."""
        attachment = attach_focusing_adverb(
            adverb="even",
            target_id="edge_001",
            scope_start=5,
            scope_end=20
        )

        assert attachment.invertible is False

    def test_focusing_adverb_scope(self):
        """Attachment should record scope boundaries."""
        attachment = attach_focusing_adverb(
            adverb="just",
            target_id="node_001",
            scope_start=10,
            scope_end=25
        )

        assert attachment.scope_start == 10
        assert attachment.scope_end == 25

    def test_various_focusing_adverbs(self):
        """Various focusing adverbs should all be attachable."""
        focusing_adverbs = ["only", "even", "just", "merely", "simply", "especially"]

        for adverb in focusing_adverbs:
            attachment = attach_focusing_adverb(
                adverb=adverb,
                target_id="test_node",
                scope_start=0,
                scope_end=10
            )
            assert attachment.adverb == adverb
            assert attachment.invertible is False


class TestAdverbMorphism:
    """Test general adverb morphism attachment at tier levels."""

    def test_attach_manner_adverb(self):
        """Manner adverbs should attach at MANNER tier."""
        attachment = attach_adverb_morphism(
            adverb="quickly",
            target_id="edge_001",
            tier=AdverbTier.MANNER
        )

        assert attachment.tier == AdverbTier.MANNER
        assert attachment.adverb == "quickly"

    def test_attach_temporal_adverb(self):
        """Temporal adverbs should attach at TEMPORAL tier."""
        attachment = attach_adverb_morphism(
            adverb="yesterday",
            target_id="edge_001",
            tier=AdverbTier.TEMPORAL
        )

        assert attachment.tier == AdverbTier.TEMPORAL

    def test_attach_locative_adverb(self):
        """Locative adverbs should attach at LOCATIVE tier."""
        attachment = attach_adverb_morphism(
            adverb="here",
            target_id="edge_001",
            tier=AdverbTier.LOCATIVE
        )

        assert attachment.tier == AdverbTier.LOCATIVE

    def test_attach_frequency_adverb(self):
        """Frequency adverbs should attach at FREQUENCY tier."""
        attachment = attach_adverb_morphism(
            adverb="always",
            target_id="edge_001",
            tier=AdverbTier.FREQUENCY
        )

        assert attachment.tier == AdverbTier.FREQUENCY

    def test_attach_degree_adverb(self):
        """Degree adverbs should attach at DEGREE tier."""
        attachment = attach_adverb_morphism(
            adverb="very",
            target_id="edge_001",
            tier=AdverbTier.DEGREE
        )

        assert attachment.tier == AdverbTier.DEGREE


class TestMorphismAttachment:
    """Test MorphismAttachment data structure."""

    def test_attachment_has_required_fields(self):
        """MorphismAttachment should have all required fields."""
        prep_mapping = map_preposition("on")
        attachment = attach_preposition(
            prep_mapping=prep_mapping,
            target_id="test",
            level=AttachmentLevel.NODE
        )

        assert hasattr(attachment, "target_id")
        assert hasattr(attachment, "level")
        assert hasattr(attachment, "symbol")

    def test_attachment_equality(self):
        """Attachments with same properties should be comparable."""
        prep1 = map_preposition("in")
        prep2 = map_preposition("in")

        att1 = attach_preposition(prep1, "node_001", AttachmentLevel.NODE)
        att2 = attach_preposition(prep2, "node_001", AttachmentLevel.NODE)

        # Should have same properties
        assert att1.symbol == att2.symbol
        assert att1.level == att2.level
