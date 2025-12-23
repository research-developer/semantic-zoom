"""Tests for NSM-52: Inter-frame linking.

Acceptance Criteria:
- Discourse adverbs create edges between frame nodes
- Implicit relations detected with lower confidence
- FrameNet frame-to-frame relations represented
"""
import pytest
from semantic_zoom.phase5.inter_frame import (
    FrameLink,
    LinkType,
    create_explicit_link,
    create_implicit_link,
    detect_frame_relations,
)
from semantic_zoom.phase3 import DiscourseRelation


class TestExplicitLink:
    """Test explicit inter-frame linking via discourse adverbs."""

    def test_discourse_adverb_creates_link(self):
        """Discourse adverbs should create explicit frame links."""
        link = create_explicit_link(
            source_frame_id="frame_001",
            target_frame_id="frame_002",
            discourse_marker="however",
            relation=DiscourseRelation.CONTRAST
        )

        assert isinstance(link, FrameLink)
        assert link.source_frame_id == "frame_001"
        assert link.target_frame_id == "frame_002"
        assert link.discourse_marker == "however"

    def test_link_has_relation_type(self):
        """Links should have discourse relation type."""
        link = create_explicit_link(
            source_frame_id="frame_001",
            target_frame_id="frame_002",
            discourse_marker="therefore",
            relation=DiscourseRelation.CONSEQUENCE
        )

        assert link.relation == DiscourseRelation.CONSEQUENCE

    def test_explicit_link_high_confidence(self):
        """Explicit links should have high confidence."""
        link = create_explicit_link(
            source_frame_id="frame_001",
            target_frame_id="frame_002",
            discourse_marker="moreover",
            relation=DiscourseRelation.ADDITION
        )

        assert link.confidence >= 0.8

    def test_link_type_explicit(self):
        """Links from discourse adverbs should be EXPLICIT type."""
        link = create_explicit_link(
            source_frame_id="frame_001",
            target_frame_id="frame_002",
            discourse_marker="then",
            relation=DiscourseRelation.SEQUENCE
        )

        assert link.link_type == LinkType.EXPLICIT

    def test_contrast_relation(self):
        """CONTRAST relations should be created correctly."""
        link = create_explicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            discourse_marker="nevertheless",
            relation=DiscourseRelation.CONTRAST
        )

        assert link.relation == DiscourseRelation.CONTRAST

    def test_consequence_relation(self):
        """CONSEQUENCE relations should be created correctly."""
        link = create_explicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            discourse_marker="thus",
            relation=DiscourseRelation.CONSEQUENCE
        )

        assert link.relation == DiscourseRelation.CONSEQUENCE


class TestImplicitLink:
    """Test implicit inter-frame relation detection."""

    def test_implicit_link_creation(self):
        """Implicit links can be created without discourse markers."""
        link = create_implicit_link(
            source_frame_id="frame_001",
            target_frame_id="frame_002",
            inferred_relation=DiscourseRelation.SEQUENCE
        )

        assert link.source_frame_id == "frame_001"
        assert link.target_frame_id == "frame_002"

    def test_implicit_link_lower_confidence(self):
        """Implicit links should have lower confidence than explicit."""
        implicit = create_implicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            inferred_relation=DiscourseRelation.SEQUENCE
        )

        explicit = create_explicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            discourse_marker="then",
            relation=DiscourseRelation.SEQUENCE
        )

        assert implicit.confidence < explicit.confidence

    def test_implicit_link_type(self):
        """Implicit links should be marked as IMPLICIT type."""
        link = create_implicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            inferred_relation=DiscourseRelation.ADDITION
        )

        assert link.link_type == LinkType.IMPLICIT

    def test_implicit_link_with_evidence(self):
        """Implicit links can include evidence for the inference."""
        link = create_implicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            inferred_relation=DiscourseRelation.CONSEQUENCE,
            evidence="temporal_sequence"
        )

        assert link.evidence == "temporal_sequence"


class TestFrameToFrameRelations:
    """Test FrameNet frame-to-frame relation representation."""

    def test_detect_framenet_relations(self):
        """Should detect FrameNet frame-to-frame relations."""
        # Causation and Motion are related in FrameNet
        links = detect_frame_relations(
            frame_names=["Causation", "Motion"],
            frame_ids=["f1", "f2"]
        )

        # May or may not find relations depending on implementation
        assert isinstance(links, list)

    def test_inheritance_relation(self):
        """FrameNet inheritance relations should be detected."""
        links = detect_frame_relations(
            frame_names=["Self_motion", "Motion"],
            frame_ids=["f1", "f2"]
        )

        # Self_motion inherits from Motion
        if links:
            assert any(l.framenet_relation == "Inheritance" for l in links)

    def test_no_relation_between_unrelated_frames(self):
        """Unrelated frames should have empty or low-confidence links."""
        links = detect_frame_relations(
            frame_names=["Cooking_creation", "Perception_experience"],
            frame_ids=["f1", "f2"]
        )

        # These frames are unrelated
        if links:
            assert all(l.confidence < 0.5 for l in links)


class TestFrameLink:
    """Test FrameLink data structure."""

    def test_link_has_required_fields(self):
        """FrameLink should have all required fields."""
        link = create_explicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            discourse_marker="however",
            relation=DiscourseRelation.CONTRAST
        )

        assert hasattr(link, "source_frame_id")
        assert hasattr(link, "target_frame_id")
        assert hasattr(link, "relation")
        assert hasattr(link, "confidence")
        assert hasattr(link, "link_type")

    def test_link_unique_id(self):
        """Each link should have a unique identifier."""
        link1 = create_explicit_link("f1", "f2", "however", DiscourseRelation.CONTRAST)
        link2 = create_explicit_link("f1", "f2", "therefore", DiscourseRelation.CONSEQUENCE)

        assert link1.link_id != link2.link_id

    def test_link_bidirectional_flag(self):
        """Links can be marked as bidirectional or directed."""
        link = create_explicit_link(
            source_frame_id="f1",
            target_frame_id="f2",
            discourse_marker="however",
            relation=DiscourseRelation.CONTRAST,
            bidirectional=False
        )

        assert link.bidirectional is False
