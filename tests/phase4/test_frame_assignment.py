"""Tests for NSM-46: FrameNet frame assignment to verb predicates.

Acceptance Criteria:
- Verbs mapped to candidate FrameNet frames
- Polysemous verbs disambiguated with confidence
- Frame lexical unit and elements available
"""
import pytest
from semantic_zoom.phase4.framenet_assignment import (
    FrameAssignment,
    FrameCandidate,
    assign_frame,
    disambiguate_polysemous,
)


class TestFrameAssignment:
    """Test verb to frame mapping."""

    def test_simple_verb_frame_assignment(self):
        """Single-sense verbs should map to one candidate frame."""
        result = assign_frame("cook", context="She cooks dinner every night")

        assert isinstance(result, FrameAssignment)
        assert len(result.candidates) >= 1
        assert result.best_frame is not None
        assert "Apply_heat" in [c.frame_name for c in result.candidates] or \
               "Absorb_heat" in [c.frame_name for c in result.candidates]

    def test_frame_has_lexical_unit(self):
        """Frame assignment should include the lexical unit info."""
        result = assign_frame("run", context="The athlete runs fast")

        assert result.best_frame is not None
        assert result.best_frame.lexical_unit is not None
        assert "run" in result.best_frame.lexical_unit.lower()

    def test_frame_has_elements(self):
        """Frame assignment should expose frame elements."""
        result = assign_frame("give", context="She gave him a book")

        assert result.best_frame is not None
        assert len(result.best_frame.frame_elements) > 0
        # Frame should have Core frame elements
        element_names = [fe.name for fe in result.best_frame.frame_elements]
        core_elements = [fe for fe in result.best_frame.frame_elements if fe.core_type == "Core"]
        assert len(core_elements) > 0, "Frame should have Core elements"

    def test_confidence_score_present(self):
        """Frame candidates should have confidence scores."""
        result = assign_frame("break", context="The window broke")

        assert result.best_frame is not None
        assert 0.0 <= result.best_frame.confidence <= 1.0
        for candidate in result.candidates:
            assert 0.0 <= candidate.confidence <= 1.0


class TestPolysemy:
    """Test disambiguation of polysemous verbs."""

    def test_polysemous_verb_has_multiple_candidates(self):
        """Polysemous verbs should return multiple frame candidates."""
        # 'run' is highly polysemous in FrameNet
        result = assign_frame("run", context=None)

        assert len(result.candidates) > 1, "Polysemous verb should have multiple candidates"

    def test_context_disambiguates_polysemy(self):
        """Context should help select the correct frame for polysemous verbs."""
        # 'run' in motion context
        motion_result = assign_frame("run", context="The athlete runs around the track")
        # 'run' in operating context
        operate_result = assign_frame("run", context="He runs the company")

        # Different contexts should yield different top frames
        assert motion_result.best_frame.frame_name != operate_result.best_frame.frame_name

    def test_disambiguation_with_sentence_embedding(self):
        """Disambiguation should use semantic similarity."""
        result = disambiguate_polysemous(
            verb="play",
            candidates=None,  # Will fetch from FrameNet
            context="The musicians play jazz"
        )

        assert result.best_frame is not None
        assert result.best_frame.confidence >= 0.0  # Should have valid confidence
        # Frame should be assigned (disambiguation worked)
        assert len(result.candidates) > 0

    def test_fallback_without_context(self):
        """Without context, should return candidates ranked by frequency/salience."""
        result = assign_frame("set", context=None)

        assert result.best_frame is not None
        # Should still pick a reasonable default
        assert result.best_frame.confidence >= 0.0

    def test_unknown_verb_handling(self):
        """Unknown verbs should return empty candidates or raise."""
        result = assign_frame("xyzzy", context="I xyzzy the widget")

        assert len(result.candidates) == 0 or result.best_frame is None


class TestFrameCandidate:
    """Test the FrameCandidate data structure."""

    def test_frame_candidate_fields(self):
        """FrameCandidate should have required fields."""
        result = assign_frame("walk", context="She walks to school")

        if result.best_frame:
            candidate = result.best_frame
            assert hasattr(candidate, "frame_name")
            assert hasattr(candidate, "frame_id")
            assert hasattr(candidate, "lexical_unit")
            assert hasattr(candidate, "frame_elements")
            assert hasattr(candidate, "confidence")
            assert hasattr(candidate, "definition")

    def test_frame_elements_have_core_type(self):
        """Frame elements should indicate core/peripheral/extra-thematic."""
        result = assign_frame("move", context="The car moves forward")

        if result.best_frame:
            for fe in result.best_frame.frame_elements:
                assert hasattr(fe, "name")
                assert hasattr(fe, "core_type")
                assert fe.core_type in ["Core", "Peripheral", "Extra-Thematic"]
