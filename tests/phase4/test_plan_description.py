"""Tests for NSM-48: Plan vs Description classification.

Acceptance Criteria:
- Propositions tagged: PLAN, DESCRIPTION, or HYBRID
- Dynamic frames -> PLAN, Stative frames -> DESCRIPTION
- Aspectual transformations may shift to HYBRID
"""
import pytest
from semantic_zoom.phase4.plan_description import (
    PropositionType,
    classify_proposition,
    ClassificationResult,
)


class TestPlanClassification:
    """Test classification of PLAN propositions."""

    def test_dynamic_verb_is_plan(self):
        """Dynamic/eventive verbs should classify as PLAN."""
        result = classify_proposition(
            verb="run",
            frame_name="Self_motion",
            context="The athlete runs to the finish line"
        )

        assert result.proposition_type == PropositionType.PLAN

    def test_action_verb_is_plan(self):
        """Action verbs with agents should be PLAN."""
        result = classify_proposition(
            verb="build",
            frame_name="Building",
            context="They build the house"
        )

        assert result.proposition_type == PropositionType.PLAN

    def test_causative_is_plan(self):
        """Causative constructions should be PLAN."""
        result = classify_proposition(
            verb="break",
            frame_name="Cause_harm",
            context="She breaks the vase"
        )

        assert result.proposition_type == PropositionType.PLAN

    def test_future_tense_is_plan(self):
        """Future-oriented statements should favor PLAN."""
        result = classify_proposition(
            verb="go",
            frame_name="Motion",
            context="He will go to the store tomorrow",
            tense="future"
        )

        assert result.proposition_type == PropositionType.PLAN

    def test_imperative_is_plan(self):
        """Imperative mood should be PLAN."""
        result = classify_proposition(
            verb="open",
            frame_name="Opening",
            context="Open the door!",
            mood="imperative"
        )

        assert result.proposition_type == PropositionType.PLAN


class TestDescriptionClassification:
    """Test classification of DESCRIPTION propositions."""

    def test_stative_verb_is_description(self):
        """Stative verbs should classify as DESCRIPTION."""
        result = classify_proposition(
            verb="know",
            frame_name="Awareness",
            context="She knows the answer"
        )

        assert result.proposition_type == PropositionType.DESCRIPTION

    def test_be_is_description(self):
        """'Be' copula should be DESCRIPTION."""
        result = classify_proposition(
            verb="be",
            frame_name="Entity",
            context="The sky is blue"
        )

        assert result.proposition_type == PropositionType.DESCRIPTION

    def test_possession_is_description(self):
        """Possession verbs should be DESCRIPTION."""
        result = classify_proposition(
            verb="have",
            frame_name="Possession",
            context="She has a car"
        )

        assert result.proposition_type == PropositionType.DESCRIPTION

    def test_perception_is_description(self):
        """Perception/experience verbs should be DESCRIPTION."""
        result = classify_proposition(
            verb="see",
            frame_name="Perception_experience",
            context="I see the mountain"
        )

        assert result.proposition_type == PropositionType.DESCRIPTION

    def test_relation_is_description(self):
        """Relational predicates should be DESCRIPTION."""
        result = classify_proposition(
            verb="resemble",
            frame_name="Similarity",
            context="She resembles her mother"
        )

        assert result.proposition_type == PropositionType.DESCRIPTION


class TestHybridClassification:
    """Test HYBRID classification for aspectual transformations."""

    def test_progressive_stative_is_hybrid(self):
        """Progressive aspect on typically stative verbs can shift to HYBRID."""
        result = classify_proposition(
            verb="know",
            frame_name="Awareness",
            context="She is getting to know him",
            aspect="progressive"
        )

        assert result.proposition_type in [PropositionType.HYBRID, PropositionType.PLAN]

    def test_habitual_dynamic_is_hybrid(self):
        """Habitual aspect on dynamic verbs can shift toward DESCRIPTION."""
        result = classify_proposition(
            verb="run",
            frame_name="Self_motion",
            context="He runs every morning",
            aspect="habitual"
        )

        assert result.proposition_type in [PropositionType.HYBRID, PropositionType.DESCRIPTION]

    def test_inchoative_is_hybrid(self):
        """Inchoative (begin to X) constructions may be HYBRID."""
        result = classify_proposition(
            verb="understand",
            frame_name="Grasp",
            context="She is beginning to understand",
            aspect="inchoative"
        )

        assert result.proposition_type in [PropositionType.HYBRID, PropositionType.PLAN]

    def test_resultative_is_hybrid(self):
        """Resultative constructions may blend PLAN and DESCRIPTION."""
        result = classify_proposition(
            verb="break",
            frame_name="Cause_change_of_phase",
            context="The vase is broken",
            aspect="resultative"
        )

        # Resultative often emphasizes the state (DESCRIPTION) from an event
        assert result.proposition_type in [PropositionType.HYBRID, PropositionType.DESCRIPTION]


class TestClassificationResult:
    """Test ClassificationResult structure."""

    def test_result_has_required_fields(self):
        """ClassificationResult should have all required fields."""
        result = classify_proposition(
            verb="walk",
            frame_name="Self_motion",
            context="She walks"
        )

        assert hasattr(result, "proposition_type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reasoning")

    def test_confidence_in_range(self):
        """Confidence should be between 0 and 1."""
        result = classify_proposition(
            verb="run",
            frame_name="Self_motion",
            context="He runs"
        )

        assert 0.0 <= result.confidence <= 1.0

    def test_reasoning_provided(self):
        """Reasoning should explain the classification."""
        result = classify_proposition(
            verb="know",
            frame_name="Awareness",
            context="I know the truth"
        )

        assert result.reasoning is not None
        assert len(result.reasoning) > 0


class TestFrameBasedClassification:
    """Test that frame type influences classification."""

    def test_dynamic_frame_detection(self):
        """Dynamic frames should be detected and favor PLAN."""
        # Motion is inherently dynamic
        result = classify_proposition(
            verb="walk",
            frame_name="Motion",
            context="She walks to school"
        )

        assert result.proposition_type == PropositionType.PLAN

    def test_stative_frame_detection(self):
        """Stative frames should be detected and favor DESCRIPTION."""
        # Possession is inherently stative
        result = classify_proposition(
            verb="own",
            frame_name="Possession",
            context="He owns the house"
        )

        assert result.proposition_type == PropositionType.DESCRIPTION

    def test_frame_overrides_verb_default(self):
        """Frame semantics should influence beyond just verb choice."""
        # 'see' can be active perception (PLAN) or passive experience (DESCRIPTION)
        active_result = classify_proposition(
            verb="look",
            frame_name="Perception_active",
            context="She looks at the painting"
        )

        passive_result = classify_proposition(
            verb="see",
            frame_name="Perception_experience",
            context="She sees the painting"
        )

        # Active perception is more PLAN-like
        assert active_result.proposition_type in [PropositionType.PLAN, PropositionType.HYBRID]
        # Passive perception is more DESCRIPTION-like
        assert passive_result.proposition_type == PropositionType.DESCRIPTION
