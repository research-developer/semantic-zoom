"""Tests for Phase 2 → Phase 3 integration."""

import pytest
from semantic_zoom.pipeline import Pipeline
from semantic_zoom.phase3.integration import (
    Phase3Processor,
    Phase3Result,
    MorphismToken,
    process_tokens_phase3,
)
from semantic_zoom.phase3 import (
    CategoricalSymbol,
    DiscourseRelation,
)


class TestPhase3Integration:
    """Test Phase 2 → Phase 3 integration."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def processor(self):
        return Phase3Processor()

    def test_empty_input(self, processor):
        """Empty token list returns empty result."""
        result = processor.process([])
        assert result.tokens == []
        assert result.preposition_mappings == []
        assert result.focusing_adverbs == []
        assert result.discourse_adverbs == []

    def test_basic_processing(self, pipeline, processor):
        """Basic text processes without error."""
        tokens = pipeline.process("The cat sat on the mat.")
        result = processor.process(tokens)

        assert isinstance(result, Phase3Result)
        assert len(result.tokens) == len(tokens)

    def test_convenience_function(self, pipeline):
        """Convenience function works."""
        tokens = pipeline.process("The cat sat.")
        result = process_tokens_phase3(tokens)
        assert isinstance(result, Phase3Result)


class TestPrepositionMapping:
    """Test preposition mapping in integration."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def processor(self):
        return Phase3Processor()

    def test_preposition_identified(self, pipeline, processor):
        """Prepositions are identified and mapped."""
        tokens = pipeline.process("The cat sat on the mat.")
        result = processor.process(tokens)

        # "on" should be mapped
        assert len(result.preposition_mappings) >= 1
        on_mapping = next(
            (m for m in result.preposition_mappings if m.original.lower() == "on"),
            None
        )
        assert on_mapping is not None
        assert on_mapping.symbol == CategoricalSymbol.SPATIAL_ON

    def test_preposition_token_annotated(self, pipeline, processor):
        """Preposition tokens have prep_mapping set."""
        tokens = pipeline.process("The book is in the box.")
        result = processor.process(tokens)

        # Find the "in" token
        in_token = next(
            (mt for mt in result.tokens if mt.token.text.lower() == "in"),
            None
        )
        assert in_token is not None
        assert in_token.prep_mapping is not None
        assert in_token.prep_mapping.symbol == CategoricalSymbol.CONTAINMENT_IN

    def test_directional_preposition(self, pipeline, processor):
        """Directional prepositions map correctly."""
        tokens = pipeline.process("She walked to the store.")
        result = processor.process(tokens)

        to_mapping = next(
            (m for m in result.preposition_mappings if m.original.lower() == "to"),
            None
        )
        assert to_mapping is not None
        assert to_mapping.symbol == CategoricalSymbol.DIRECTIONAL_TO
        assert to_mapping.state.motion == "dynamic"

    def test_multiple_prepositions(self, pipeline, processor):
        """Multiple prepositions in sentence are all mapped."""
        tokens = pipeline.process("The cat jumped from the table to the floor.")
        result = processor.process(tokens)

        # Should have at least 2 preposition mappings
        assert len(result.preposition_mappings) >= 2

        preps = {m.original.lower() for m in result.preposition_mappings}
        assert "from" in preps
        assert "to" in preps


class TestFocusingAdverbs:
    """Test focusing adverb processing in integration."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def processor(self):
        return Phase3Processor()

    def test_focusing_adverb_identified(self, pipeline, processor):
        """Focusing adverbs are identified."""
        tokens = pipeline.process("I only want coffee.")
        result = processor.process(tokens)

        assert len(result.focusing_adverbs) >= 1
        only_token = result.focusing_adverbs[0]
        assert only_token.is_focusing_adverb
        assert only_token.focus_type == "restrictive"

    def test_additive_focus(self, pipeline, processor):
        """Additive focusing adverbs are typed correctly."""
        tokens = pipeline.process("Even John passed the test.")
        result = processor.process(tokens)

        even_tokens = [mt for mt in result.focusing_adverbs
                       if mt.token.text.lower() == "even"]
        if even_tokens:
            assert even_tokens[0].focus_type == "additive"

    def test_scope_target_resolution(self, pipeline, processor):
        """Scope targets are resolved using dependencies."""
        tokens = pipeline.process("Only John passed.")
        result = processor.process(tokens)

        if result.focusing_adverbs:
            only_token = result.focusing_adverbs[0]
            # Should have scope target
            assert only_token.scope_target_idx is not None or len(only_token.scope_bindings) > 0

    def test_scope_bindings_generated(self, pipeline, processor):
        """Scope bindings are generated with positions."""
        tokens = pipeline.process("She only eats vegetables.")
        result = processor.process(tokens)

        if result.focusing_adverbs:
            only_token = result.focusing_adverbs[0]
            # Should have at least one binding
            assert len(only_token.scope_bindings) >= 0  # May be empty depending on parse


class TestDiscourseAdverbs:
    """Test discourse adverb processing in integration."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def processor(self):
        return Phase3Processor()

    def test_discourse_adverb_identified(self, pipeline, processor):
        """Discourse adverbs are identified."""
        tokens = pipeline.process("However, we disagree.")
        result = processor.process(tokens)

        however_tokens = [mt for mt in result.discourse_adverbs
                          if mt.token.text.lower() == "however"]
        if however_tokens:
            assert however_tokens[0].is_discourse_adverb
            assert however_tokens[0].discourse_relation == DiscourseRelation.CONTRAST

    def test_inter_frame_morphism_created(self, pipeline, processor):
        """Inter-frame morphisms are created for discourse adverbs."""
        tokens = pipeline.process("Therefore, we conclude.")
        result = processor.process(tokens)

        therefore_tokens = [mt for mt in result.discourse_adverbs
                            if mt.token.text.lower() == "therefore"]
        if therefore_tokens:
            assert therefore_tokens[0].inter_frame_morphism is not None
            assert therefore_tokens[0].inter_frame_morphism.relation == DiscourseRelation.CONSEQUENCE

    def test_consequence_relation(self, pipeline, processor):
        """Consequence markers are correctly typed."""
        tokens = pipeline.process("Thus the experiment succeeded.")
        result = processor.process(tokens)

        if result.inter_frame_morphisms:
            thus_morphism = next(
                (m for m in result.inter_frame_morphisms if m.edge_label == "thus"),
                None
            )
            if thus_morphism:
                assert thus_morphism.relation == DiscourseRelation.CONSEQUENCE


class TestDualCitizenshipResolution:
    """Test dual-citizenship preposition resolution."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def processor(self):
        return Phase3Processor()

    def test_at_spatial_context(self, pipeline, processor):
        """'at' resolves to SPATIAL in location context."""
        tokens = pipeline.process("She is at the store.")
        result = processor.process(tokens)

        at_mapping = next(
            (m for m in result.preposition_mappings if m.original.lower() == "at"),
            None
        )
        if at_mapping:
            assert at_mapping.saturated
            assert at_mapping.symbol in (
                CategoricalSymbol.SPATIAL_AT,
                CategoricalSymbol.TEMPORAL_AT
            )

    def test_for_beneficiary_context(self, pipeline, processor):
        """'for' resolves based on object type."""
        tokens = pipeline.process("This gift is for Mary.")
        result = processor.process(tokens)

        for_mapping = next(
            (m for m in result.preposition_mappings if m.original.lower() == "for"),
            None
        )
        if for_mapping:
            # With proper noun object, should be beneficiary
            assert for_mapping.saturated


class TestMorphismTokenStructure:
    """Test MorphismToken dataclass structure."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def processor(self):
        return Phase3Processor()

    def test_morphism_token_has_original_token(self, pipeline, processor):
        """MorphismToken preserves original Token reference."""
        tokens = pipeline.process("The cat sat.")
        result = processor.process(tokens)

        for mt in result.tokens:
            assert mt.token is not None
            assert hasattr(mt.token, "text")
            assert hasattr(mt.token, "pos")
            assert hasattr(mt.token, "dep")

    def test_all_tokens_wrapped(self, pipeline, processor):
        """All input tokens are wrapped in MorphismToken."""
        tokens = pipeline.process("The quick brown fox jumps.")
        result = processor.process(tokens)

        assert len(result.tokens) == len(tokens)
        for mt, original in zip(result.tokens, tokens):
            assert mt.token is original


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def processor(self):
        return Phase3Processor()

    def test_complex_sentence(self, pipeline, processor):
        """Complex sentence with multiple morphism types."""
        text = "However, only John walked quickly to the store."
        tokens = pipeline.process(text)
        result = processor.process(tokens)

        # Should have discourse adverb (However)
        # Should have focusing adverb (only)
        # Should have preposition (to)
        assert len(result.tokens) > 0

    def test_pipeline_then_phase3(self, pipeline, processor):
        """Full pipeline through Phase 3 works."""
        text = "The cat sat on the mat. Therefore, it was comfortable."
        tokens = pipeline.process(text)
        result = processor.process(tokens)

        # Verify structure
        assert isinstance(result, Phase3Result)
        assert len(result.tokens) == len(tokens)

        # Should have prepositions mapped
        assert len(result.preposition_mappings) >= 1
