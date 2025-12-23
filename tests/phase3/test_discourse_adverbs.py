"""Tests for discourse adverb → inter-frame relation mapping (NSM-45)."""
import pytest
from semantic_zoom.phase3.discourse_adverbs import (
    DiscourseAdverb,
    DiscourseRelation,
    InterFrameMorphism,
    identify_discourse_adverbs,
    map_to_inter_frame_relation,
)


class TestDiscourseAdverbIdentification:
    """Test identification of discourse adverbs."""

    @pytest.mark.parametrize("adverb,expected_relation", [
        # CONTRAST relations
        ("however", DiscourseRelation.CONTRAST),
        ("nevertheless", DiscourseRelation.CONTRAST),
        ("nonetheless", DiscourseRelation.CONTRAST),
        ("conversely", DiscourseRelation.CONTRAST),
        ("instead", DiscourseRelation.CONTRAST),
        ("rather", DiscourseRelation.CONTRAST),
        ("on the other hand", DiscourseRelation.CONTRAST),
        # CONSEQUENCE relations
        ("therefore", DiscourseRelation.CONSEQUENCE),
        ("thus", DiscourseRelation.CONSEQUENCE),
        ("hence", DiscourseRelation.CONSEQUENCE),
        ("consequently", DiscourseRelation.CONSEQUENCE),
        ("accordingly", DiscourseRelation.CONSEQUENCE),
        ("as a result", DiscourseRelation.CONSEQUENCE),
        # ADDITION relations
        ("moreover", DiscourseRelation.ADDITION),
        ("furthermore", DiscourseRelation.ADDITION),
        ("additionally", DiscourseRelation.ADDITION),
        ("also", DiscourseRelation.ADDITION),
        ("besides", DiscourseRelation.ADDITION),
        ("in addition", DiscourseRelation.ADDITION),
        # CONCESSION relations
        ("although", DiscourseRelation.CONCESSION),
        ("though", DiscourseRelation.CONCESSION),
        ("albeit", DiscourseRelation.CONCESSION),
        ("still", DiscourseRelation.CONCESSION),
        ("yet", DiscourseRelation.CONCESSION),
        # SEQUENCE relations
        ("then", DiscourseRelation.SEQUENCE),
        ("subsequently", DiscourseRelation.SEQUENCE),
        ("afterwards", DiscourseRelation.SEQUENCE),
        ("next", DiscourseRelation.SEQUENCE),
        ("finally", DiscourseRelation.SEQUENCE),
        ("meanwhile", DiscourseRelation.SEQUENCE),
        # EXEMPLIFICATION relations
        ("for example", DiscourseRelation.EXEMPLIFICATION),
        ("for instance", DiscourseRelation.EXEMPLIFICATION),
        ("namely", DiscourseRelation.EXEMPLIFICATION),
        ("specifically", DiscourseRelation.EXEMPLIFICATION),
    ])
    def test_discourse_adverb_to_relation(self, adverb: str, expected_relation: DiscourseRelation):
        """Discourse adverbs map to correct relation types."""
        result = identify_discourse_adverbs(f"{adverb.capitalize()}, we should proceed.")
        assert len(result) >= 1
        assert result[0].relation == expected_relation


class TestInterFrameMorphism:
    """Test inter-frame morphism marking."""

    def test_morphism_type(self):
        """Discourse adverbs produce INTER_FRAME_MORPHISM type."""
        result = identify_discourse_adverbs("However, we disagree.")
        assert len(result) == 1
        morphism = map_to_inter_frame_relation(result[0])
        assert isinstance(morphism, InterFrameMorphism)

    def test_morphism_frames_as_nodes(self):
        """Morphisms connect frames as source and target nodes."""
        result = identify_discourse_adverbs("Therefore, we conclude.")
        morphism = map_to_inter_frame_relation(result[0])

        # Source and target are frame references (placeholders until Phase 4)
        assert hasattr(morphism, "source_frame")
        assert hasattr(morphism, "target_frame")

    def test_morphism_adverb_as_edge(self):
        """The discourse adverb acts as the edge label."""
        result = identify_discourse_adverbs("Moreover, it works.")
        morphism = map_to_inter_frame_relation(result[0])

        assert morphism.edge_label == "moreover"
        assert morphism.relation == DiscourseRelation.ADDITION


class TestRelationTypes:
    """Test all required relation types exist and are distinguishable."""

    def test_all_relation_types_exist(self):
        """All six required relation types are defined."""
        required = {
            "CONTRAST", "CONSEQUENCE", "ADDITION",
            "CONCESSION", "SEQUENCE", "EXEMPLIFICATION"
        }
        defined = {r.name for r in DiscourseRelation}
        assert required.issubset(defined)

    def test_contrast_distinguishable_from_concession(self):
        """CONTRAST (but/however) vs CONCESSION (although) are distinct."""
        contrast = identify_discourse_adverbs("However, it failed.")
        concession = identify_discourse_adverbs("Although expensive, it works.")

        assert contrast[0].relation == DiscourseRelation.CONTRAST
        assert concession[0].relation == DiscourseRelation.CONCESSION
        assert contrast[0].relation != concession[0].relation


class TestMultipleDiscourseAdverbs:
    """Test handling of multiple discourse markers."""

    def test_multiple_adverbs_in_text(self):
        """Multiple discourse adverbs are all identified."""
        text = "First, we plan. Then, we execute. Finally, we review."
        result = identify_discourse_adverbs(text)

        assert len(result) >= 2  # At least 'then' and 'finally'
        relations = {r.relation for r in result}
        assert DiscourseRelation.SEQUENCE in relations

    def test_adverb_order_preserved(self):
        """Discourse adverbs returned in text order."""
        text = "However, this failed. Therefore, we changed approach."
        result = identify_discourse_adverbs(text)

        assert len(result) == 2
        assert result[0].relation == DiscourseRelation.CONTRAST
        assert result[1].relation == DiscourseRelation.CONSEQUENCE
        assert result[0].start_char < result[1].start_char


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        result = identify_discourse_adverbs("")
        assert result == []

    def test_no_discourse_adverbs(self):
        """Text without discourse adverbs returns empty."""
        result = identify_discourse_adverbs("The cat sat on the mat.")
        assert result == []

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        lower = identify_discourse_adverbs("however, we disagree.")
        upper = identify_discourse_adverbs("HOWEVER, we disagree.")
        mixed = identify_discourse_adverbs("However, we disagree.")

        assert len(lower) == len(upper) == len(mixed) == 1
        assert lower[0].relation == upper[0].relation == mixed[0].relation

    def test_position_recorded(self):
        """Character positions are recorded correctly."""
        result = identify_discourse_adverbs("Well, however, it works.")
        however_result = [r for r in result if r.text.lower() == "however"]

        assert len(however_result) == 1
        assert however_result[0].start_char > 0

    def test_multi_word_adverbs(self):
        """Multi-word discourse markers are recognized."""
        result = identify_discourse_adverbs("On the other hand, this approach works.")
        assert len(result) == 1
        assert result[0].text.lower() == "on the other hand"
        assert result[0].relation == DiscourseRelation.CONTRAST


class TestMorphismProperties:
    """Test properties of generated morphisms."""

    def test_morphism_is_directional(self):
        """Inter-frame morphisms have direction (source → target)."""
        result = identify_discourse_adverbs("Therefore, X follows.")
        morphism = map_to_inter_frame_relation(result[0])

        # Direction is implicit: previous frame → following frame
        assert morphism.source_frame is not None or morphism.source_frame == "PREVIOUS"

    def test_morphism_relation_strength(self):
        """Morphisms have relation strength/confidence."""
        result = identify_discourse_adverbs("Consequently, this happens.")
        morphism = map_to_inter_frame_relation(result[0])

        assert hasattr(morphism, "strength")
        assert 0.0 <= morphism.strength <= 1.0
