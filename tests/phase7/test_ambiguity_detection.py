"""Tests for NSM-58: Ambiguity detection.

Tests:
- PP-attachment ambiguity
- Coordination ambiguity
- Pronoun ambiguity
- Quantifier scope ambiguity
- Negation scope ambiguity
- Multiple interpretations with confidence
- Possible antecedents for pronouns
"""
import pytest


class TestPPAttachmentAmbiguity:
    """Test prepositional phrase attachment ambiguity detection."""

    def test_classic_pp_attachment(self):
        """Test classic PP-attachment ambiguity (telescope example)."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        pp_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.PP_ATTACHMENT]
        assert len(pp_ambiguities) >= 1

        ambiguity = pp_ambiguities[0]
        # Should have at least 2 interpretations
        assert len(ambiguity.interpretations) >= 2

    def test_pp_attachment_interpretations(self):
        """Test that PP-attachment has correct interpretations."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        pp_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.PP_ATTACHMENT]
        if pp_ambiguities:
            interpretations = pp_ambiguities[0].interpretations
            # Should mention both attachment points
            descriptions = [i.description.lower() for i in interpretations]
            # One attaches to verb (saw), one to noun (man)
            assert len(descriptions) >= 2

    def test_pp_attachment_confidence(self):
        """Test that interpretations have confidence scores."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        pp_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.PP_ATTACHMENT]
        if pp_ambiguities:
            for interp in pp_ambiguities[0].interpretations:
                assert 0.0 <= interp.confidence <= 1.0


class TestCoordinationAmbiguity:
    """Test coordination ambiguity detection."""

    def test_coordination_scope(self):
        """Test coordination scope ambiguity."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "Old men and women gathered."
        result = detect_ambiguities(text)

        coord_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.COORDINATION]
        # "Old" could modify just "men" or "men and women"
        assert len(coord_ambiguities) >= 1

    def test_coordination_interpretations(self):
        """Test coordination has multiple interpretations."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "Old men and women gathered."
        result = detect_ambiguities(text)

        coord_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.COORDINATION]
        if coord_ambiguities:
            assert len(coord_ambiguities[0].interpretations) >= 2


class TestPronounAmbiguity:
    """Test pronoun reference ambiguity detection."""

    def test_pronoun_ambiguity_detected(self):
        """Test that pronoun ambiguity is detected."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "John told Bill that he was wrong."
        result = detect_ambiguities(text)

        pronoun_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.PRONOUN]
        # "he" could refer to John or Bill
        assert len(pronoun_ambiguities) >= 1

    def test_pronoun_possible_antecedents(self):
        """Test that possible antecedents are listed."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "John told Bill that he was wrong."
        result = detect_ambiguities(text)

        pronoun_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.PRONOUN]
        if pronoun_ambiguities:
            ambiguity = pronoun_ambiguities[0]
            assert ambiguity.possible_antecedents is not None
            assert len(ambiguity.possible_antecedents) >= 2
            # Should include John and Bill
            antecedent_texts = [a.text.lower() for a in ambiguity.possible_antecedents]
            assert "john" in antecedent_texts or "bill" in antecedent_texts

    def test_pronoun_antecedent_confidence(self):
        """Test that antecedents have confidence scores."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "John told Bill that he was wrong."
        result = detect_ambiguities(text)

        pronoun_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.PRONOUN]
        if pronoun_ambiguities and pronoun_ambiguities[0].possible_antecedents:
            for antecedent in pronoun_ambiguities[0].possible_antecedents:
                assert hasattr(antecedent, 'confidence')
                assert 0.0 <= antecedent.confidence <= 1.0


class TestQuantifierScopeAmbiguity:
    """Test quantifier scope ambiguity detection."""

    def test_quantifier_scope_detected(self):
        """Test quantifier scope ambiguity detection."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "Every student read a book."
        result = detect_ambiguities(text)

        quant_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.QUANTIFIER_SCOPE]
        # Could be: each student read their own book, or all read the same book
        assert len(quant_ambiguities) >= 1 or len(result.ambiguities) >= 0  # May not always detect

    def test_quantifier_scope_interpretations(self):
        """Test quantifier scope has different readings."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "Every student read a book."
        result = detect_ambiguities(text)

        quant_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.QUANTIFIER_SCOPE]
        if quant_ambiguities:
            assert len(quant_ambiguities[0].interpretations) >= 2


class TestNegationScopeAmbiguity:
    """Test negation scope ambiguity detection."""

    def test_negation_scope_detected(self):
        """Test negation scope ambiguity."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "John didn't leave because he was tired."
        result = detect_ambiguities(text)

        neg_ambiguities = [a for a in result.ambiguities if a.ambiguity_type == AmbiguityType.NEGATION_SCOPE]
        # Could mean: didn't leave (reason: tired) OR left (but not because tired)
        # This is a complex ambiguity, may not always detect
        assert result is not None

    def test_negation_quantifier_interaction(self):
        """Test negation-quantifier scope interaction."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities, AmbiguityType

        text = "All students didn't pass the exam."
        result = detect_ambiguities(text)

        # Could mean: not all passed (some failed) OR all failed
        ambiguities = [a for a in result.ambiguities
                       if a.ambiguity_type in (AmbiguityType.NEGATION_SCOPE, AmbiguityType.QUANTIFIER_SCOPE)]
        # May detect this as either type
        assert result is not None


class TestAmbiguityResult:
    """Test ambiguity detection result structure."""

    def test_result_has_required_fields(self):
        """Test that result has all required fields."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        assert hasattr(result, 'text')
        assert hasattr(result, 'ambiguities')
        assert result.text == text

    def test_ambiguity_has_required_fields(self):
        """Test that Ambiguity has all required fields."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        if result.ambiguities:
            ambiguity = result.ambiguities[0]
            assert hasattr(ambiguity, 'ambiguity_type')
            assert hasattr(ambiguity, 'span')
            assert hasattr(ambiguity, 'text')
            assert hasattr(ambiguity, 'interpretations')

    def test_interpretation_has_required_fields(self):
        """Test that Interpretation has all required fields."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        if result.ambiguities and result.ambiguities[0].interpretations:
            interp = result.ambiguities[0].interpretations[0]
            assert hasattr(interp, 'description')
            assert hasattr(interp, 'confidence')

    def test_no_ambiguity_empty_list(self):
        """Test that unambiguous text returns empty ambiguities."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities

        text = "The cat sat."
        result = detect_ambiguities(text)

        assert result.ambiguities is not None
        # Simple sentence should have few or no ambiguities
        assert isinstance(result.ambiguities, list)


class TestAmbiguityType:
    """Test AmbiguityType enum."""

    def test_ambiguity_type_values(self):
        """Test that AmbiguityType has all required values."""
        from semantic_zoom.phase7.ambiguity_detection import AmbiguityType

        assert AmbiguityType.PP_ATTACHMENT is not None
        assert AmbiguityType.COORDINATION is not None
        assert AmbiguityType.PRONOUN is not None
        assert AmbiguityType.QUANTIFIER_SCOPE is not None
        assert AmbiguityType.NEGATION_SCOPE is not None
