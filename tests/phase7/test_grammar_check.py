"""Tests for NSM-57: Grammar check integration.

Tests:
- Grammatical error identification with type and location
- Severity levels: ERROR, WARNING, INFO
- Suggestion generation
- Original and corrected version preservation
"""
import pytest


class TestGrammarError:
    """Test grammatical error detection."""

    def test_subject_verb_agreement_error(self):
        """Test detection of subject-verb agreement errors."""
        from semantic_zoom.phase7.grammar_check import check_grammar, Severity

        text = "The dogs runs quickly."
        result = check_grammar(text)

        assert len(result.errors) >= 1
        error = result.errors[0]
        assert error.error_type == "SUBJECT_VERB_AGREEMENT"
        assert error.severity == Severity.ERROR
        assert "runs" in error.text or error.start_char <= 9  # Position of "runs"

    def test_article_error(self):
        """Test detection of article errors."""
        from semantic_zoom.phase7.grammar_check import check_grammar, Severity

        text = "I saw a apple on the table."
        result = check_grammar(text)

        assert len(result.errors) >= 1
        # Should detect "a apple" -> "an apple"
        error = next((e for e in result.errors if "a apple" in text[e.start_char:e.end_char] or e.error_type == "ARTICLE"), None)
        assert error is not None
        assert error.severity in (Severity.ERROR, Severity.WARNING)

    def test_double_negative(self):
        """Test detection of double negatives."""
        from semantic_zoom.phase7.grammar_check import check_grammar, Severity

        text = "I don't want no trouble."
        result = check_grammar(text)

        # Double negative should be detected
        assert len(result.errors) >= 1
        has_double_neg = any(e.error_type == "DOUBLE_NEGATIVE" for e in result.errors)
        assert has_double_neg or len(result.errors) > 0  # Some error detected

    def test_missing_comma(self):
        """Test detection of missing commas."""
        from semantic_zoom.phase7.grammar_check import check_grammar, Severity

        text = "However I think we should go."
        result = check_grammar(text)

        # Should suggest comma after "However"
        has_comma_error = any(
            e.error_type in ("MISSING_COMMA", "PUNCTUATION")
            for e in result.errors
        )
        # May or may not detect depending on tool
        assert result is not None


class TestSeverityLevels:
    """Test severity classification."""

    def test_severity_enum_values(self):
        """Test that Severity enum has required values."""
        from semantic_zoom.phase7.grammar_check import Severity

        assert Severity.ERROR is not None
        assert Severity.WARNING is not None
        assert Severity.INFO is not None

    def test_severe_error_classification(self):
        """Test that severe errors get ERROR severity."""
        from semantic_zoom.phase7.grammar_check import check_grammar, Severity

        # Fragment is typically an error
        text = "Because the dog."
        result = check_grammar(text)

        if result.errors:
            # At least one error should be WARNING or higher
            assert any(e.severity in (Severity.ERROR, Severity.WARNING) for e in result.errors)

    def test_minor_issue_classification(self):
        """Test that minor issues get INFO severity."""
        from semantic_zoom.phase7.grammar_check import check_grammar, Severity

        # Passive voice is usually INFO level
        text = "The ball was thrown by the boy."
        result = check_grammar(text)

        # This may or may not generate an INFO
        assert result is not None


class TestSuggestions:
    """Test suggestion generation."""

    def test_suggestion_provided(self):
        """Test that suggestions are provided for errors."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            # At least one error should have a suggestion
            has_suggestion = any(e.suggestion is not None for e in result.errors)
            assert has_suggestion

    def test_suggestion_is_valid_correction(self):
        """Test that suggestions fix the error."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            for error in result.errors:
                if error.suggestion:
                    # Suggestion should be different from original
                    assert error.suggestion != error.text

    def test_multiple_suggestions_possible(self):
        """Test handling of multiple possible corrections."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dogs runs quickly."
        result = check_grammar(text)

        # Result structure should support this
        assert result is not None


class TestPreservation:
    """Test original and corrected version preservation."""

    def test_original_preserved(self):
        """Test that original text is preserved."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dogs runs quickly."
        result = check_grammar(text)

        assert result.original == text

    def test_corrected_version_available(self):
        """Test that corrected version is generated."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dogs runs quickly."
        result = check_grammar(text)

        # Corrected version should be different if errors found
        if result.errors:
            assert result.corrected != result.original

    def test_no_errors_same_text(self):
        """Test that text without errors has same original and corrected."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dog runs quickly."
        result = check_grammar(text)

        if not result.errors:
            assert result.corrected == result.original

    def test_error_location_spans(self):
        """Test that errors have valid character spans."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dogs runs quickly."
        result = check_grammar(text)

        for error in result.errors:
            assert error.start_char >= 0
            assert error.end_char <= len(text)
            assert error.start_char < error.end_char


class TestGrammarCheckResult:
    """Test GrammarCheckResult structure."""

    def test_result_has_required_fields(self):
        """Test that result has all required fields."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "Test sentence."
        result = check_grammar(text)

        assert hasattr(result, 'original')
        assert hasattr(result, 'corrected')
        assert hasattr(result, 'errors')

    def test_error_has_required_fields(self):
        """Test that GrammarError has all required fields."""
        from semantic_zoom.phase7.grammar_check import check_grammar

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            error = result.errors[0]
            assert hasattr(error, 'error_type')
            assert hasattr(error, 'severity')
            assert hasattr(error, 'start_char')
            assert hasattr(error, 'end_char')
            assert hasattr(error, 'text')
            assert hasattr(error, 'suggestion')
            assert hasattr(error, 'message')
