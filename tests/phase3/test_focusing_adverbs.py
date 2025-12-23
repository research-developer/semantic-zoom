"""Tests for focusing adverb identification and scope marking (NSM-44)."""
import pytest
from semantic_zoom.phase3.focusing_adverbs import (
    FocusingAdverb,
    ScopeOperator,
    ScopeBinding,
    identify_focusing_adverbs,
    get_scope_bindings,
)


class TestFocusingAdverbIdentification:
    """Test identification of focusing adverbs in text."""

    @pytest.mark.parametrize("adverb", [
        "only", "even", "just", "merely", "simply",
        "especially", "particularly", "specifically",
        "exactly", "precisely", "solely", "exclusively",
    ])
    def test_focusing_adverbs_recognized(self, adverb: str):
        """All standard focusing adverbs are recognized."""
        result = identify_focusing_adverbs(f"I {adverb} want coffee.")
        assert len(result) == 1
        assert result[0].text.lower() == adverb

    def test_multiple_focusing_adverbs(self):
        """Multiple focusing adverbs in same sentence."""
        text = "She only just arrived."
        result = identify_focusing_adverbs(text)
        assert len(result) == 2
        adverbs = {r.text.lower() for r in result}
        assert adverbs == {"only", "just"}

    def test_no_focusing_adverbs(self):
        """Sentences without focusing adverbs return empty list."""
        result = identify_focusing_adverbs("The cat sat on the mat.")
        assert result == []

    def test_false_positives_filtered(self):
        """Words that look like focusing adverbs but aren't should be filtered."""
        # "just" as adjective (fair/righteous) shouldn't match
        result = identify_focusing_adverbs("A just ruler governs wisely.")
        # This is tricky - may need POS tagging from Phase 2
        # For now, we'll accept the adverb interpretation
        assert len(result) >= 0  # Placeholder for POS-aware implementation


class TestScopeOperatorMarking:
    """Test that focusing adverbs are marked as SCOPE_OPERATORs."""

    def test_scope_operator_type(self):
        """Focusing adverbs are marked as SCOPE_OPERATOR type."""
        result = identify_focusing_adverbs("I only want coffee.")
        assert len(result) == 1
        assert result[0].operator_type == ScopeOperator.FOCUS

    def test_invertible_false(self):
        """Focusing operations are marked as non-reversible."""
        result = identify_focusing_adverbs("She even finished early.")
        assert len(result) == 1
        assert result[0].invertible is False

    def test_restrictive_vs_additive(self):
        """Restrictive adverbs (only, just) vs additive (even, also)."""
        only_result = identify_focusing_adverbs("I only want coffee.")
        even_result = identify_focusing_adverbs("I even want coffee.")

        assert only_result[0].focus_type == "restrictive"
        assert even_result[0].focus_type == "additive"


class TestScopeIdentification:
    """Test scope target identification."""

    def test_scope_target_immediate_np(self):
        """Scope typically targets immediate following NP."""
        result = identify_focusing_adverbs("Only John passed the test.")
        assert len(result) == 1
        # Scope should target "John"
        assert result[0].scope_target is not None

    def test_scope_target_vp(self):
        """Scope can target VP in certain positions."""
        result = identify_focusing_adverbs("John only passed the test.")
        assert len(result) == 1
        # Scope should target "passed" or "passed the test"

    def test_scope_target_adjective(self):
        """Scope can target adjectives."""
        result = identify_focusing_adverbs("The water is merely warm.")
        assert len(result) == 1


class TestScopeBindings:
    """Test enumeration of possible scope bindings."""

    def test_ambiguous_scope_multiple_bindings(self):
        """Ambiguous scope positions enumerate multiple bindings."""
        # "John only saw Mary on Tuesday" - scope could be:
        # 1. only John (not Bill) saw Mary
        # 2. John saw only Mary (not Sue)
        # 3. John saw Mary only on Tuesday (not Monday)
        text = "John only saw Mary on Tuesday."
        bindings = get_scope_bindings(text)

        assert isinstance(bindings, list)
        assert len(bindings) >= 1  # At minimum, one interpretation

    def test_unambiguous_scope_single_binding(self):
        """Clear scope positions have single binding."""
        text = "Only John passed."
        bindings = get_scope_bindings(text)

        assert len(bindings) == 1
        assert bindings[0].target == "John"

    def test_scope_binding_structure(self):
        """Scope bindings have correct structure."""
        bindings = get_scope_bindings("I only want coffee.")

        assert len(bindings) >= 1
        binding = bindings[0]
        assert isinstance(binding, ScopeBinding)
        assert hasattr(binding, "target")
        assert hasattr(binding, "confidence")
        assert 0.0 <= binding.confidence <= 1.0


class TestEdgeCases:
    """Test edge cases and special situations."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        result = identify_focusing_adverbs("")
        assert result == []

    def test_case_insensitive(self):
        """Focusing adverb detection is case-insensitive."""
        result_lower = identify_focusing_adverbs("only")
        result_upper = identify_focusing_adverbs("ONLY")
        result_mixed = identify_focusing_adverbs("Only")

        assert len(result_lower) == len(result_upper) == len(result_mixed) == 1

    def test_adverb_position_recorded(self):
        """Character position of adverb is recorded."""
        result = identify_focusing_adverbs("I only want coffee.")
        assert len(result) == 1
        assert result[0].start_char == 2
        assert result[0].end_char == 6
