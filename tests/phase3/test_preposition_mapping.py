"""Tests for preposition → categorical symbol mapping (NSM-43)."""
import pytest
from semantic_zoom.phase3.preposition_symbols import (
    CategoricalSymbol,
    SymbolState,
    PrepositionMapping,
    map_preposition,
)


class TestCategoricalSymbols:
    """Test categorical symbol enum and types."""

    def test_directional_symbols_exist(self):
        """DIRECTIONAL symbols (◃▹) for to, from, toward, away."""
        assert CategoricalSymbol.DIRECTIONAL_TO.value == "◃"
        assert CategoricalSymbol.DIRECTIONAL_FROM.value == "▹"

    def test_containment_symbols_exist(self):
        """CONTAINMENT symbols (∈∉) for in, out, inside, within."""
        assert CategoricalSymbol.CONTAINMENT_IN.value == "∈"
        assert CategoricalSymbol.CONTAINMENT_OUT.value == "∉"

    def test_spatial_symbols_exist(self):
        """SPATIAL symbols (⊥) for on, under, above, below."""
        assert CategoricalSymbol.SPATIAL_ON.value == "⊤"
        assert CategoricalSymbol.SPATIAL_UNDER.value == "⊥"

    def test_accompaniment_symbols_exist(self):
        """ACCOMPANIMENT symbols (⊕⊖) for with, without."""
        assert CategoricalSymbol.ACCOMPANIMENT_WITH.value == "⊕"
        assert CategoricalSymbol.ACCOMPANIMENT_WITHOUT.value == "⊖"

    def test_temporal_symbols_exist(self):
        """TEMPORAL symbols (◁▷) for before, after, during, until."""
        assert CategoricalSymbol.TEMPORAL_BEFORE.value == "◁"
        assert CategoricalSymbol.TEMPORAL_AFTER.value == "▷"
        assert CategoricalSymbol.TEMPORAL_DURING.value == "◇"


class TestPrepositionMapping:
    """Test mapping of prepositions to categorical symbols."""

    @pytest.mark.parametrize("prep,expected_symbol", [
        ("to", CategoricalSymbol.DIRECTIONAL_TO),
        ("toward", CategoricalSymbol.DIRECTIONAL_TO),
        ("towards", CategoricalSymbol.DIRECTIONAL_TO),
        ("from", CategoricalSymbol.DIRECTIONAL_FROM),
        ("away", CategoricalSymbol.DIRECTIONAL_FROM),
    ])
    def test_directional_prepositions(self, prep, expected_symbol):
        """Directional prepositions map to DIRECTIONAL symbols."""
        result = map_preposition(prep)
        assert result.symbol == expected_symbol

    @pytest.mark.parametrize("prep,expected_symbol", [
        ("in", CategoricalSymbol.CONTAINMENT_IN),
        ("inside", CategoricalSymbol.CONTAINMENT_IN),
        ("within", CategoricalSymbol.CONTAINMENT_IN),
        ("into", CategoricalSymbol.CONTAINMENT_IN),
        ("out", CategoricalSymbol.CONTAINMENT_OUT),
        ("outside", CategoricalSymbol.CONTAINMENT_OUT),
        ("out of", CategoricalSymbol.CONTAINMENT_OUT),
    ])
    def test_containment_prepositions(self, prep, expected_symbol):
        """Containment prepositions map to CONTAINMENT symbols."""
        result = map_preposition(prep)
        assert result.symbol == expected_symbol

    @pytest.mark.parametrize("prep,expected_symbol", [
        ("on", CategoricalSymbol.SPATIAL_ON),
        ("upon", CategoricalSymbol.SPATIAL_ON),
        ("atop", CategoricalSymbol.SPATIAL_ON),
        ("above", CategoricalSymbol.SPATIAL_ON),
        ("over", CategoricalSymbol.SPATIAL_ON),
        ("under", CategoricalSymbol.SPATIAL_UNDER),
        ("below", CategoricalSymbol.SPATIAL_UNDER),
        ("beneath", CategoricalSymbol.SPATIAL_UNDER),
        ("underneath", CategoricalSymbol.SPATIAL_UNDER),
    ])
    def test_spatial_prepositions(self, prep, expected_symbol):
        """Spatial prepositions map to SPATIAL symbols."""
        result = map_preposition(prep)
        assert result.symbol == expected_symbol

    @pytest.mark.parametrize("prep,expected_symbol", [
        ("with", CategoricalSymbol.ACCOMPANIMENT_WITH),
        ("along with", CategoricalSymbol.ACCOMPANIMENT_WITH),
        ("together with", CategoricalSymbol.ACCOMPANIMENT_WITH),
        ("without", CategoricalSymbol.ACCOMPANIMENT_WITHOUT),
    ])
    def test_accompaniment_prepositions(self, prep, expected_symbol):
        """Accompaniment prepositions map to ACCOMPANIMENT symbols."""
        result = map_preposition(prep)
        assert result.symbol == expected_symbol

    @pytest.mark.parametrize("prep,expected_symbol", [
        ("before", CategoricalSymbol.TEMPORAL_BEFORE),
        ("prior to", CategoricalSymbol.TEMPORAL_BEFORE),
        ("after", CategoricalSymbol.TEMPORAL_AFTER),
        ("following", CategoricalSymbol.TEMPORAL_AFTER),
        ("during", CategoricalSymbol.TEMPORAL_DURING),
        ("throughout", CategoricalSymbol.TEMPORAL_DURING),
        ("until", CategoricalSymbol.TEMPORAL_UNTIL),
        ("till", CategoricalSymbol.TEMPORAL_UNTIL),
    ])
    def test_temporal_prepositions(self, prep, expected_symbol):
        """Temporal prepositions map to TEMPORAL symbols."""
        result = map_preposition(prep)
        assert result.symbol == expected_symbol


class TestDualCitizenship:
    """Test handling of prepositions with multiple category membership."""

    def test_at_is_dual_spatial_temporal(self):
        """'at' has dual citizenship: spatial (at the store) and temporal (at noon)."""
        result = map_preposition("at")
        assert result.is_dual_citizen
        assert CategoricalSymbol.SPATIAL_AT in result.possible_symbols
        assert CategoricalSymbol.TEMPORAL_AT in result.possible_symbols

    def test_by_is_dual_spatial_agent(self):
        """'by' has dual citizenship: spatial (by the door) and agent (by the author)."""
        result = map_preposition("by")
        assert result.is_dual_citizen
        assert CategoricalSymbol.SPATIAL_PROXIMITY in result.possible_symbols
        assert CategoricalSymbol.AGENT_BY in result.possible_symbols

    def test_for_is_dual_purpose_beneficiary(self):
        """'for' has dual citizenship: purpose and beneficiary."""
        result = map_preposition("for")
        assert result.is_dual_citizen
        assert CategoricalSymbol.PURPOSE_FOR in result.possible_symbols
        assert CategoricalSymbol.BENEFICIARY_FOR in result.possible_symbols

    def test_saturated_flag_resolves_dual_citizenship(self):
        """Saturated flag indicates context has resolved dual citizenship."""
        result = map_preposition("at")
        assert not result.saturated  # Initially unsaturated

        # When context resolves, saturated becomes True
        saturated_result = result.saturate(CategoricalSymbol.SPATIAL_AT)
        assert saturated_result.saturated
        assert saturated_result.symbol == CategoricalSymbol.SPATIAL_AT

    def test_cannot_saturate_non_dual_citizen(self):
        """Non-dual-citizen prepositions don't need saturation."""
        result = map_preposition("into")
        assert not result.is_dual_citizen
        # Should already be saturated
        assert result.saturated


class TestSymbolState:
    """Test state flags on preposition mappings."""

    def test_polarity_positive(self):
        """Positive polarity prepositions have state.polarity = 1."""
        result = map_preposition("with")
        assert result.state.polarity == 1

    def test_polarity_negative(self):
        """Negative polarity prepositions have state.polarity = -1."""
        result = map_preposition("without")
        assert result.state.polarity == -1

    def test_motion_dynamic(self):
        """Dynamic motion prepositions have state.motion = 'dynamic'."""
        result = map_preposition("into")
        assert result.state.motion == "dynamic"

    def test_motion_static(self):
        """Static prepositions have state.motion = 'static'."""
        result = map_preposition("in")
        assert result.state.motion == "static"

    def test_reversible_pair(self):
        """Reversible preposition pairs are marked."""
        to_result = map_preposition("to")
        from_result = map_preposition("from")
        assert to_result.state.inverse == "from"
        assert from_result.state.inverse == "to"


class TestUnknownPrepositions:
    """Test handling of unknown or rare prepositions."""

    def test_unknown_returns_generic(self):
        """Unknown prepositions return GENERIC symbol."""
        result = map_preposition("notwithstanding")
        assert result.symbol == CategoricalSymbol.GENERIC

    def test_case_insensitive(self):
        """Preposition mapping is case-insensitive."""
        result_lower = map_preposition("in")
        result_upper = map_preposition("IN")
        result_mixed = map_preposition("In")
        assert result_lower.symbol == result_upper.symbol == result_mixed.symbol
