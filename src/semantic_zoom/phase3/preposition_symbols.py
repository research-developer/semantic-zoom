"""Preposition → categorical symbol mapping with state (NSM-43).

Maps ~60 English prepositions to ~15-20 categorical symbols with state flags.
Handles dual-citizenship prepositions via saturation mechanism.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CategoricalSymbol(Enum):
    """Categorical symbols for preposition morphisms.

    Symbol families:
    - DIRECTIONAL (◃▹): motion toward/away
    - CONTAINMENT (∈∉): inside/outside relationship
    - SPATIAL (⊤⊥): vertical positioning
    - ACCOMPANIMENT (⊕⊖): with/without
    - TEMPORAL (◁▷◇): time relationships
    - PURPOSE/BENEFICIARY/AGENT: semantic role markers
    - GENERIC: fallback for unmapped prepositions
    """
    # Directional symbols
    DIRECTIONAL_TO = "◃"
    DIRECTIONAL_FROM = "▹"

    # Containment symbols
    CONTAINMENT_IN = "∈"
    CONTAINMENT_OUT = "∉"

    # Spatial symbols (vertical axis)
    SPATIAL_ON = "⊤"
    SPATIAL_UNDER = "⊥"
    SPATIAL_AT = "⊛"  # Locative point
    SPATIAL_PROXIMITY = "≈"  # Near/by

    # Accompaniment symbols
    ACCOMPANIMENT_WITH = "⊕"
    ACCOMPANIMENT_WITHOUT = "⊖"

    # Temporal symbols
    TEMPORAL_BEFORE = "◁"
    TEMPORAL_AFTER = "▷"
    TEMPORAL_DURING = "◇"
    TEMPORAL_UNTIL = "⊣"
    TEMPORAL_AT = "⊙"  # Point in time

    # Semantic role symbols
    AGENT_BY = "λ"  # Agent marker (by the author)
    PURPOSE_FOR = "→"  # Purpose (for the goal)
    BENEFICIARY_FOR = "⇒"  # Beneficiary (for the person)
    MANNER_LIKE = "∼"  # Manner (like X)
    INSTRUMENT_WITH = "⊗"  # Instrument (with a hammer)

    # Identity morphism (categorical requirement)
    IDENTITY = "ε"  # Reflexive/no-change relation (X is X, as X)

    # Generic fallback
    GENERIC = "•"


@dataclass(frozen=True)
class SymbolState:
    """State flags for preposition symbols.

    Attributes:
        polarity: 1 for positive (with, in), -1 for negative (without, out)
        motion: 'static' or 'dynamic' indicating movement
        inverse: The inverse preposition if reversible (to↔from)
    """
    polarity: int = 1
    motion: str = "static"
    inverse: Optional[str] = None


@dataclass
class PrepositionMapping:
    """Result of mapping a preposition to categorical symbol(s).

    Attributes:
        original: The original preposition text
        symbol: The primary categorical symbol (may be None if dual-citizen unsaturated)
        possible_symbols: All possible symbols for dual-citizen prepositions
        state: State flags (polarity, motion, inverse)
        saturated: Whether dual-citizenship has been resolved by context
        is_dual_citizen: Whether this preposition has multiple category memberships
    """
    original: str
    symbol: CategoricalSymbol
    state: SymbolState
    possible_symbols: list[CategoricalSymbol] = field(default_factory=list)
    saturated: bool = True
    is_dual_citizen: bool = False

    def saturate(self, resolved_symbol: CategoricalSymbol) -> "PrepositionMapping":
        """Resolve dual-citizenship to a specific symbol via context.

        Args:
            resolved_symbol: The symbol determined by context analysis

        Returns:
            New PrepositionMapping with saturated=True and resolved symbol
        """
        if resolved_symbol not in self.possible_symbols:
            raise ValueError(
                f"Cannot saturate to {resolved_symbol}; "
                f"valid options are {self.possible_symbols}"
            )
        return PrepositionMapping(
            original=self.original,
            symbol=resolved_symbol,
            state=self.state,
            possible_symbols=self.possible_symbols,
            saturated=True,
            is_dual_citizen=self.is_dual_citizen,
        )


# Preposition mapping tables
_DIRECTIONAL_TO: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "to": (CategoricalSymbol.DIRECTIONAL_TO, SymbolState(motion="dynamic", inverse="from")),
    "toward": (CategoricalSymbol.DIRECTIONAL_TO, SymbolState(motion="dynamic")),
    "towards": (CategoricalSymbol.DIRECTIONAL_TO, SymbolState(motion="dynamic")),
    "into": (CategoricalSymbol.CONTAINMENT_IN, SymbolState(motion="dynamic")),
}

_DIRECTIONAL_FROM: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "from": (CategoricalSymbol.DIRECTIONAL_FROM, SymbolState(motion="dynamic", inverse="to")),
    "away": (CategoricalSymbol.DIRECTIONAL_FROM, SymbolState(motion="dynamic")),
    "away from": (CategoricalSymbol.DIRECTIONAL_FROM, SymbolState(motion="dynamic")),
}

_CONTAINMENT_IN: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "in": (CategoricalSymbol.CONTAINMENT_IN, SymbolState(motion="static")),
    "inside": (CategoricalSymbol.CONTAINMENT_IN, SymbolState(motion="static")),
    "within": (CategoricalSymbol.CONTAINMENT_IN, SymbolState(motion="static")),
}

_CONTAINMENT_OUT: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "out": (CategoricalSymbol.CONTAINMENT_OUT, SymbolState(polarity=-1, motion="dynamic")),
    "outside": (CategoricalSymbol.CONTAINMENT_OUT, SymbolState(polarity=-1, motion="static")),
    "out of": (CategoricalSymbol.CONTAINMENT_OUT, SymbolState(polarity=-1, motion="dynamic")),
}

_SPATIAL_ON: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "on": (CategoricalSymbol.SPATIAL_ON, SymbolState(motion="static")),
    "upon": (CategoricalSymbol.SPATIAL_ON, SymbolState(motion="static")),
    "atop": (CategoricalSymbol.SPATIAL_ON, SymbolState(motion="static")),
    "above": (CategoricalSymbol.SPATIAL_ON, SymbolState(motion="static")),
    "over": (CategoricalSymbol.SPATIAL_ON, SymbolState(motion="static")),
    "onto": (CategoricalSymbol.SPATIAL_ON, SymbolState(motion="dynamic")),
}

_SPATIAL_UNDER: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "under": (CategoricalSymbol.SPATIAL_UNDER, SymbolState(motion="static", inverse="over")),
    "below": (CategoricalSymbol.SPATIAL_UNDER, SymbolState(motion="static", inverse="above")),
    "beneath": (CategoricalSymbol.SPATIAL_UNDER, SymbolState(motion="static")),
    "underneath": (CategoricalSymbol.SPATIAL_UNDER, SymbolState(motion="static")),
}

_ACCOMPANIMENT: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "with": (CategoricalSymbol.ACCOMPANIMENT_WITH, SymbolState(polarity=1, inverse="without")),
    "along with": (CategoricalSymbol.ACCOMPANIMENT_WITH, SymbolState(polarity=1)),
    "together with": (CategoricalSymbol.ACCOMPANIMENT_WITH, SymbolState(polarity=1)),
    "without": (CategoricalSymbol.ACCOMPANIMENT_WITHOUT, SymbolState(polarity=-1, inverse="with")),
}

_TEMPORAL: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "before": (CategoricalSymbol.TEMPORAL_BEFORE, SymbolState(inverse="after")),
    "prior to": (CategoricalSymbol.TEMPORAL_BEFORE, SymbolState()),
    "after": (CategoricalSymbol.TEMPORAL_AFTER, SymbolState(inverse="before")),
    "following": (CategoricalSymbol.TEMPORAL_AFTER, SymbolState()),
    "during": (CategoricalSymbol.TEMPORAL_DURING, SymbolState()),
    "throughout": (CategoricalSymbol.TEMPORAL_DURING, SymbolState()),
    "until": (CategoricalSymbol.TEMPORAL_UNTIL, SymbolState()),
    "till": (CategoricalSymbol.TEMPORAL_UNTIL, SymbolState()),
    "since": (CategoricalSymbol.TEMPORAL_AFTER, SymbolState()),
}

_IDENTITY: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    "as": (CategoricalSymbol.IDENTITY, SymbolState()),  # X as X, X qua X
    "qua": (CategoricalSymbol.IDENTITY, SymbolState()),  # Formal identity
}

# Dual-citizenship prepositions (require context for resolution)
_DUAL_CITIZENS: dict[str, tuple[list[CategoricalSymbol], SymbolState]] = {
    "at": (
        [CategoricalSymbol.SPATIAL_AT, CategoricalSymbol.TEMPORAL_AT],
        SymbolState(motion="static"),
    ),
    "by": (
        [CategoricalSymbol.SPATIAL_PROXIMITY, CategoricalSymbol.AGENT_BY],
        SymbolState(),
    ),
    "for": (
        [CategoricalSymbol.PURPOSE_FOR, CategoricalSymbol.BENEFICIARY_FOR],
        SymbolState(),
    ),
}

# Combine all single-citizenship mappings
_SIMPLE_MAPPINGS: dict[str, tuple[CategoricalSymbol, SymbolState]] = {
    **_DIRECTIONAL_TO,
    **_DIRECTIONAL_FROM,
    **_CONTAINMENT_IN,
    **_CONTAINMENT_OUT,
    **_SPATIAL_ON,
    **_SPATIAL_UNDER,
    **_ACCOMPANIMENT,
    **_TEMPORAL,
    **_IDENTITY,
}


def map_preposition(preposition: str) -> PrepositionMapping:
    """Map a preposition to its categorical symbol(s) with state.

    Args:
        preposition: The preposition text (case-insensitive)

    Returns:
        PrepositionMapping containing symbol(s) and state flags

    Examples:
        >>> result = map_preposition("into")
        >>> result.symbol
        CategoricalSymbol.CONTAINMENT_IN
        >>> result.state.motion
        'dynamic'

        >>> result = map_preposition("at")
        >>> result.is_dual_citizen
        True
        >>> result.saturated
        False
    """
    prep_lower = preposition.lower().strip()

    # Check dual-citizenship prepositions first
    if prep_lower in _DUAL_CITIZENS:
        possible_symbols, state = _DUAL_CITIZENS[prep_lower]
        return PrepositionMapping(
            original=preposition,
            symbol=possible_symbols[0],  # Primary, but unsaturated
            state=state,
            possible_symbols=possible_symbols,
            saturated=False,
            is_dual_citizen=True,
        )

    # Check simple mappings
    if prep_lower in _SIMPLE_MAPPINGS:
        symbol, state = _SIMPLE_MAPPINGS[prep_lower]
        return PrepositionMapping(
            original=preposition,
            symbol=symbol,
            state=state,
            possible_symbols=[symbol],
            saturated=True,
            is_dual_citizen=False,
        )

    # Unknown preposition - return GENERIC
    return PrepositionMapping(
        original=preposition,
        symbol=CategoricalSymbol.GENERIC,
        state=SymbolState(),
        possible_symbols=[CategoricalSymbol.GENERIC],
        saturated=True,
        is_dual_citizen=False,
    )
