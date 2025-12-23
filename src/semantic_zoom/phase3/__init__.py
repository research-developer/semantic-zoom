"""Phase 3: Morphism Mapping.

Maps linguistic elements to categorical morphisms:
- Prepositions → categorical symbols with state (NSM-43)
- Focusing adverbs → scope operators (NSM-44)
- Discourse adverbs → inter-frame relations (NSM-45)
"""
from semantic_zoom.phase3.preposition_symbols import (
    CategoricalSymbol,
    PrepositionMapping,
    SymbolState,
    map_preposition,
)
from semantic_zoom.phase3.focusing_adverbs import (
    FocusingAdverb,
    ScopeBinding,
    ScopeOperator,
    get_scope_bindings,
    identify_focusing_adverbs,
)
from semantic_zoom.phase3.discourse_adverbs import (
    DiscourseAdverb,
    DiscourseRelation,
    InterFrameMorphism,
    identify_discourse_adverbs,
    map_to_inter_frame_relation,
)
from semantic_zoom.phase3.integration import (
    MorphismToken,
    Phase3Processor,
    Phase3Result,
    process_tokens_phase3,
)

__all__ = [
    # NSM-43: Preposition symbols
    "CategoricalSymbol",
    "PrepositionMapping",
    "SymbolState",
    "map_preposition",
    # NSM-44: Focusing adverbs
    "FocusingAdverb",
    "ScopeBinding",
    "ScopeOperator",
    "get_scope_bindings",
    "identify_focusing_adverbs",
    # NSM-45: Discourse adverbs
    "DiscourseAdverb",
    "DiscourseRelation",
    "InterFrameMorphism",
    "identify_discourse_adverbs",
    "map_to_inter_frame_relation",
    # Integration (Phase 2 → Phase 3)
    "MorphismToken",
    "Phase3Processor",
    "Phase3Result",
    "process_tokens_phase3",
]
