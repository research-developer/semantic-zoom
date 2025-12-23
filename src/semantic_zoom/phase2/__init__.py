"""Phase 2: Grammatical Classification.

This phase classifies tokens from Phase 1 into grammatical categories:
- NSM-39: Noun person classification (1st/2nd/3rd)
- NSM-40: Verb tense/aspect classification
- NSM-41: Adjective ordering and normalization
- NSM-42: Adverb tier assignment
"""

from semantic_zoom.phase2.noun_person import (
    classify_person,
    is_generic_construction,
    classify_token_person,
    classify_tokens_person,
)
from semantic_zoom.phase2.verb_tense import (
    classify_tense,
    classify_aspect,
    analyze_verb_compound,
    classify_token_tense,
)
from semantic_zoom.phase2.adjective_order import (
    classify_adjective_slot,
    extract_adjective_chains,
    is_canonical_order,
    normalize_chain,
    classify_tokens_adjectives,
)
from semantic_zoom.phase2.adverb_tier import (
    classify_adverb_tier,
    get_degree_attachment,
    classify_tokens_adverbs,
)

__all__ = [
    # Noun person (NSM-39)
    "classify_person",
    "is_generic_construction",
    "classify_token_person",
    "classify_tokens_person",
    # Verb tense (NSM-40)
    "classify_tense",
    "classify_aspect",
    "analyze_verb_compound",
    "classify_token_tense",
    # Adjective ordering (NSM-41)
    "classify_adjective_slot",
    "extract_adjective_chains",
    "is_canonical_order",
    "normalize_chain",
    "classify_tokens_adjectives",
    # Adverb tier (NSM-42)
    "classify_adverb_tier",
    "get_degree_attachment",
    "classify_tokens_adverbs",
]
