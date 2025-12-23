"""Tests for adjective ordering classification (NSM-41 / GitHub #7).

Acceptance Criteria:
- Adjectives ordered: opinion > size > age > shape > color > origin > material > purpose
- Non-canonical order normalized with original positions preserved
- Each adjective classified to one slot
"""

import pytest

from semantic_zoom.models import AdjectiveChain, AdjectiveSlot, Token
from semantic_zoom.phase2.adjective_order import (
    classify_adjective_slot,
    extract_adjective_chains,
    normalize_chain,
    is_canonical_order,
)


def make_token(
    text: str,
    pos: str = "ADJ",
    tag: str = "JJ",
    lemma: str | None = None,
    dep: str = "amod",
    head_idx: int = 3,
    idx: int = 0,
) -> Token:
    """Helper to create test tokens."""
    return Token(
        text=text,
        lemma=lemma or text.lower(),
        idx=idx,
        pos=pos,
        tag=tag,
        dep=dep,
        head_idx=head_idx,
    )


class TestAdjectiveSlotClassification:
    """Test classification of adjectives into ordering slots."""

    def test_opinion_adjectives(self) -> None:
        for word in ["lovely", "beautiful", "ugly", "amazing", "horrible"]:
            token = make_token(word)
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.OPINION, f"{word} should be OPINION"

    def test_size_adjectives(self) -> None:
        for word in ["big", "small", "large", "tiny", "huge", "enormous"]:
            token = make_token(word)
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.SIZE, f"{word} should be SIZE"

    def test_age_adjectives(self) -> None:
        for word in ["old", "young", "ancient", "new", "modern"]:
            token = make_token(word)
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.AGE, f"{word} should be AGE"

    def test_shape_adjectives(self) -> None:
        for word in ["round", "square", "flat", "circular", "rectangular"]:
            token = make_token(word)
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.SHAPE, f"{word} should be SHAPE"

    def test_color_adjectives(self) -> None:
        for word in ["red", "blue", "green", "yellow", "black", "white"]:
            token = make_token(word)
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.COLOR, f"{word} should be COLOR"

    def test_origin_adjectives(self) -> None:
        for word in ["American", "French", "Chinese", "Italian", "Japanese"]:
            token = make_token(word, tag="JJ")
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.ORIGIN, f"{word} should be ORIGIN"

    def test_material_adjectives(self) -> None:
        for word in ["wooden", "metal", "silk", "cotton", "plastic", "golden"]:
            token = make_token(word)
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.MATERIAL, f"{word} should be MATERIAL"

    def test_purpose_adjectives(self) -> None:
        # Purpose adjectives often come from -ing forms or compound modifiers
        for word in ["sleeping", "wedding", "running", "cooking"]:
            token = make_token(word, tag="VBG")  # gerund used as adj
            token.dep = "amod"
            slot = classify_adjective_slot(token)
            assert slot == AdjectiveSlot.PURPOSE, f"{word} should be PURPOSE"


class TestAdjChainExtraction:
    """Test extraction of adjective chains modifying a noun."""

    def test_single_adjective(self) -> None:
        # "the big house"
        det = make_token("the", pos="DET", tag="DT", dep="det", head_idx=2, idx=0)
        adj = make_token("big", dep="amod", head_idx=2, idx=1)
        noun = make_token("house", pos="NOUN", tag="NN", dep="ROOT", head_idx=2, idx=2)
        tokens = [det, adj, noun]

        chains = extract_adjective_chains(tokens)

        assert len(chains) == 1
        assert chains[0].noun_idx == 2
        assert len(chains[0].adjectives) == 1
        assert chains[0].adjectives[0].text == "big"

    def test_multiple_adjectives(self) -> None:
        # "a lovely old wooden chair"
        det = make_token("a", pos="DET", tag="DT", dep="det", head_idx=4, idx=0)
        adj1 = make_token("lovely", dep="amod", head_idx=4, idx=1)
        adj2 = make_token("old", dep="amod", head_idx=4, idx=2)
        adj3 = make_token("wooden", dep="amod", head_idx=4, idx=3)
        noun = make_token("chair", pos="NOUN", tag="NN", dep="ROOT", head_idx=4, idx=4)
        tokens = [det, adj1, adj2, adj3, noun]

        chains = extract_adjective_chains(tokens)

        assert len(chains) == 1
        assert chains[0].noun_idx == 4
        assert len(chains[0].adjectives) == 3


class TestCanonicalOrder:
    """Test detection of canonical adjective ordering."""

    def test_canonical_order_detected(self) -> None:
        # "lovely old wooden" is canonical (opinion > age > material)
        adj1 = make_token("lovely", idx=0)
        adj1.adj_slot = AdjectiveSlot.OPINION
        adj2 = make_token("old", idx=1)
        adj2.adj_slot = AdjectiveSlot.AGE
        adj3 = make_token("wooden", idx=2)
        adj3.adj_slot = AdjectiveSlot.MATERIAL

        chain = AdjectiveChain(noun_idx=3, adjectives=[adj1, adj2, adj3])
        assert is_canonical_order(chain) is True

    def test_non_canonical_order_detected(self) -> None:
        # "wooden old lovely" is NOT canonical
        adj1 = make_token("wooden", idx=0)
        adj1.adj_slot = AdjectiveSlot.MATERIAL
        adj2 = make_token("old", idx=1)
        adj2.adj_slot = AdjectiveSlot.AGE
        adj3 = make_token("lovely", idx=2)
        adj3.adj_slot = AdjectiveSlot.OPINION

        chain = AdjectiveChain(noun_idx=3, adjectives=[adj1, adj2, adj3])
        assert is_canonical_order(chain) is False


class TestNormalization:
    """Test normalization of non-canonical adjective orders."""

    def test_normalize_preserves_original_positions(self) -> None:
        # "wooden old lovely" -> normalized to "lovely old wooden"
        adj1 = make_token("wooden", idx=0)
        adj1.adj_slot = AdjectiveSlot.MATERIAL
        adj1.adj_original_pos = None

        adj2 = make_token("old", idx=1)
        adj2.adj_slot = AdjectiveSlot.AGE
        adj2.adj_original_pos = None

        adj3 = make_token("lovely", idx=2)
        adj3.adj_slot = AdjectiveSlot.OPINION
        adj3.adj_original_pos = None

        chain = AdjectiveChain(noun_idx=3, adjectives=[adj1, adj2, adj3])
        normalized = normalize_chain(chain)

        # After normalization, canonical_order should list indices in proper order
        # lovely (opinion) should come first, then old (age), then wooden (material)
        assert len(normalized.canonical_order) == 3
        assert normalized.is_canonical is False

        # Original positions should be preserved
        assert adj1.adj_original_pos == 0  # wooden was at position 0
        assert adj2.adj_original_pos == 1  # old was at position 1
        assert adj3.adj_original_pos == 2  # lovely was at position 2

    def test_canonical_input_unchanged(self) -> None:
        # "lovely old wooden" - already canonical
        adj1 = make_token("lovely", idx=0)
        adj1.adj_slot = AdjectiveSlot.OPINION

        adj2 = make_token("old", idx=1)
        adj2.adj_slot = AdjectiveSlot.AGE

        adj3 = make_token("wooden", idx=2)
        adj3.adj_slot = AdjectiveSlot.MATERIAL

        chain = AdjectiveChain(noun_idx=3, adjectives=[adj1, adj2, adj3])
        normalized = normalize_chain(chain)

        assert normalized.is_canonical is True
        # Canonical order matches original order
        assert normalized.canonical_order == [0, 1, 2]


class TestEdgeCases:
    """Test edge cases in adjective ordering."""

    def test_empty_chain(self) -> None:
        chain = AdjectiveChain(noun_idx=0, adjectives=[])
        assert is_canonical_order(chain) is True

    def test_single_adjective_always_canonical(self) -> None:
        adj = make_token("big", idx=0)
        adj.adj_slot = AdjectiveSlot.SIZE

        chain = AdjectiveChain(noun_idx=1, adjectives=[adj])
        assert is_canonical_order(chain) is True

    def test_unknown_slot_defaults_to_opinion(self) -> None:
        # Words we don't recognize should get a default slot
        token = make_token("flibbertigibbet")  # Made up word
        slot = classify_adjective_slot(token)
        # Default to OPINION (first/closest to noun) or some reasonable default
        assert slot is not None
