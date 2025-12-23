"""Tests for adverb tier assignment (NSM-42 / GitHub #8).

Acceptance Criteria:
- Adverbs assigned: MANNER, PLACE, FREQUENCY, TIME, PURPOSE
- Sentence-level adverbs marked as SENTENCE tier
- Degree adverbs marked with attachment to modified word
- Canonical order: Manner > Place > Frequency > Time > Purpose
"""

import pytest

from semantic_zoom.models import AdverbTier, Token
from semantic_zoom.phase2.adverb_tier import (
    classify_adverb_tier,
    get_degree_attachment,
    classify_tokens_adverbs,
)


def make_token(
    text: str,
    pos: str = "ADV",
    tag: str = "RB",
    lemma: str | None = None,
    dep: str = "advmod",
    head_idx: int = 1,
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


class TestMannerAdverbs:
    """Test manner adverb classification."""

    @pytest.mark.parametrize(
        "word",
        ["quickly", "carefully", "slowly", "loudly", "quietly", "beautifully", "badly"],
    )
    def test_manner_adverbs(self, word: str) -> None:
        token = make_token(word)
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.MANNER, f"{word} should be MANNER"

    def test_manner_from_ly_suffix(self) -> None:
        # Most -ly adverbs are manner
        token = make_token("wonderfully")
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.MANNER


class TestPlaceAdverbs:
    """Test place adverb classification."""

    @pytest.mark.parametrize(
        "word",
        ["here", "there", "everywhere", "somewhere", "nowhere", "outside", "inside", "upstairs", "abroad"],
    )
    def test_place_adverbs(self, word: str) -> None:
        token = make_token(word)
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.PLACE, f"{word} should be PLACE"


class TestFrequencyAdverbs:
    """Test frequency adverb classification."""

    @pytest.mark.parametrize(
        "word",
        ["always", "never", "often", "sometimes", "rarely", "usually", "frequently", "occasionally"],
    )
    def test_frequency_adverbs(self, word: str) -> None:
        token = make_token(word)
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.FREQUENCY, f"{word} should be FREQUENCY"


class TestTimeAdverbs:
    """Test time adverb classification."""

    @pytest.mark.parametrize(
        "word",
        ["now", "then", "yesterday", "today", "tomorrow", "soon", "already", "still", "lately", "recently"],
    )
    def test_time_adverbs(self, word: str) -> None:
        token = make_token(word)
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.TIME, f"{word} should be TIME"


class TestPurposeAdverbs:
    """Test purpose/reason adverb classification."""

    @pytest.mark.parametrize(
        "word",
        ["therefore", "consequently", "hence", "thus", "accordingly"],
    )
    def test_purpose_adverbs(self, word: str) -> None:
        token = make_token(word)
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.PURPOSE, f"{word} should be PURPOSE"


class TestSentenceAdverbs:
    """Test sentence-level adverb classification."""

    @pytest.mark.parametrize(
        "word",
        ["frankly", "honestly", "unfortunately", "fortunately", "surprisingly", "obviously", "clearly"],
    )
    def test_sentence_adverbs(self, word: str) -> None:
        # Sentence adverbs typically modify the whole clause
        token = make_token(word, dep="advmod")
        # When head is ROOT or sentence-initial, it's a sentence adverb
        token.head_idx = 0  # Points to root
        tier = classify_adverb_tier(token, is_sentence_initial=True)
        assert tier == AdverbTier.SENTENCE, f"{word} should be SENTENCE"


class TestDegreeAdverbs:
    """Test degree adverb classification and attachment."""

    @pytest.mark.parametrize(
        "word",
        ["very", "extremely", "quite", "rather", "too", "fairly", "highly", "incredibly"],
    )
    def test_degree_adverbs(self, word: str) -> None:
        token = make_token(word)
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.DEGREE, f"{word} should be DEGREE"

    def test_degree_attachment_to_adjective(self) -> None:
        # "very big" - "very" attaches to "big"
        very = make_token("very", dep="advmod", head_idx=1, idx=0)
        big = make_token("big", pos="ADJ", tag="JJ", dep="amod", head_idx=2, idx=1)
        noun = make_token("house", pos="NOUN", tag="NN", dep="ROOT", head_idx=2, idx=2)
        tokens = [very, big, noun]

        attachment = get_degree_attachment(very, tokens)
        assert attachment == 1  # Attached to "big" at index 1

    def test_degree_attachment_to_adverb(self) -> None:
        # "very quickly" - "very" attaches to "quickly"
        very = make_token("very", dep="advmod", head_idx=1, idx=0)
        quickly = make_token("quickly", dep="advmod", head_idx=2, idx=1)
        verb = make_token("runs", pos="VERB", tag="VBZ", dep="ROOT", head_idx=2, idx=2)
        tokens = [very, quickly, verb]

        attachment = get_degree_attachment(very, tokens)
        assert attachment == 1  # Attached to "quickly" at index 1


class TestAdverbClassification:
    """Test full adverb classification on token lists."""

    def test_classify_updates_tokens(self) -> None:
        quickly = make_token("quickly", dep="advmod", head_idx=1, idx=0)
        ran = make_token("ran", pos="VERB", tag="VBD", dep="ROOT", head_idx=1, idx=1)
        tokens = [quickly, ran]

        classify_tokens_adverbs(tokens)

        assert tokens[0].adv_tier == AdverbTier.MANNER

    def test_degree_adverb_sets_attachment(self) -> None:
        very = make_token("very", dep="advmod", head_idx=1, idx=0)
        quickly = make_token("quickly", dep="advmod", head_idx=2, idx=1)
        ran = make_token("ran", pos="VERB", tag="VBD", dep="ROOT", head_idx=2, idx=2)
        tokens = [very, quickly, ran]

        classify_tokens_adverbs(tokens)

        assert tokens[0].adv_tier == AdverbTier.DEGREE
        assert tokens[0].adv_attachment == 1  # Attached to "quickly"


class TestEdgeCases:
    """Test edge cases."""

    def test_non_adverb_returns_none(self) -> None:
        token = make_token("dog", pos="NOUN", tag="NN")
        tier = classify_adverb_tier(token)
        assert tier is None

    def test_ambiguous_adverb_defaults_to_manner(self) -> None:
        # Unknown adverbs ending in -ly default to manner
        token = make_token("flibbertigibbetly")
        tier = classify_adverb_tier(token)
        assert tier == AdverbTier.MANNER
