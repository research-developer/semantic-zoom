"""Tests for noun/pronoun person classification (NSM-39 / GitHub #5).

Acceptance Criteria:
- Nouns/pronouns tagged: FIRST, SECOND, THIRD, or NONE
- Generic constructions marked with generic=True flag
"""

import pytest

from semantic_zoom.models import Person, Token
from semantic_zoom.phase2.noun_person import classify_person, is_generic_construction


def make_token(
    text: str,
    pos: str = "NOUN",
    tag: str = "NN",
    lemma: str | None = None,
    dep: str = "nsubj",
) -> Token:
    """Helper to create test tokens."""
    return Token(
        text=text,
        lemma=lemma or text.lower(),
        idx=0,
        pos=pos,
        tag=tag,
        dep=dep,
        head_idx=1,
    )


class TestFirstPerson:
    """Test first person pronoun classification."""

    @pytest.mark.parametrize(
        "text,tag",
        [
            ("I", "PRP"),
            ("me", "PRP"),
            ("myself", "PRP"),
            ("we", "PRP"),
            ("us", "PRP"),
            ("ourselves", "PRP"),
        ],
    )
    def test_first_person_pronouns(self, text: str, tag: str) -> None:
        token = make_token(text, pos="PRON", tag=tag)
        result = classify_person(token)
        assert result == Person.FIRST

    def test_possessive_my(self) -> None:
        token = make_token("my", pos="PRON", tag="PRP$", lemma="my")
        result = classify_person(token)
        assert result == Person.FIRST

    def test_possessive_our(self) -> None:
        token = make_token("our", pos="PRON", tag="PRP$", lemma="our")
        result = classify_person(token)
        assert result == Person.FIRST


class TestSecondPerson:
    """Test second person pronoun classification."""

    @pytest.mark.parametrize(
        "text",
        ["you", "yourself", "yourselves"],
    )
    def test_second_person_pronouns(self, text: str) -> None:
        token = make_token(text, pos="PRON", tag="PRP")
        result = classify_person(token)
        assert result == Person.SECOND

    def test_possessive_your(self) -> None:
        token = make_token("your", pos="PRON", tag="PRP$", lemma="your")
        result = classify_person(token)
        assert result == Person.SECOND


class TestThirdPerson:
    """Test third person classification."""

    @pytest.mark.parametrize(
        "text",
        ["he", "she", "it", "him", "her", "they", "them", "himself", "herself", "itself", "themselves"],
    )
    def test_third_person_pronouns(self, text: str) -> None:
        token = make_token(text, pos="PRON", tag="PRP")
        result = classify_person(token)
        assert result == Person.THIRD

    def test_proper_noun_is_third(self) -> None:
        token = make_token("John", pos="PROPN", tag="NNP")
        result = classify_person(token)
        assert result == Person.THIRD

    def test_possessive_their(self) -> None:
        token = make_token("their", pos="PRON", tag="PRP$", lemma="their")
        result = classify_person(token)
        assert result == Person.THIRD

    def test_possessive_his_her_its(self) -> None:
        for text in ["his", "her", "its"]:
            token = make_token(text, pos="PRON", tag="PRP$", lemma=text)
            result = classify_person(token)
            assert result == Person.THIRD


class TestNoPersonMarking:
    """Test nouns without person marking."""

    def test_common_noun_is_none(self) -> None:
        token = make_token("dog", pos="NOUN", tag="NN")
        result = classify_person(token)
        assert result == Person.NONE

    def test_abstract_noun_is_none(self) -> None:
        token = make_token("freedom", pos="NOUN", tag="NN")
        result = classify_person(token)
        assert result == Person.NONE


class TestGenericConstruction:
    """Test generic construction marking."""

    def test_generic_one(self) -> None:
        # "One must be careful"
        token = make_token("one", pos="PRON", tag="PRP", lemma="one")
        assert is_generic_construction(token) is True
        result = classify_person(token)
        assert result == Person.THIRD  # "one" is third person but generic

    def test_generic_people(self) -> None:
        # "People say..."
        token = make_token("people", pos="NOUN", tag="NNS", lemma="people", dep="nsubj")
        assert is_generic_construction(token) is True

    def test_non_generic_specific_person(self) -> None:
        # "The people in the room"
        token = make_token("people", pos="NOUN", tag="NNS", lemma="people", dep="pobj")
        assert is_generic_construction(token) is False


class TestEdgeCases:
    """Test edge cases and special forms."""

    def test_who_relative_pronoun(self) -> None:
        token = make_token("who", pos="PRON", tag="WP")
        result = classify_person(token)
        assert result == Person.THIRD

    def test_non_pronoun_returns_none(self) -> None:
        token = make_token("quickly", pos="ADV", tag="RB")
        result = classify_person(token)
        assert result == Person.NONE
