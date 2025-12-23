"""Tests for verb tense/aspect classification (NSM-40 / GitHub #6).

Acceptance Criteria:
- Verbs tagged: PAST, PRESENT, FUTURE, or INFINITIVE
- Compound tenses identified from auxiliaries
- Aspect captured: SIMPLE, PROGRESSIVE, PERFECT, PERFECT_PROGRESSIVE
"""

import pytest

from semantic_zoom.models import Aspect, Tense, Token, VerbCompound
from semantic_zoom.phase2.verb_tense import (
    classify_tense,
    classify_aspect,
    analyze_verb_compound,
)


def make_token(
    text: str,
    tag: str,
    pos: str = "VERB",
    lemma: str | None = None,
    dep: str = "ROOT",
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
        head_idx=0,
    )


class TestSimpleTense:
    """Test simple (non-compound) tense classification from verb tag."""

    def test_past_tense_vbd(self) -> None:
        # "walked", "ate", "went"
        token = make_token("walked", tag="VBD", lemma="walk")
        assert classify_tense(token) == Tense.PAST

    def test_present_tense_vbz(self) -> None:
        # "walks", "eats", "goes" (3rd person singular)
        token = make_token("walks", tag="VBZ", lemma="walk")
        assert classify_tense(token) == Tense.PRESENT

    def test_present_tense_vbp(self) -> None:
        # "walk", "eat", "go" (non-3rd person singular)
        token = make_token("walk", tag="VBP", lemma="walk")
        assert classify_tense(token) == Tense.PRESENT

    def test_infinitive_vb(self) -> None:
        # Base form "to walk", "to eat"
        token = make_token("walk", tag="VB", lemma="walk")
        assert classify_tense(token) == Tense.INFINITIVE

    def test_gerund_vbg_no_context(self) -> None:
        # "walking" without context - defaults to present progressive
        token = make_token("walking", tag="VBG", lemma="walk")
        assert classify_tense(token) == Tense.PRESENT

    def test_participle_vbn_no_context(self) -> None:
        # "walked" (past participle) without aux - ambiguous, defaults to past
        token = make_token("walked", tag="VBN", lemma="walk")
        assert classify_tense(token) == Tense.PAST


class TestSimpleAspect:
    """Test simple aspect classification."""

    def test_simple_past(self) -> None:
        token = make_token("walked", tag="VBD", lemma="walk")
        assert classify_aspect(token) == Aspect.SIMPLE

    def test_simple_present(self) -> None:
        token = make_token("walks", tag="VBZ", lemma="walk")
        assert classify_aspect(token) == Aspect.SIMPLE

    def test_infinitive_is_simple(self) -> None:
        token = make_token("walk", tag="VB", lemma="walk")
        assert classify_aspect(token) == Aspect.SIMPLE


class TestCompoundTense:
    """Test compound verb forms with auxiliaries."""

    def test_future_will(self) -> None:
        # "will walk"
        aux = make_token("will", tag="MD", pos="AUX", lemma="will", idx=0)
        main = make_token("walk", tag="VB", lemma="walk", idx=1)
        tokens = [aux, main]
        main.head_idx = 0

        compound = analyze_verb_compound(tokens, main_verb_idx=1)

        assert compound.tense == Tense.FUTURE
        assert compound.aspect == Aspect.SIMPLE

    def test_present_progressive(self) -> None:
        # "is walking"
        aux = make_token("is", tag="VBZ", pos="AUX", lemma="be", idx=0)
        main = make_token("walking", tag="VBG", lemma="walk", idx=1)
        tokens = [aux, main]
        main.head_idx = 0

        compound = analyze_verb_compound(tokens, main_verb_idx=1)

        assert compound.tense == Tense.PRESENT
        assert compound.aspect == Aspect.PROGRESSIVE

    def test_past_progressive(self) -> None:
        # "was walking"
        aux = make_token("was", tag="VBD", pos="AUX", lemma="be", idx=0)
        main = make_token("walking", tag="VBG", lemma="walk", idx=1)
        tokens = [aux, main]
        main.head_idx = 0

        compound = analyze_verb_compound(tokens, main_verb_idx=1)

        assert compound.tense == Tense.PAST
        assert compound.aspect == Aspect.PROGRESSIVE

    def test_present_perfect(self) -> None:
        # "has walked"
        aux = make_token("has", tag="VBZ", pos="AUX", lemma="have", idx=0)
        main = make_token("walked", tag="VBN", lemma="walk", idx=1)
        tokens = [aux, main]
        main.head_idx = 0

        compound = analyze_verb_compound(tokens, main_verb_idx=1)

        assert compound.tense == Tense.PRESENT
        assert compound.aspect == Aspect.PERFECT

    def test_past_perfect(self) -> None:
        # "had walked"
        aux = make_token("had", tag="VBD", pos="AUX", lemma="have", idx=0)
        main = make_token("walked", tag="VBN", lemma="walk", idx=1)
        tokens = [aux, main]
        main.head_idx = 0

        compound = analyze_verb_compound(tokens, main_verb_idx=1)

        assert compound.tense == Tense.PAST
        assert compound.aspect == Aspect.PERFECT

    def test_future_progressive(self) -> None:
        # "will be walking"
        will = make_token("will", tag="MD", pos="AUX", lemma="will", idx=0)
        be = make_token("be", tag="VB", pos="AUX", lemma="be", idx=1)
        main = make_token("walking", tag="VBG", lemma="walk", idx=2)
        tokens = [will, be, main]
        be.head_idx = 0
        main.head_idx = 1

        compound = analyze_verb_compound(tokens, main_verb_idx=2)

        assert compound.tense == Tense.FUTURE
        assert compound.aspect == Aspect.PROGRESSIVE

    def test_present_perfect_progressive(self) -> None:
        # "has been walking"
        has = make_token("has", tag="VBZ", pos="AUX", lemma="have", idx=0)
        been = make_token("been", tag="VBN", pos="AUX", lemma="be", idx=1)
        main = make_token("walking", tag="VBG", lemma="walk", idx=2)
        tokens = [has, been, main]
        been.head_idx = 0
        main.head_idx = 1

        compound = analyze_verb_compound(tokens, main_verb_idx=2)

        assert compound.tense == Tense.PRESENT
        assert compound.aspect == Aspect.PERFECT_PROGRESSIVE

    def test_past_perfect_progressive(self) -> None:
        # "had been walking"
        had = make_token("had", tag="VBD", pos="AUX", lemma="have", idx=0)
        been = make_token("been", tag="VBN", pos="AUX", lemma="be", idx=1)
        main = make_token("walking", tag="VBG", lemma="walk", idx=2)
        tokens = [had, been, main]
        been.head_idx = 0
        main.head_idx = 1

        compound = analyze_verb_compound(tokens, main_verb_idx=2)

        assert compound.tense == Tense.PAST
        assert compound.aspect == Aspect.PERFECT_PROGRESSIVE

    def test_future_perfect(self) -> None:
        # "will have walked"
        will = make_token("will", tag="MD", pos="AUX", lemma="will", idx=0)
        have = make_token("have", tag="VB", pos="AUX", lemma="have", idx=1)
        main = make_token("walked", tag="VBN", lemma="walk", idx=2)
        tokens = [will, have, main]
        have.head_idx = 0
        main.head_idx = 1

        compound = analyze_verb_compound(tokens, main_verb_idx=2)

        assert compound.tense == Tense.FUTURE
        assert compound.aspect == Aspect.PERFECT


class TestCompoundTracking:
    """Test that compound verb analysis tracks auxiliary indices."""

    def test_compound_tracks_auxiliaries(self) -> None:
        # "has been walking"
        has = make_token("has", tag="VBZ", pos="AUX", lemma="have", idx=0)
        been = make_token("been", tag="VBN", pos="AUX", lemma="be", idx=1)
        main = make_token("walking", tag="VBG", lemma="walk", idx=2)
        tokens = [has, been, main]
        been.head_idx = 0
        main.head_idx = 1

        compound = analyze_verb_compound(tokens, main_verb_idx=2)

        assert compound.main_verb_idx == 2
        assert 0 in compound.auxiliary_indices
        assert 1 in compound.auxiliary_indices


class TestEdgeCases:
    """Test edge cases."""

    def test_non_verb_returns_none(self) -> None:
        token = make_token("quickly", tag="RB", pos="ADV")
        assert classify_tense(token) is None

    def test_modal_auxiliary_alone(self) -> None:
        # "can", "could", "should" are auxiliaries
        token = make_token("can", tag="MD", pos="AUX", lemma="can")
        # Modal alone doesn't have tense, but indicates future/hypothetical
        assert classify_tense(token) is None
