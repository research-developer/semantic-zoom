"""Tests for the Phase 1 → Phase 2 integration pipeline."""

import pytest
from semantic_zoom.pipeline import Pipeline, adapt_phase1_to_phase2
from semantic_zoom.models import Token, Person, Tense, Aspect, AdjectiveSlot, AdverbTier


class TestPipelineIntegration:
    """Test full pipeline from text to classified tokens."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    def test_empty_input(self, pipeline):
        """Empty text returns empty list."""
        assert pipeline.process("") == []
        assert pipeline.process("   ") == []

    def test_basic_tokenization(self, pipeline):
        """Tokens have correct basic fields."""
        tokens = pipeline.process("The cat sat.")

        assert len(tokens) == 4  # The, cat, sat, .
        assert tokens[0].text == "The"
        assert tokens[1].text == "cat"
        assert tokens[2].text == "sat"
        assert tokens[3].text == "."

    def test_lemma_extraction(self, pipeline):
        """Lemmas are correctly extracted."""
        tokens = pipeline.process("The cats were sitting quickly.")

        # Find relevant tokens
        cats = next(t for t in tokens if t.text == "cats")
        were = next(t for t in tokens if t.text == "were")
        sitting = next(t for t in tokens if t.text == "sitting")
        quickly = next(t for t in tokens if t.text == "quickly")

        assert cats.lemma == "cat"
        assert were.lemma == "be"
        assert sitting.lemma == "sit"
        assert quickly.lemma == "quickly"

    def test_pos_tags(self, pipeline):
        """POS tags are correctly assigned."""
        tokens = pipeline.process("The big cat runs quickly.")

        the = next(t for t in tokens if t.text == "The")
        big = next(t for t in tokens if t.text == "big")
        cat = next(t for t in tokens if t.text == "cat")
        runs = next(t for t in tokens if t.text == "runs")
        quickly = next(t for t in tokens if t.text == "quickly")

        assert the.pos == "DET"
        assert big.pos == "ADJ"
        assert cat.pos == "NOUN"
        assert runs.pos == "VERB"
        assert quickly.pos == "ADV"

    def test_dependency_parsing(self, pipeline):
        """Dependency relations are correct."""
        tokens = pipeline.process("The cat sat.")

        cat = next(t for t in tokens if t.text == "cat")
        sat = next(t for t in tokens if t.text == "sat")

        # sat is ROOT
        assert sat.dep == "ROOT"
        # cat is subject of sat
        assert cat.dep == "nsubj"
        assert cat.head_idx == sat.idx

    def test_idx_assignment(self, pipeline):
        """Token indices are sequential starting from 0."""
        tokens = pipeline.process("One two three.")

        for i, token in enumerate(tokens):
            assert token.idx == i


class TestPhase2NounPerson:
    """Test noun person classification in pipeline."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    def test_first_person_pronouns(self, pipeline):
        """First person pronouns are classified."""
        tokens = pipeline.process("I saw her.")

        i_token = next(t for t in tokens if t.text == "I")
        assert i_token.person == Person.FIRST

    def test_second_person_pronouns(self, pipeline):
        """Second person pronouns are classified."""
        tokens = pipeline.process("You are great.")

        you_token = next(t for t in tokens if t.text == "You")
        assert you_token.person == Person.SECOND

    def test_third_person_pronouns(self, pipeline):
        """Third person pronouns are classified."""
        tokens = pipeline.process("She likes him.")

        she = next(t for t in tokens if t.text == "She")
        him = next(t for t in tokens if t.text == "him")

        assert she.person == Person.THIRD
        assert him.person == Person.THIRD


class TestPhase2VerbTense:
    """Test verb tense/aspect classification in pipeline."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    def test_past_tense(self, pipeline):
        """Past tense verbs are classified."""
        tokens = pipeline.process("She walked home.")

        walked = next(t for t in tokens if t.text == "walked")
        assert walked.tense == Tense.PAST

    def test_present_tense(self, pipeline):
        """Present tense verbs are classified."""
        tokens = pipeline.process("She walks home.")

        walks = next(t for t in tokens if t.text == "walks")
        assert walks.tense == Tense.PRESENT


class TestPhase2Adjectives:
    """Test adjective classification in pipeline."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    def test_adjective_slot_assignment(self, pipeline):
        """Adjectives get slot assignments."""
        tokens = pipeline.process("The big red ball.")

        big = next(t for t in tokens if t.text == "big")
        red = next(t for t in tokens if t.text == "red")

        assert big.adj_slot == AdjectiveSlot.SIZE
        assert red.adj_slot == AdjectiveSlot.COLOR


class TestPhase2Adverbs:
    """Test adverb tier classification in pipeline."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    def test_manner_adverb(self, pipeline):
        """Manner adverbs are classified."""
        tokens = pipeline.process("She ran quickly.")

        quickly = next(t for t in tokens if t.text == "quickly")
        assert quickly.adv_tier == AdverbTier.MANNER

    def test_time_adverb(self, pipeline):
        """Time adverbs are classified."""
        # Note: "yesterday" is often tagged as NOUN by spaCy
        # Using "soon" which is reliably tagged as ADV
        tokens = pipeline.process("She will arrive soon.")

        soon = next(t for t in tokens if t.text == "soon")
        assert soon.adv_tier == AdverbTier.TIME

    def test_degree_adverb(self, pipeline):
        """Degree adverbs are classified with attachment."""
        tokens = pipeline.process("She ran very quickly.")

        very = next(t for t in tokens if t.text == "very")
        assert very.adv_tier == AdverbTier.DEGREE


class TestAdapterFunction:
    """Test standalone adapter function."""

    def test_adapt_parsed_tokens(self):
        """Adapter function converts Phase 1 to Phase 2 format."""
        from semantic_zoom.phase1.tokenizer import Tokenizer
        from semantic_zoom.phase1.dependency_parser import DependencyParser

        tokenizer = Tokenizer()
        parser = DependencyParser()

        text = "The cat sat."
        phase1_tokens = tokenizer.tokenize(text)
        parsed = parser.parse(phase1_tokens)

        # Adapt to Phase 2
        tokens = adapt_phase1_to_phase2(parsed, text)

        assert len(tokens) == 4
        assert all(isinstance(t, Token) for t in tokens)
        assert tokens[1].lemma == "cat"  # Lemma extraction works
        assert tokens[1].idx == 1  # idx mapped correctly


class TestPhase1Only:
    """Test Phase 1 only processing."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    def test_phase1_only(self, pipeline):
        """Can run Phase 1 only."""
        parsed = pipeline.process_phase1_only("The cat sat.")

        assert len(parsed) == 4
        assert parsed[0].text == "The"
        assert parsed[2].dep == "ROOT"


class TestConvertParsedTokens:
    """Test converting existing Phase 1 output."""

    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    def test_convert_without_text(self, pipeline):
        """Can convert without providing original text."""
        parsed = pipeline.process_phase1_only("The cat sat.")
        tokens = pipeline.convert_parsed_tokens(parsed)

        assert len(tokens) == 4
        assert tokens[1].lemma == "cat"

    def test_convert_with_text(self, pipeline):
        """Can convert with original text provided."""
        text = "The cat sat."
        parsed = pipeline.process_phase1_only(text)
        tokens = pipeline.convert_parsed_tokens(parsed, text)

        assert len(tokens) == 4
        assert tokens[1].lemma == "cat"
