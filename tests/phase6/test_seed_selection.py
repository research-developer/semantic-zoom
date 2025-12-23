"""Tests for seed statement selection (NSM-53)."""

import pytest
from semantic_zoom.phase1.tokenizer import Tokenizer
from semantic_zoom.phase1.dependency_parser import DependencyParser
from semantic_zoom.phase6.seed_selection import SeedSelector, Seed


class TestSeedSelection:
    """Test that text selection captures word ID range as seed."""

    def test_select_word_range(self):
        """Text selection should capture word ID range."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select "brown fox" (IDs 2, 3)
        seed = selector.select_range(parsed, start_id=2, end_id=3)
        
        assert seed is not None
        assert seed.start_id == 2
        assert seed.end_id == 3
        assert "brown" in seed.text
        assert "fox" in seed.text

    def test_select_by_char_offset(self):
        """Selection by character offset should map to word IDs."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "The quick brown fox jumps."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select characters for "quick brown" (chars 4-15)
        seed = selector.select_by_chars(parsed, start_char=4, end_char=15)
        
        assert seed is not None
        assert "quick" in seed.text
        assert "brown" in seed.text

    def test_seed_has_word_ids(self):
        """Seed should contain list of word IDs."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "Hello world."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=0, end_id=1)
        
        assert hasattr(seed, 'word_ids')
        assert seed.word_ids == [0, 1]


class TestContainingClause:
    """Test that containing sentence/clause is identified."""

    def test_identify_sentence(self):
        """Seed should identify its containing sentence."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "First sentence. Second sentence here."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select word from second sentence
        seed = selector.select_range(parsed, start_id=3, end_id=3)  # "Second"
        
        assert seed.sentence_start_id is not None
        assert seed.sentence_end_id is not None
        # Should span the second sentence
        assert seed.sentence_start_id >= 3

    def test_identify_clause(self):
        """Seed should identify its containing clause."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "When the rain came, we stayed inside."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select "rain" from subordinate clause
        rain_id = next(t.id for t in parsed if t.text == "rain")
        seed = selector.select_range(parsed, start_id=rain_id, end_id=rain_id)
        
        assert seed.clause_root_id is not None

    def test_get_clause_tokens(self):
        """Should be able to get all tokens in the containing clause."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "The cat sat on the mat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=1, end_id=1)  # "cat"
        clause_ids = selector.get_clause_token_ids(parsed, seed)
        
        assert len(clause_ids) > 0
        assert 1 in clause_ids  # "cat" should be in its own clause


class TestMultiSeed:
    """Test that multiple seeds are storable with graph node highlighting."""

    def test_store_multiple_seeds(self):
        """Should be able to store multiple seeds."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "The cat chased the mouse. The dog watched."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Create two seeds
        seed1 = selector.select_range(parsed, start_id=1, end_id=1)  # "cat"
        seed2 = selector.select_range(parsed, start_id=4, end_id=4)  # "mouse"
        
        selector.add_seed(seed1)
        selector.add_seed(seed2)
        
        assert len(selector.seeds) == 2

    def test_seeds_have_unique_ids(self):
        """Each seed should have a unique identifier."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "Hello world today."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed1 = selector.select_range(parsed, start_id=0, end_id=0)
        seed2 = selector.select_range(parsed, start_id=1, end_id=1)
        
        selector.add_seed(seed1)
        selector.add_seed(seed2)
        
        ids = [s.id for s in selector.seeds]
        assert len(ids) == len(set(ids)), "Seed IDs should be unique"

    def test_get_all_highlighted_ids(self):
        """Should get union of all word IDs from all seeds."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "The quick fox jumps high."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed1 = selector.select_range(parsed, start_id=1, end_id=1)  # "quick"
        seed2 = selector.select_range(parsed, start_id=3, end_id=3)  # "jumps"
        
        selector.add_seed(seed1)
        selector.add_seed(seed2)
        
        highlighted = selector.get_all_seed_word_ids()
        
        assert 1 in highlighted
        assert 3 in highlighted

    def test_remove_seed(self):
        """Should be able to remove a seed."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "Hello world."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed1 = selector.select_range(parsed, start_id=0, end_id=0)
        seed2 = selector.select_range(parsed, start_id=1, end_id=1)
        
        selector.add_seed(seed1)
        selector.add_seed(seed2)
        
        selector.remove_seed(seed1.id)
        
        assert len(selector.seeds) == 1

    def test_clear_seeds(self):
        """Should be able to clear all seeds."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "Hello world."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed1 = selector.select_range(parsed, start_id=0, end_id=0)
        selector.add_seed(seed1)
        
        selector.clear_seeds()
        
        assert len(selector.seeds) == 0


class TestSeedAttributes:
    """Test Seed dataclass attributes."""

    def test_seed_has_required_fields(self):
        """Seed should have all required fields."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        
        text = "Hello world."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=0, end_id=1)
        
        assert hasattr(seed, 'id')
        assert hasattr(seed, 'start_id')
        assert hasattr(seed, 'end_id')
        assert hasattr(seed, 'word_ids')
        assert hasattr(seed, 'text')
        assert hasattr(seed, 'sentence_start_id')
        assert hasattr(seed, 'sentence_end_id')
        assert hasattr(seed, 'clause_root_id')
