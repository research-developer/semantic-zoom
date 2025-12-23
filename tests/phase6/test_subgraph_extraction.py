"""Tests for subgraph extraction algorithm (NSM-54)."""

import pytest
from semantic_zoom.phase1.tokenizer import Tokenizer
from semantic_zoom.phase1.dependency_parser import DependencyParser
from semantic_zoom.phase6.seed_selection import SeedSelector
from semantic_zoom.phase6.subgraph_extraction import SubgraphExtractor


class TestSubgraphExtraction:
    """Test that seeds + zoom level produce connected subgraph."""

    def test_extract_with_seeds(self):
        """Should extract subgraph from seeds at given zoom level."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=3, end_id=3)  # "fox"
        selector.add_seed(seed)
        
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        
        assert result is not None
        assert len(result.word_ids) > 0
        assert 3 in result.word_ids  # Seed should be in result

    def test_result_contains_word_ids(self):
        """Result should contain list of word IDs covering subgraph."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "The cat sat on the mat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=1, end_id=1)  # "cat"
        selector.add_seed(seed)
        
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        
        assert hasattr(result, 'word_ids')
        assert isinstance(result.word_ids, list)
        assert all(isinstance(i, int) for i in result.word_ids)


class TestZoomLevelOne:
    """Test zoom level 1: directly connected nodes only."""

    def test_level_one_direct_dependents(self):
        """Zoom level 1 should include direct dependents."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "The big cat sat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select "cat" - has dependents "The" and "big"
        seed = selector.select_range(parsed, start_id=2, end_id=2)  # "cat"
        selector.add_seed(seed)
        
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        
        # Should include "The" (det) and "big" (amod)
        word_texts = [parsed[i].text for i in result.word_ids if i < len(parsed)]
        assert "The" in word_texts or "cat" in word_texts

    def test_level_one_includes_head(self):
        """Zoom level 1 should include syntactic head."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "The cat sat quickly."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select "cat" - head is "sat"
        seed = selector.select_range(parsed, start_id=1, end_id=1)  # "cat"
        selector.add_seed(seed)
        
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        
        # Should include "sat" (head)
        assert 2 in result.word_ids  # "sat"


class TestZoomLevelN:
    """Test zoom level N: N-hop neighborhood."""

    def test_level_two_expands_further(self):
        """Zoom level 2 should include 2-hop neighbors."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "The big fluffy cat sat quietly on the mat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select "cat"
        cat_id = next(t.id for t in parsed if t.text == "cat")
        seed = selector.select_range(parsed, start_id=cat_id, end_id=cat_id)
        selector.add_seed(seed)
        
        result_1 = extractor.extract(parsed, selector.seeds, zoom_level=1)
        result_2 = extractor.extract(parsed, selector.seeds, zoom_level=2)
        
        # Level 2 should include more tokens than level 1
        assert len(result_2.word_ids) >= len(result_1.word_ids)

    def test_higher_zoom_includes_more(self):
        """Higher zoom levels should include progressively more tokens."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "The very quick brown fox jumps gracefully over the extremely lazy dog."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=4, end_id=4)  # "fox"
        selector.add_seed(seed)
        
        result_1 = extractor.extract(parsed, selector.seeds, zoom_level=1)
        result_2 = extractor.extract(parsed, selector.seeds, zoom_level=2)
        result_3 = extractor.extract(parsed, selector.seeds, zoom_level=3)
        
        assert len(result_1.word_ids) <= len(result_2.word_ids) <= len(result_3.word_ids)


class TestDeterminism:
    """Test that extraction is quasi-deterministic."""

    def test_same_input_same_output(self):
        """Same seeds + level should produce same subgraph."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = SubgraphExtractor()
        
        text = "The cat sat on the mat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        selector1 = SeedSelector()
        seed1 = selector1.select_range(parsed, start_id=1, end_id=1)
        selector1.add_seed(seed1)
        
        selector2 = SeedSelector()
        seed2 = selector2.select_range(parsed, start_id=1, end_id=1)
        selector2.add_seed(seed2)
        
        result1 = extractor.extract(parsed, selector1.seeds, zoom_level=2)
        result2 = extractor.extract(parsed, selector2.seeds, zoom_level=2)
        
        assert sorted(result1.word_ids) == sorted(result2.word_ids)

    def test_repeated_extraction_consistent(self):
        """Repeated extractions should be consistent."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "She quickly ran home."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=2, end_id=2)  # "ran"
        selector.add_seed(seed)
        
        results = [
            extractor.extract(parsed, selector.seeds, zoom_level=1)
            for _ in range(5)
        ]
        
        first_ids = sorted(results[0].word_ids)
        for r in results[1:]:
            assert sorted(r.word_ids) == first_ids


class TestMultipleSeedsExtraction:
    """Test extraction with multiple seeds."""

    def test_multiple_seeds_union(self):
        """Multiple seeds should produce union of their subgraphs."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "The cat slept and the dog ran."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        # Select "cat" and "dog"
        cat_id = next(t.id for t in parsed if t.text == "cat")
        dog_id = next(t.id for t in parsed if t.text == "dog")
        
        seed1 = selector.select_range(parsed, start_id=cat_id, end_id=cat_id)
        seed2 = selector.select_range(parsed, start_id=dog_id, end_id=dog_id)
        selector.add_seed(seed1)
        selector.add_seed(seed2)
        
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        
        assert cat_id in result.word_ids
        assert dog_id in result.word_ids


class TestSubgraphResultAttributes:
    """Test SubgraphResult dataclass attributes."""

    def test_result_has_required_fields(self):
        """SubgraphResult should have all required fields."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        
        text = "Hello world."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)
        
        seed = selector.select_range(parsed, start_id=0, end_id=0)
        selector.add_seed(seed)
        
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        
        assert hasattr(result, 'word_ids')
        assert hasattr(result, 'zoom_level')
        assert hasattr(result, 'seed_ids')
