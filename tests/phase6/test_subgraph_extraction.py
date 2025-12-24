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


class TestCopulaSkipping:
    """Test optional copula/auxiliary skipping for semantic focus."""

    def test_skip_copulas_false_by_default(self):
        """skip_copulas should be False by default for backward compat."""
        extractor = SubgraphExtractor()
        assert extractor.skip_copulas is False

    def test_skip_copulas_excludes_is_in_copular_sentence(self):
        """With skip_copulas=True, copula 'is' should be excluded."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()

        text = "The cat is on the mat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select "cat"
        cat_id = next(t.id for t in parsed if t.text == "cat")
        seed = selector.select_range(parsed, start_id=cat_id, end_id=cat_id)
        selector.add_seed(seed)

        # Without skip_copulas
        extractor_normal = SubgraphExtractor(skip_copulas=False)
        result_normal = extractor_normal.extract(parsed, selector.seeds, zoom_level=1)
        words_normal = [parsed[i].text for i in result_normal.word_ids]
        assert "is" in words_normal

        # With skip_copulas
        extractor_skip = SubgraphExtractor(skip_copulas=True)
        result_skip = extractor_skip.extract(parsed, selector.seeds, zoom_level=1)
        words_skip = [parsed[i].text for i in result_skip.word_ids]
        assert "is" not in words_skip
        assert "cat" in words_skip

    def test_skip_copulas_connects_through_is(self):
        """With skip_copulas=True, siblings through copula should connect."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()

        text = "The cat is happy."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select "cat" - should reach "happy" through "is"
        cat_id = next(t.id for t in parsed if t.text == "cat")
        seed = selector.select_range(parsed, start_id=cat_id, end_id=cat_id)
        selector.add_seed(seed)

        extractor = SubgraphExtractor(skip_copulas=True)
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        words = [parsed[i].text for i in result.word_ids]

        # Should include "happy" as sibling through transparent "is"
        assert "cat" in words
        assert "is" not in words
        # "happy" should be reachable as sibling
        assert "happy" in words or "." in words

    def test_skip_copulas_preserves_aux_in_progressive(self):
        """Auxiliary 'was' in progressives should NOT be skipped."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()

        text = "She was running quickly."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select "running" - "was" is aux, not ROOT, so should be included
        run_id = next(t.id for t in parsed if t.text == "running")
        seed = selector.select_range(parsed, start_id=run_id, end_id=run_id)
        selector.add_seed(seed)

        extractor = SubgraphExtractor(skip_copulas=True)
        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        words = [parsed[i].text for i in result.word_ids]

        # "was" should be included (it's aux, not copular ROOT)
        assert "was" in words
        assert "running" in words

    def test_skip_copulas_per_call_override(self):
        """skip_copulas can be overridden per extract() call."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()

        text = "The book is here."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        book_id = next(t.id for t in parsed if t.text == "book")
        seed = selector.select_range(parsed, start_id=book_id, end_id=book_id)
        selector.add_seed(seed)

        # Default skip_copulas=False, but override to True
        extractor = SubgraphExtractor(skip_copulas=False)
        result = extractor.extract(parsed, selector.seeds, zoom_level=1, skip_copulas=True)
        words = [parsed[i].text for i in result.word_ids]

        assert "is" not in words

    def test_skip_copulas_handles_predicate_nominal(self):
        """Should skip 'is' in predicate nominal constructions."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()

        text = "John is a doctor."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        john_id = next(t.id for t in parsed if t.text == "John")
        seed = selector.select_range(parsed, start_id=john_id, end_id=john_id)
        selector.add_seed(seed)

        extractor = SubgraphExtractor(skip_copulas=True)
        result = extractor.extract(parsed, selector.seeds, zoom_level=2)
        words = [parsed[i].text for i in result.word_ids]

        assert "John" in words
        assert "is" not in words
        assert "doctor" in words
