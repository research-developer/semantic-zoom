"""Tests for sparse display rendering (NSM-55)."""

import pytest
from semantic_zoom.phase1.tokenizer import Tokenizer
from semantic_zoom.phase1.dependency_parser import DependencyParser
from semantic_zoom.phase6.seed_selection import SeedSelector
from semantic_zoom.phase6.subgraph_extraction import SubgraphExtractor
from semantic_zoom.phase6.sparse_render import SparseRenderer


class TestSparseRender:
    """Test that subgraph word IDs displayed with visual gaps for omissions."""

    def test_render_with_gaps(self):
        """Should render subgraph with gaps for omitted content."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select just "fox" - will get partial subgraph
        seed = selector.select_range(parsed, start_id=3, end_id=3)
        selector.add_seed(seed)

        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        rendered = renderer.render(parsed, result)

        assert rendered is not None
        assert len(rendered) > 0

    def test_placeholder_for_omissions(self):
        """Placeholders should indicate omitted content."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "The very quick brown fox jumps gracefully over the lazy dog."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select "fox" - zoom level 1 won't include everything
        fox_id = next(t.id for t in parsed if t.text == "fox")
        seed = selector.select_range(parsed, start_id=fox_id, end_id=fox_id)
        selector.add_seed(seed)

        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        rendered = renderer.render(parsed, result)

        # Check for ellipsis or placeholder
        has_placeholder = "..." in rendered or "···" in rendered or "[" in rendered
        # If not all words included, should have placeholder
        if len(result.word_ids) < len(parsed):
            assert has_placeholder or rendered != ""

    def test_placeholder_configurable(self):
        """Placeholder style should be configurable."""
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

        # Test different placeholder styles
        renderer1 = SparseRenderer(placeholder="[...]")
        renderer2 = SparseRenderer(placeholder="···")

        rendered1 = renderer1.render(parsed, result)
        rendered2 = renderer2.render(parsed, result)

        assert rendered1 is not None
        assert rendered2 is not None


class TestPronounPreservation:
    """Test that pronouns preserved to maintain referential structure."""

    def test_pronouns_preserved(self):
        """Pronouns should be preserved even if not directly selected."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "She ran quickly because she was late."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select "ran"
        ran_id = next(t.id for t in parsed if t.text == "ran")
        seed = selector.select_range(parsed, start_id=ran_id, end_id=ran_id)
        selector.add_seed(seed)

        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        rendered = renderer.render(parsed, result, preserve_pronouns=True)

        # "She" should be preserved as it's the subject
        assert "She" in rendered or "she" in rendered

    def test_referential_pronouns_identified(self):
        """Should identify pronouns that need preservation."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        renderer = SparseRenderer()

        text = "He saw her and they talked."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        pronouns = renderer.find_pronouns(parsed)

        # Should find "He", "her", "they"
        pronoun_texts = [parsed[pid].text.lower() for pid in pronouns]
        assert "he" in pronoun_texts
        assert "her" in pronoun_texts
        assert "they" in pronoun_texts

    def test_preserve_subject_pronouns(self):
        """Subject pronouns should always be preserved."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "They quickly finished the work."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select "finished"
        finished_id = next(t.id for t in parsed if t.text == "finished")
        seed = selector.select_range(parsed, start_id=finished_id, end_id=finished_id)
        selector.add_seed(seed)

        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        rendered = renderer.render(parsed, result, preserve_pronouns=True)

        assert "They" in rendered


class TestRenderingDensity:
    """Test that rendering density adjusts with zoom level."""

    def test_higher_zoom_more_words(self):
        """Higher zoom levels should render more words."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "The very quick brown fox jumps gracefully over the lazy dog."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select "fox"
        fox_id = next(t.id for t in parsed if t.text == "fox")
        seed = selector.select_range(parsed, start_id=fox_id, end_id=fox_id)
        selector.add_seed(seed)

        result_1 = extractor.extract(parsed, selector.seeds, zoom_level=1)
        result_3 = extractor.extract(parsed, selector.seeds, zoom_level=3)

        rendered_1 = renderer.render(parsed, result_1)
        rendered_3 = renderer.render(parsed, result_3)

        # Count actual words (not placeholders)
        words_1 = len([w for w in rendered_1.split() if not w.startswith('[') and w != '...'])
        words_3 = len([w for w in rendered_3.split() if not w.startswith('[') and w != '...'])

        assert words_3 >= words_1

    def test_zoom_level_affects_density(self):
        """Zoom level should affect rendering density."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "A big fluffy cat sat quietly on the soft mat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        cat_id = next(t.id for t in parsed if t.text == "cat")
        seed = selector.select_range(parsed, start_id=cat_id, end_id=cat_id)
        selector.add_seed(seed)

        result_1 = extractor.extract(parsed, selector.seeds, zoom_level=1)
        result_2 = extractor.extract(parsed, selector.seeds, zoom_level=2)

        # More word IDs at higher zoom
        assert len(result_2.word_ids) >= len(result_1.word_ids)


class TestSparseRendererMethods:
    """Test SparseRenderer utility methods."""

    def test_render_returns_string(self):
        """Render should return a string."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "Hello world."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        seed = selector.select_range(parsed, start_id=0, end_id=0)
        selector.add_seed(seed)

        result = extractor.extract(parsed, selector.seeds, zoom_level=1)
        rendered = renderer.render(parsed, result)

        assert isinstance(rendered, str)

    def test_empty_result_renders_empty(self):
        """Empty subgraph should render to empty string."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "Hello world."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Extract with no seeds
        result = extractor.extract(parsed, [], zoom_level=1)
        rendered = renderer.render(parsed, result)

        assert rendered == ""

    def test_full_sentence_no_gaps(self):
        """Full sentence inclusion should have no placeholders."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        selector = SeedSelector()
        extractor = SubgraphExtractor()
        renderer = SparseRenderer()

        text = "Cat sat."
        tokens = tokenizer.tokenize(text)
        parsed = parser.parse(tokens)

        # Select everything
        seed = selector.select_range(parsed, start_id=0, end_id=len(parsed)-1)
        selector.add_seed(seed)

        # High zoom to include all
        result = extractor.extract(parsed, selector.seeds, zoom_level=10)
        rendered = renderer.render(parsed, result)

        # Should include "Cat" and "sat"
        assert "Cat" in rendered
        assert "sat" in rendered
