"""Tests for dependency parsing (NSM-37)."""

import pytest
from semantic_zoom.phase1.tokenizer import Tokenizer
from semantic_zoom.phase1.dependency_parser import DependencyParser, ParsedToken


class TestDependencyRelations:
    """Test that each token has head pointer and dependency relation label."""

    def test_tokens_have_head(self):
        """Each token should have a head pointer."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The cat sat.")
        parsed = parser.parse(tokens)
        
        for t in parsed:
            assert hasattr(t, 'head_id'), f"Token '{t.text}' missing head_id"
            # Head can be -1 for root, or a valid token ID
            assert t.head_id >= -1, f"Invalid head_id {t.head_id} for '{t.text}'"

    def test_tokens_have_dep_label(self):
        """Each token should have a dependency relation label."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The cat sat.")
        parsed = parser.parse(tokens)
        
        for t in parsed:
            assert hasattr(t, 'dep'), f"Token '{t.text}' missing dep label"
            assert t.dep is not None and len(t.dep) > 0

    def test_det_relation(self):
        """Determiner should have 'det' relation to its noun."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The cat sat.")
        parsed = parser.parse(tokens)
        
        the_token = next(t for t in parsed if t.text == "The")
        cat_token = next(t for t in parsed if t.text == "cat")
        
        assert the_token.dep == "det", f"Expected 'det', got '{the_token.dep}'"
        assert the_token.head_id == cat_token.id

    def test_nsubj_relation(self):
        """Subject noun should have 'nsubj' relation to verb."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The cat sat.")
        parsed = parser.parse(tokens)
        
        cat_token = next(t for t in parsed if t.text == "cat")
        sat_token = next(t for t in parsed if t.text == "sat")
        
        assert cat_token.dep == "nsubj", f"Expected 'nsubj', got '{cat_token.dep}'"
        assert cat_token.head_id == sat_token.id


class TestRootQuery:
    """Test that root (main predicate) is queryable."""

    def test_find_root(self):
        """Should be able to find the root token."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The cat sat on the mat.")
        parsed = parser.parse(tokens)
        
        root = parser.get_root(parsed)
        assert root is not None, "Should find a root"
        assert root.text == "sat", f"Expected 'sat' as root, got '{root.text}'"

    def test_root_has_special_marker(self):
        """Root should have ROOT dependency label and head_id of -1."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("She runs.")
        parsed = parser.parse(tokens)
        
        root = parser.get_root(parsed)
        assert root.dep == "ROOT", f"Expected ROOT, got '{root.dep}'"
        assert root.head_id == -1, f"Root head_id should be -1, got {root.head_id}"

    def test_complex_sentence_root(self):
        """Complex sentence should still have identifiable root."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("When the rain came, we stayed inside.")
        parsed = parser.parse(tokens)
        
        root = parser.get_root(parsed)
        assert root is not None
        # "stayed" should be the main verb
        assert root.text == "stayed", f"Expected 'stayed' as root, got '{root.text}'"


class TestModifierRetrieval:
    """Test that noun dependents (modifiers) are retrievable with word IDs."""

    def test_get_dependents(self):
        """Should be able to get all dependents of a token."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The big cat sat.")
        parsed = parser.parse(tokens)
        
        cat_token = next(t for t in parsed if t.text == "cat")
        dependents = parser.get_dependents(parsed, cat_token.id)
        
        dep_texts = [d.text for d in dependents]
        assert "The" in dep_texts
        assert "big" in dep_texts

    def test_get_modifiers(self):
        """Should be able to get modifiers (adjectives) of a noun."""
        tokenizer = Tokenizer()
        parser = DependencyParser()

        tokens = tokenizer.tokenize("A big red ball rolled.")
        parsed = parser.parse(tokens)

        ball_token = next(t for t in parsed if t.text == "ball")
        modifiers = parser.get_modifiers(parsed, ball_token.id)

        mod_texts = [m.text for m in modifiers]
        assert "big" in mod_texts
        assert "red" in mod_texts

    def test_dependents_have_correct_ids(self):
        """Retrieved dependents should have their original token IDs."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The small cat slept.")
        parsed = parser.parse(tokens)
        
        cat_token = next(t for t in parsed if t.text == "cat")
        dependents = parser.get_dependents(parsed, cat_token.id)
        
        the_dep = next(d for d in dependents if d.text == "The")
        small_dep = next(d for d in dependents if d.text == "small")
        
        assert the_dep.id == 0, f"Expected 'The' id=0, got {the_dep.id}"
        assert small_dep.id == 1, f"Expected 'small' id=1, got {small_dep.id}"


class TestParsedTokenAttributes:
    """Test ParsedToken dataclass attributes."""

    def test_inherits_tagged_token_attributes(self):
        """ParsedToken should have all Token and POS attributes plus dep info."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("Hello world.")
        parsed = parser.parse(tokens)
        
        for t in parsed:
            # Original Token attributes
            assert hasattr(t, 'id')
            assert hasattr(t, 'text')
            assert hasattr(t, 'whitespace_after')
            assert hasattr(t, 'is_punct')
            # POS attributes
            assert hasattr(t, 'pos')
            assert hasattr(t, 'tag')
            # Dependency attributes
            assert hasattr(t, 'head_id')
            assert hasattr(t, 'dep')

    def test_children_list_available(self):
        """ParsedToken should have list of child IDs for traversal."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        
        tokens = tokenizer.tokenize("The cat sat.")
        parsed = parser.parse(tokens)
        
        for t in parsed:
            assert hasattr(t, 'children_ids')
            assert isinstance(t.children_ids, list)
