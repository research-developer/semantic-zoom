"""Tests for word tokenization with sequential ID assignment (NSM-35)."""

import pytest
from semantic_zoom.phase1.tokenizer import Tokenizer, Token


class TestSequentialIds:
    """Test that each word receives unique sequential integer ID starting from 0."""

    def test_simple_sentence_ids(self):
        """Simple sentence should have sequential IDs starting from 0."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("The cat sat.")
        
        ids = [t.id for t in tokens]
        assert ids == [0, 1, 2, 3], f"Expected [0, 1, 2, 3], got {ids}"

    def test_ids_are_unique(self):
        """All token IDs in a document should be unique."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello, world! How are you?")
        
        ids = [t.id for t in tokens]
        assert len(ids) == len(set(ids)), "Token IDs must be unique"

    def test_ids_are_sequential(self):
        """IDs should be sequential with no gaps."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("This is a longer test sentence with many words.")
        
        ids = [t.id for t in tokens]
        expected = list(range(len(tokens)))
        assert ids == expected, f"Expected sequential IDs {expected}, got {ids}"

    def test_empty_string(self):
        """Empty string should return empty list."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_whitespace_only(self):
        """Whitespace-only string should return empty list."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("   \n\t  ")
        assert tokens == []


class TestPunctuationPreservation:
    """Test that punctuation is preserved as separate tokens with own IDs."""

    def test_period_separate_token(self):
        """Period should be its own token."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello.")
        
        texts = [t.text for t in tokens]
        assert "." in texts, "Period should be a separate token"
        assert len(tokens) == 2, f"Expected 2 tokens (word + period), got {len(tokens)}"

    def test_comma_separate_token(self):
        """Comma should be its own token."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello, world")
        
        texts = [t.text for t in tokens]
        assert "," in texts, "Comma should be a separate token"

    def test_multiple_punctuation(self):
        """Multiple punctuation marks should each be separate tokens."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Wait... what?!")
        
        texts = [t.text for t in tokens]
        # Should have: "Wait", "...", "what", "?", "!"
        # Or potentially: "Wait", ".", ".", ".", "what", "?", "!"
        # Let's check that punctuation is included
        assert any(t.is_punct for t in tokens), "Should have punctuation tokens"

    def test_apostrophe_contractions(self):
        """Contractions like don't should be handled appropriately."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("I don't know")
        
        # Tokens should include the contraction parts
        texts = [t.text for t in tokens]
        assert len(tokens) >= 3, f"Expected at least 3 tokens, got {texts}"


class TestRoundtrip:
    """Test that full roundtrip recovery is possible."""

    def test_simple_roundtrip(self):
        """Reconstructed text should match original."""
        tokenizer = Tokenizer()
        original = "The cat sat."
        tokens = tokenizer.tokenize(original)
        reconstructed = tokenizer.reconstruct(tokens)
        
        assert reconstructed == original, f"Expected '{original}', got '{reconstructed}'"

    def test_complex_roundtrip(self):
        """Complex sentence with various punctuation should roundtrip."""
        tokenizer = Tokenizer()
        original = "Hello, world! How are you doing today?"
        tokens = tokenizer.tokenize(original)
        reconstructed = tokenizer.reconstruct(tokens)
        
        assert reconstructed == original

    def test_multiline_roundtrip(self):
        """Multi-line text should preserve newlines."""
        tokenizer = Tokenizer()
        original = "First line.\nSecond line."
        tokens = tokenizer.tokenize(original)
        reconstructed = tokenizer.reconstruct(tokens)
        
        assert reconstructed == original

    def test_multiple_spaces_roundtrip(self):
        """Multiple spaces should be preserved in roundtrip."""
        tokenizer = Tokenizer()
        original = "Hello  world"  # Double space
        tokens = tokenizer.tokenize(original)
        reconstructed = tokenizer.reconstruct(tokens)
        
        assert reconstructed == original


class TestTokenAttributes:
    """Test Token dataclass attributes."""

    def test_token_has_required_fields(self):
        """Token should have id, text, and whitespace_after."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello world")
        
        for token in tokens:
            assert hasattr(token, 'id'), "Token must have 'id' attribute"
            assert hasattr(token, 'text'), "Token must have 'text' attribute"
            assert hasattr(token, 'whitespace_after'), "Token must have 'whitespace_after' attribute"

    def test_token_is_punct_flag(self):
        """Token should have is_punct flag for punctuation detection."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello, world.")
        
        punct_tokens = [t for t in tokens if t.is_punct]
        assert len(punct_tokens) == 2, f"Expected 2 punct tokens (comma, period), got {punct_tokens}"
