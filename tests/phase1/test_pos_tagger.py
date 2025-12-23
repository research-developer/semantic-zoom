"""Tests for part-of-speech tagging (NSM-36)."""

import pytest
from semantic_zoom.phase1.tokenizer import Tokenizer
from semantic_zoom.phase1.pos_tagger import POSTagger, TaggedToken


class TestPOSTagging:
    """Test that each token is annotated with grammatical category."""

    def test_basic_pos_tags(self):
        """Each token should have a POS tag."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("The quick brown fox jumps.")
        tagged = tagger.tag(tokens)
        
        for t in tagged:
            assert t.pos is not None, f"Token '{t.text}' missing POS tag"
            assert len(t.pos) > 0, f"Token '{t.text}' has empty POS tag"

    def test_noun_detection(self):
        """Nouns should be tagged correctly."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("The cat sat on the mat.")
        tagged = tagger.tag(tokens)
        
        cat_token = next(t for t in tagged if t.text == "cat")
        mat_token = next(t for t in tagged if t.text == "mat")
        
        assert cat_token.pos == "NOUN", f"Expected 'cat' to be NOUN, got {cat_token.pos}"
        assert mat_token.pos == "NOUN", f"Expected 'mat' to be NOUN, got {mat_token.pos}"

    def test_verb_detection(self):
        """Verbs should be tagged correctly."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("She runs quickly.")
        tagged = tagger.tag(tokens)
        
        runs_token = next(t for t in tagged if t.text == "runs")
        assert runs_token.pos == "VERB", f"Expected 'runs' to be VERB, got {runs_token.pos}"

    def test_adjective_detection(self):
        """Adjectives should be tagged correctly."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("The quick brown fox.")
        tagged = tagger.tag(tokens)
        
        quick_token = next(t for t in tagged if t.text == "quick")
        brown_token = next(t for t in tagged if t.text == "brown")
        
        assert quick_token.pos == "ADJ", f"Expected 'quick' to be ADJ, got {quick_token.pos}"
        assert brown_token.pos == "ADJ", f"Expected 'brown' to be ADJ, got {brown_token.pos}"

    def test_fine_grained_tag(self):
        """Fine-grained tag should also be available."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("She runs quickly.")
        tagged = tagger.tag(tokens)
        
        runs_token = next(t for t in tagged if t.text == "runs")
        assert runs_token.tag is not None, "Fine-grained tag should be present"
        # VBZ = verb, 3rd person singular present
        assert runs_token.tag == "VBZ", f"Expected 'runs' tag to be VBZ, got {runs_token.tag}"


class TestAmbiguousWords:
    """Test that ambiguous words get context-appropriate tag with confidence."""

    def test_run_as_noun_vs_verb(self):
        """'run' should be tagged differently based on context."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        # As noun
        tokens1 = tokenizer.tokenize("I went for a run.")
        tagged1 = tagger.tag(tokens1)
        run_noun = next(t for t in tagged1 if t.text == "run")
        
        # As verb
        tokens2 = tokenizer.tokenize("I run every day.")
        tagged2 = tagger.tag(tokens2)
        run_verb = next(t for t in tagged2 if t.text == "run")
        
        assert run_noun.pos == "NOUN", f"Expected 'run' as noun, got {run_noun.pos}"
        assert run_verb.pos == "VERB", f"Expected 'run' as verb, got {run_verb.pos}"

    def test_confidence_score(self):
        """Tokens should have a confidence score for their tag."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("The cat sat.")
        tagged = tagger.tag(tokens)
        
        for t in tagged:
            assert hasattr(t, 'confidence'), f"Token '{t.text}' missing confidence"
            assert 0.0 <= t.confidence <= 1.0, f"Confidence should be 0-1, got {t.confidence}"


class TestQueryByCategory:
    """Test that tokens are queryable by category with word IDs."""

    def test_get_nouns(self):
        """Should be able to query all nouns with their IDs."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("The cat and dog saw a bird.")
        tagged = tagger.tag(tokens)

        nouns = tagger.get_by_pos(tagged, "NOUN")
        noun_texts = [n.text for n in nouns]

        assert "cat" in noun_texts
        assert "dog" in noun_texts
        assert "bird" in noun_texts

    def test_get_verbs(self):
        """Should be able to query all verbs with their IDs."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("She walks and talks constantly.")
        tagged = tagger.tag(tokens)
        
        verbs = tagger.get_by_pos(tagged, "VERB")
        verb_texts = [v.text for v in verbs]
        
        assert "walks" in verb_texts
        assert "talks" in verb_texts

    def test_query_preserves_ids(self):
        """Queried tokens should retain their original sequential IDs."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("Big cats run fast.")
        tagged = tagger.tag(tokens)
        
        nouns = tagger.get_by_pos(tagged, "NOUN")
        cat_token = next(n for n in nouns if n.text == "cats")
        
        # "cats" should be token ID 1 (Big=0, cats=1, run=2, fast=3, .=4)
        assert cat_token.id == 1, f"Expected ID 1, got {cat_token.id}"


class TestTaggedTokenAttributes:
    """Test TaggedToken dataclass attributes."""

    def test_inherits_token_attributes(self):
        """TaggedToken should have all original Token attributes plus POS info."""
        tokenizer = Tokenizer()
        tagger = POSTagger()
        
        tokens = tokenizer.tokenize("Hello world.")
        tagged = tagger.tag(tokens)
        
        for t in tagged:
            # Original Token attributes
            assert hasattr(t, 'id')
            assert hasattr(t, 'text')
            assert hasattr(t, 'whitespace_after')
            assert hasattr(t, 'is_punct')
            assert hasattr(t, 'start_char')
            assert hasattr(t, 'end_char')
            # New POS attributes
            assert hasattr(t, 'pos')
            assert hasattr(t, 'tag')
            assert hasattr(t, 'confidence')
