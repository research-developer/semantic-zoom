"""Part-of-speech tagging (NSM-36).

This module provides POS tagging that:
- Annotates each token with grammatical category
- Handles ambiguous words with context-appropriate tags
- Provides confidence scores for tags
- Enables querying tokens by category with word IDs
"""

from dataclasses import dataclass
from typing import List
import spacy

from semantic_zoom.phase1.tokenizer import Token


@dataclass
class TaggedToken:
    """A token with POS tagging information.
    
    Extends Token with grammatical annotations.
    
    Attributes:
        id: Unique sequential integer ID from tokenization
        text: The actual token text
        whitespace_after: Whitespace following this token
        is_punct: Whether this token is punctuation
        start_char: Start character position in original text
        end_char: End character position in original text
        pos: Universal POS tag (NOUN, VERB, ADJ, etc.)
        tag: Fine-grained POS tag (NN, VBZ, JJ, etc.)
        confidence: Confidence score for the tag (0.0-1.0)
    """
    id: int
    text: str
    whitespace_after: str
    is_punct: bool
    start_char: int
    end_char: int
    pos: str
    tag: str
    confidence: float


class POSTagger:
    """POS tagger that annotates tokens with grammatical categories.
    
    Uses spaCy for context-aware tagging with confidence scores.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize POS tagger with spaCy model.
        
        Args:
            model: spaCy model name (default: en_core_web_sm)
        """
        try:
            self._nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            self._nlp = spacy.load(model)
    
    def tag(self, tokens: List[Token]) -> List[TaggedToken]:
        """Add POS tags to tokens.
        
        Re-processes the text through spaCy to get context-aware tags.
        
        Args:
            tokens: List of Token objects from tokenizer
            
        Returns:
            List of TaggedToken objects with POS information
        """
        if not tokens:
            return []
        
        # Reconstruct text for context-aware tagging
        text = "".join(t.text + t.whitespace_after for t in tokens)
        doc = self._nlp(text)
        
        tagged_tokens = []
        for token, spacy_token in zip(tokens, doc):
            tagged = TaggedToken(
                id=token.id,
                text=token.text,
                whitespace_after=token.whitespace_after,
                is_punct=token.is_punct,
                start_char=token.start_char,
                end_char=token.end_char,
                pos=spacy_token.pos_,
                tag=spacy_token.tag_,
                confidence=self._get_confidence(spacy_token),
            )
            tagged_tokens.append(tagged)
        
        return tagged_tokens
    
    def _get_confidence(self, spacy_token) -> float:
        """Extract confidence score for a token's POS tag.
        
        spaCy doesn't directly expose per-token confidence, so we use
        a heuristic based on the token's properties.
        
        Args:
            spacy_token: A spaCy Token object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # spaCy doesn't provide direct confidence scores for POS tags
        # We use a heuristic: common/unambiguous words get high confidence
        # This is a simplified approach - could be enhanced with actual model scores
        
        # Punctuation is always high confidence
        if spacy_token.is_punct:
            return 1.0
        
        # Check if token has multiple possible tags in the lexeme
        # For now, use a default high confidence since spaCy is generally accurate
        return 0.95
    
    def get_by_pos(self, tagged_tokens: List[TaggedToken], pos: str) -> List[TaggedToken]:
        """Get all tokens with a specific POS tag.
        
        Args:
            tagged_tokens: List of TaggedToken objects
            pos: Universal POS tag to filter by (NOUN, VERB, ADJ, etc.)
            
        Returns:
            List of TaggedToken objects matching the POS
        """
        return [t for t in tagged_tokens if t.pos == pos]
    
    def get_by_fine_tag(self, tagged_tokens: List[TaggedToken], tag: str) -> List[TaggedToken]:
        """Get all tokens with a specific fine-grained tag.
        
        Args:
            tagged_tokens: List of TaggedToken objects
            tag: Fine-grained POS tag to filter by (NN, VBZ, JJ, etc.)
            
        Returns:
            List of TaggedToken objects matching the tag
        """
        return [t for t in tagged_tokens if t.tag == tag]
    
    def get_pos_counts(self, tagged_tokens: List[TaggedToken]) -> dict:
        """Get count of each POS tag in the token list.
        
        Args:
            tagged_tokens: List of TaggedToken objects
            
        Returns:
            Dictionary mapping POS tags to their counts
        """
        counts = {}
        for t in tagged_tokens:
            counts[t.pos] = counts.get(t.pos, 0) + 1
        return counts
