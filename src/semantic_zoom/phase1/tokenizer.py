"""Word tokenization with sequential ID assignment (NSM-35).

This module provides tokenization that:
- Assigns unique sequential integer IDs starting from 0
- Preserves punctuation as separate tokens
- Enables full roundtrip recovery of original text
"""

from dataclasses import dataclass
from typing import List
import spacy


@dataclass
class Token:
    """A token with sequential ID and metadata for roundtrip recovery.
    
    Attributes:
        id: Unique sequential integer ID starting from 0
        text: The actual token text
        whitespace_after: Whitespace following this token (for reconstruction)
        is_punct: Whether this token is punctuation
        start_char: Start character position in original text
        end_char: End character position in original text
    """
    id: int
    text: str
    whitespace_after: str
    is_punct: bool
    start_char: int
    end_char: int


class Tokenizer:
    """Tokenizer that assigns sequential IDs and preserves roundtrip capability.
    
    Uses spaCy for robust tokenization while maintaining:
    - Unique sequential IDs starting from 0
    - Punctuation as separate tokens
    - Full text reconstruction capability
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize tokenizer with spaCy model.
        
        Args:
            model: spaCy model name (default: en_core_web_sm)
        """
        try:
            self._nlp = spacy.load(model)
        except OSError:
            # Model not installed, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            self._nlp = spacy.load(model)
        
        # Disable components we don't need for tokenization
        self._nlp.select_pipes(enable=["tok2vec"])
    
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize text and assign sequential IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of Token objects with sequential IDs starting from 0
        """
        if not text or text.isspace():
            return []
        
        doc = self._nlp(text)
        tokens = []
        
        for idx, spacy_token in enumerate(doc):
            token = Token(
                id=idx,
                text=spacy_token.text,
                whitespace_after=spacy_token.whitespace_,
                is_punct=spacy_token.is_punct,
                start_char=spacy_token.idx,
                end_char=spacy_token.idx + len(spacy_token.text),
            )
            tokens.append(token)
        
        return tokens
    
    def reconstruct(self, tokens: List[Token]) -> str:
        """Reconstruct original text from tokens.
        
        Args:
            tokens: List of Token objects
            
        Returns:
            Reconstructed text matching the original input
        """
        if not tokens:
            return ""
        
        parts = []
        for token in tokens:
            parts.append(token.text)
            parts.append(token.whitespace_after)
        
        return "".join(parts)
    
    def get_by_id(self, tokens: List[Token], token_id: int) -> Token | None:
        """Retrieve a token by its ID.
        
        Args:
            tokens: List of Token objects
            token_id: The ID to look up
            
        Returns:
            The Token with matching ID, or None if not found
        """
        for token in tokens:
            if token.id == token_id:
                return token
        return None
    
    def get_id_range(self, tokens: List[Token], start_id: int, end_id: int) -> List[Token]:
        """Get tokens within an ID range (inclusive).
        
        Args:
            tokens: List of Token objects
            start_id: Start of range (inclusive)
            end_id: End of range (inclusive)
            
        Returns:
            List of tokens within the specified ID range
        """
        return [t for t in tokens if start_id <= t.id <= end_id]
