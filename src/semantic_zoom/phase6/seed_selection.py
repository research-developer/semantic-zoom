"""Seed statement selection (NSM-53).

This module provides seed selection that:
- Captures word ID ranges as seeds from text selection
- Identifies containing sentence/clause for each seed
- Stores multiple seeds with graph node highlighting
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
import uuid

from semantic_zoom.phase1.dependency_parser import ParsedToken


@dataclass
class Seed:
    """A seed selection representing a text span for zoom operations.
    
    Attributes:
        id: Unique identifier for this seed
        start_id: First token ID in the selection
        end_id: Last token ID in the selection (inclusive)
        word_ids: List of all token IDs in the selection
        text: The selected text
        sentence_start_id: Start of containing sentence
        sentence_end_id: End of containing sentence
        clause_root_id: Root token ID of containing clause
    """
    id: str
    start_id: int
    end_id: int
    word_ids: List[int]
    text: str
    sentence_start_id: Optional[int] = None
    sentence_end_id: Optional[int] = None
    clause_root_id: Optional[int] = None


class SeedSelector:
    """Manages seed selections for semantic zoom operations.
    
    Provides:
    - Word ID range selection
    - Character offset to word ID mapping
    - Sentence/clause boundary detection
    - Multiple seed storage with highlighting
    """
    
    def __init__(self):
        """Initialize seed selector."""
        self._seeds: List[Seed] = []
    
    @property
    def seeds(self) -> List[Seed]:
        """Get all stored seeds."""
        return self._seeds.copy()
    
    def select_range(
        self, 
        tokens: List[ParsedToken], 
        start_id: int, 
        end_id: int
    ) -> Seed:
        """Create a seed from a word ID range.
        
        Args:
            tokens: Parsed tokens from the document
            start_id: First token ID to include
            end_id: Last token ID to include (inclusive)
            
        Returns:
            Seed object representing the selection
        """
        # Get tokens in range
        word_ids = list(range(start_id, end_id + 1))
        selected_tokens = [t for t in tokens if t.id in word_ids]
        
        # Build text from selected tokens
        text = self._build_text(selected_tokens)
        
        # Find containing sentence
        sentence_start, sentence_end = self._find_sentence_bounds(tokens, start_id)
        
        # Find clause root
        clause_root = self._find_clause_root(tokens, start_id)
        
        return Seed(
            id=str(uuid.uuid4())[:8],
            start_id=start_id,
            end_id=end_id,
            word_ids=word_ids,
            text=text,
            sentence_start_id=sentence_start,
            sentence_end_id=sentence_end,
            clause_root_id=clause_root,
        )
    
    def select_by_chars(
        self, 
        tokens: List[ParsedToken], 
        start_char: int, 
        end_char: int
    ) -> Seed:
        """Create a seed from character offsets.
        
        Args:
            tokens: Parsed tokens from the document
            start_char: Start character offset
            end_char: End character offset
            
        Returns:
            Seed object representing the selection
        """
        # Find tokens that overlap with the character range
        start_id = None
        end_id = None
        
        for t in tokens:
            # Token overlaps if it starts before end and ends after start
            if t.start_char < end_char and t.end_char > start_char:
                if start_id is None:
                    start_id = t.id
                end_id = t.id
        
        if start_id is None:
            start_id = 0
            end_id = 0
        
        return self.select_range(tokens, start_id, end_id)
    
    def add_seed(self, seed: Seed) -> None:
        """Add a seed to the stored seeds.
        
        Args:
            seed: Seed to add
        """
        self._seeds.append(seed)
    
    def remove_seed(self, seed_id: str) -> bool:
        """Remove a seed by ID.
        
        Args:
            seed_id: ID of seed to remove
            
        Returns:
            True if seed was found and removed
        """
        for i, s in enumerate(self._seeds):
            if s.id == seed_id:
                self._seeds.pop(i)
                return True
        return False
    
    def clear_seeds(self) -> None:
        """Remove all stored seeds."""
        self._seeds.clear()
    
    def get_all_seed_word_ids(self) -> Set[int]:
        """Get union of all word IDs from all seeds.
        
        Returns:
            Set of all word IDs covered by any seed
        """
        all_ids: Set[int] = set()
        for seed in self._seeds:
            all_ids.update(seed.word_ids)
        return all_ids
    
    def get_clause_token_ids(
        self, 
        tokens: List[ParsedToken], 
        seed: Seed
    ) -> List[int]:
        """Get all token IDs in the clause containing the seed.
        
        Args:
            tokens: Parsed tokens from the document
            seed: Seed to find clause for
            
        Returns:
            List of token IDs in the containing clause
        """
        if seed.clause_root_id is None:
            return seed.word_ids
        
        # Get subtree of clause root
        return self._get_subtree_ids(tokens, seed.clause_root_id)
    
    def _build_text(self, tokens: List[ParsedToken]) -> str:
        """Build text string from tokens.
        
        Args:
            tokens: Tokens to build text from
            
        Returns:
            Reconstructed text
        """
        if not tokens:
            return ""
        
        tokens = sorted(tokens, key=lambda t: t.id)
        parts = []
        for i, t in enumerate(tokens):
            parts.append(t.text)
            if i < len(tokens) - 1:
                parts.append(t.whitespace_after)
        
        return "".join(parts)
    
    def _find_sentence_bounds(
        self, 
        tokens: List[ParsedToken], 
        token_id: int
    ) -> tuple[Optional[int], Optional[int]]:
        """Find sentence boundaries containing a token.
        
        Uses punctuation (., !, ?) as sentence delimiters.
        
        Args:
            tokens: All tokens in document
            token_id: Token ID to find sentence for
            
        Returns:
            Tuple of (start_id, end_id) for the sentence
        """
        sentence_end_punct = {'.', '!', '?'}
        
        # Find sentence start (after previous sentence-ending punct)
        start_id = 0
        for t in tokens:
            if t.id >= token_id:
                break
            if t.text in sentence_end_punct:
                start_id = t.id + 1
        
        # Find sentence end (next sentence-ending punct)
        end_id = len(tokens) - 1
        for t in tokens:
            if t.id >= token_id and t.text in sentence_end_punct:
                end_id = t.id
                break
        
        return start_id, end_id
    
    def _find_clause_root(
        self, 
        tokens: List[ParsedToken], 
        token_id: int
    ) -> Optional[int]:
        """Find the root of the clause containing a token.
        
        Traverses up the dependency tree to find clause head.
        
        Args:
            tokens: All tokens in document
            token_id: Token ID to find clause root for
            
        Returns:
            ID of clause root token
        """
        token = next((t for t in tokens if t.id == token_id), None)
        if token is None:
            return None
        
        # Clause-marking dependencies
        clause_deps = {"ccomp", "xcomp", "advcl", "relcl", "acl", "ROOT"}
        
        # Walk up to find clause root
        current = token
        visited = {current.id}
        
        while current.head_id >= 0 and current.head_id not in visited:
            if current.dep in clause_deps or current.dep == "ROOT":
                return current.id
            
            visited.add(current.head_id)
            head = next((t for t in tokens if t.id == current.head_id), None)
            if head is None:
                break
            current = head
        
        # Return the highest node we reached
        return current.id
    
    def _get_subtree_ids(self, tokens: List[ParsedToken], root_id: int) -> List[int]:
        """Get all token IDs in a subtree.
        
        Args:
            tokens: All tokens
            root_id: Root token ID
            
        Returns:
            List of token IDs in subtree
        """
        ids = [root_id]
        for t in tokens:
            if t.head_id == root_id and t.id != root_id:
                ids.extend(self._get_subtree_ids(tokens, t.id))
        return sorted(ids)
