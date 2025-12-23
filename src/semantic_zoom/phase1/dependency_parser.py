"""Dependency parsing (NSM-37).

This module provides dependency parsing that:
- Annotates each token with head pointer and dependency relation
- Enables root (main predicate) queries
- Supports retrieval of noun modifiers with word IDs
"""

from dataclasses import dataclass, field
from typing import List, Optional
import spacy

from semantic_zoom.phase1.tokenizer import Token


@dataclass
class ParsedToken:
    """A token with dependency parse information.
    
    Extends Token with syntactic structure annotations.
    
    Attributes:
        id: Unique sequential integer ID from tokenization
        text: The actual token text
        whitespace_after: Whitespace following this token
        is_punct: Whether this token is punctuation
        start_char: Start character position in original text
        end_char: End character position in original text
        pos: Universal POS tag
        tag: Fine-grained POS tag
        head_id: ID of the syntactic head (-1 for root)
        dep: Dependency relation label (nsubj, dobj, ROOT, etc.)
        children_ids: List of IDs for tokens that depend on this token
    """
    id: int
    text: str
    whitespace_after: str
    is_punct: bool
    start_char: int
    end_char: int
    pos: str
    tag: str
    head_id: int
    dep: str
    children_ids: List[int] = field(default_factory=list)


class DependencyParser:
    """Parser that annotates tokens with syntactic dependency structure.
    
    Uses spaCy for dependency parsing.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize dependency parser with spaCy model.
        
        Args:
            model: spaCy model name (default: en_core_web_sm)
        """
        try:
            self._nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            self._nlp = spacy.load(model)
    
    def parse(self, tokens: List[Token]) -> List[ParsedToken]:
        """Parse tokens and add dependency information.
        
        Args:
            tokens: List of Token objects from tokenizer
            
        Returns:
            List of ParsedToken objects with dependency information
        """
        if not tokens:
            return []
        
        # Reconstruct text for parsing
        text = "".join(t.text + t.whitespace_after for t in tokens)
        doc = self._nlp(text)
        
        # First pass: create ParsedToken objects
        parsed_tokens = []
        for token, spacy_token in zip(tokens, doc):
            # Determine head_id: -1 for root, otherwise the head's index
            if spacy_token.head == spacy_token:  # Root condition
                head_id = -1
            else:
                head_id = spacy_token.head.i
            
            parsed = ParsedToken(
                id=token.id,
                text=token.text,
                whitespace_after=token.whitespace_after,
                is_punct=token.is_punct,
                start_char=token.start_char,
                end_char=token.end_char,
                pos=spacy_token.pos_,
                tag=spacy_token.tag_,
                head_id=head_id,
                dep=spacy_token.dep_,
                children_ids=[],
            )
            parsed_tokens.append(parsed)
        
        # Second pass: populate children_ids
        for pt in parsed_tokens:
            if pt.head_id >= 0 and pt.head_id < len(parsed_tokens):
                parsed_tokens[pt.head_id].children_ids.append(pt.id)
        
        return parsed_tokens
    
    def get_root(self, parsed_tokens: List[ParsedToken]) -> Optional[ParsedToken]:
        """Find the root (main predicate) token.
        
        Args:
            parsed_tokens: List of ParsedToken objects
            
        Returns:
            The root token, or None if not found
        """
        for t in parsed_tokens:
            if t.dep == "ROOT":
                return t
        return None
    
    def get_dependents(self, parsed_tokens: List[ParsedToken], head_id: int) -> List[ParsedToken]:
        """Get all tokens that directly depend on a given head.
        
        Args:
            parsed_tokens: List of ParsedToken objects
            head_id: ID of the head token
            
        Returns:
            List of tokens that have head_id as their head
        """
        return [t for t in parsed_tokens if t.head_id == head_id]
    
    def get_modifiers(self, parsed_tokens: List[ParsedToken], noun_id: int) -> List[ParsedToken]:
        """Get adjectival and other modifiers of a noun.
        
        Args:
            parsed_tokens: List of ParsedToken objects
            noun_id: ID of the noun token
            
        Returns:
            List of modifier tokens (amod, acomp, compound, etc.)
        """
        modifier_deps = {"amod", "acomp", "compound", "nummod", "quantmod"}
        return [
            t for t in parsed_tokens 
            if t.head_id == noun_id and t.dep in modifier_deps
        ]
    
    def get_subtree(self, parsed_tokens: List[ParsedToken], root_id: int) -> List[ParsedToken]:
        """Get all tokens in the subtree rooted at a given token.
        
        Args:
            parsed_tokens: List of ParsedToken objects
            root_id: ID of the subtree root
            
        Returns:
            List of all tokens in the subtree (including root)
        """
        subtree = []
        to_visit = [root_id]
        
        while to_visit:
            current_id = to_visit.pop()
            current = next((t for t in parsed_tokens if t.id == current_id), None)
            if current:
                subtree.append(current)
                to_visit.extend(current.children_ids)
        
        return sorted(subtree, key=lambda t: t.id)
    
    def get_head_chain(self, parsed_tokens: List[ParsedToken], token_id: int) -> List[ParsedToken]:
        """Get the chain of heads from a token up to the root.
        
        Args:
            parsed_tokens: List of ParsedToken objects
            token_id: ID of the starting token
            
        Returns:
            List of tokens from the given token up to (and including) root
        """
        chain = []
        current_id = token_id
        visited = set()
        
        while current_id >= 0 and current_id not in visited:
            visited.add(current_id)
            current = next((t for t in parsed_tokens if t.id == current_id), None)
            if current:
                chain.append(current)
                current_id = current.head_id
            else:
                break
        
        return chain
