"""Subgraph extraction algorithm (NSM-54).

This module provides subgraph extraction that:
- Takes seeds + zoom level and produces connected subgraph
- Zoom level 1: directly connected nodes only
- Zoom level N: N-hop neighborhood via dependency structure
- Produces quasi-deterministic results
"""

from dataclasses import dataclass
from typing import List, Set

from semantic_zoom.phase1.dependency_parser import ParsedToken
from semantic_zoom.phase6.seed_selection import Seed


@dataclass
class SubgraphResult:
    """Result of subgraph extraction.
    
    Attributes:
        word_ids: List of token IDs in the extracted subgraph
        zoom_level: The zoom level used for extraction
        seed_ids: IDs of the seeds used for extraction
    """
    word_ids: List[int]
    zoom_level: int
    seed_ids: List[str]


class SubgraphExtractor:
    """Extracts connected subgraphs from seeds at specified zoom levels.
    
    Provides:
    - N-hop neighborhood extraction via dependency tree
    - Quasi-deterministic extraction (same input → same output)
    - Multi-seed support with subgraph union
    """
    
    def extract(
        self,
        tokens: List[ParsedToken],
        seeds: List[Seed],
        zoom_level: int = 1
    ) -> SubgraphResult:
        """Extract subgraph from seeds at given zoom level.
        
        Args:
            tokens: Parsed tokens from the document
            seeds: List of seeds to extract around
            zoom_level: Number of hops from seeds (1 = direct only)
            
        Returns:
            SubgraphResult with word IDs covering the subgraph
        """
        if not seeds:
            return SubgraphResult(
                word_ids=[],
                zoom_level=zoom_level,
                seed_ids=[]
            )
        
        # Collect all seed word IDs as starting points
        seed_word_ids: Set[int] = set()
        for seed in seeds:
            seed_word_ids.update(seed.word_ids)
        
        # Build adjacency map for efficient traversal
        adjacency = self._build_adjacency(tokens)
        
        # Expand from seeds by zoom_level hops
        expanded_ids = self._expand_n_hops(
            seed_word_ids, 
            adjacency, 
            zoom_level,
            len(tokens)
        )
        
        return SubgraphResult(
            word_ids=sorted(expanded_ids),
            zoom_level=zoom_level,
            seed_ids=[s.id for s in seeds]
        )
    
    def _build_adjacency(self, tokens: List[ParsedToken]) -> dict:
        """Build adjacency map from dependency structure.
        
        Creates bidirectional links between tokens and their heads/dependents.
        
        Args:
            tokens: Parsed tokens
            
        Returns:
            Dict mapping token ID to set of adjacent token IDs
        """
        adjacency: dict[int, Set[int]] = {t.id: set() for t in tokens}
        
        for token in tokens:
            # Link to head (if not root)
            if token.head_id >= 0 and token.head_id < len(tokens):
                adjacency[token.id].add(token.head_id)
                adjacency[token.head_id].add(token.id)
            
            # Link to children
            for child_id in token.children_ids:
                if child_id < len(tokens):
                    adjacency[token.id].add(child_id)
                    adjacency[child_id].add(token.id)
        
        return adjacency
    
    def _expand_n_hops(
        self,
        start_ids: Set[int],
        adjacency: dict,
        n_hops: int,
        max_id: int
    ) -> Set[int]:
        """Expand from starting IDs by N hops through adjacency.
        
        Args:
            start_ids: Initial token IDs
            adjacency: Adjacency map
            n_hops: Number of hops to expand
            max_id: Maximum valid token ID
            
        Returns:
            Set of all token IDs within N hops
        """
        current = set(start_ids)
        expanded = set(start_ids)
        
        for _ in range(n_hops):
            next_frontier: Set[int] = set()
            for token_id in current:
                if token_id in adjacency:
                    for neighbor_id in adjacency[token_id]:
                        if neighbor_id not in expanded and neighbor_id < max_id:
                            next_frontier.add(neighbor_id)
            
            expanded.update(next_frontier)
            current = next_frontier
            
            if not current:
                break
        
        return expanded
    
    def extract_with_similarity(
        self,
        tokens: List[ParsedToken],
        seeds: List[Seed],
        zoom_level: int = 1,
        similarity_threshold: float = 0.5
    ) -> SubgraphResult:
        """Extract subgraph using both structure and semantic similarity.
        
        For higher zoom levels, includes semantically similar tokens
        even if not structurally adjacent.
        
        Args:
            tokens: Parsed tokens from the document
            seeds: List of seeds to extract around
            zoom_level: Number of hops from seeds
            similarity_threshold: Minimum similarity to include
            
        Returns:
            SubgraphResult with word IDs
        """
        # Start with structural extraction
        base_result = self.extract(tokens, seeds, zoom_level)
        
        # For zoom level > 2, could add similarity-based expansion
        # This is a placeholder for future embedding-based expansion
        # Would use sentence-transformers for token/phrase embeddings
        
        return base_result
    
    def get_subgraph_text(
        self,
        tokens: List[ParsedToken],
        result: SubgraphResult
    ) -> str:
        """Get text representation of extracted subgraph.
        
        Args:
            tokens: All parsed tokens
            result: Extraction result
            
        Returns:
            Text of tokens in subgraph (may have gaps)
        """
        subgraph_tokens = [t for t in tokens if t.id in result.word_ids]
        subgraph_tokens.sort(key=lambda t: t.id)
        
        if not subgraph_tokens:
            return ""
        
        parts = []
        for i, t in enumerate(subgraph_tokens):
            parts.append(t.text)
            if i < len(subgraph_tokens) - 1:
                parts.append(t.whitespace_after)
        
        return "".join(parts)
