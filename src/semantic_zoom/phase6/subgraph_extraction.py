"""Subgraph extraction algorithm (NSM-54).

This module provides subgraph extraction that:
- Takes seeds + zoom level and produces connected subgraph
- Zoom level 1: directly connected nodes only
- Zoom level N: N-hop neighborhood via dependency structure
- Produces quasi-deterministic results
- Optionally skips copulas/auxiliaries for semantic focus
"""

from dataclasses import dataclass
from typing import List, Optional, Set

from semantic_zoom.phase1.dependency_parser import ParsedToken
from semantic_zoom.phase6.seed_selection import Seed


# Copular/auxiliary verbs to optionally skip during expansion
COPULA_LEMMAS = {"be", "is", "am", "are", "was", "were", "been", "being"}


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
    - Optional copula skipping for semantic focus
    """

    def __init__(self, skip_copulas: bool = False):
        """Initialize extractor.

        Args:
            skip_copulas: If True, copulas/auxiliaries are treated as
                "transparent" during expansion - we expand through them
                to reach content words without including them in results.
                Default False for backward compatibility.
        """
        self.skip_copulas = skip_copulas

    def extract(
        self,
        tokens: List[ParsedToken],
        seeds: List[Seed],
        zoom_level: int = 1,
        skip_copulas: Optional[bool] = None
    ) -> SubgraphResult:
        """Extract subgraph from seeds at given zoom level.

        Args:
            tokens: Parsed tokens from the document
            seeds: List of seeds to extract around
            zoom_level: Number of hops from seeds (1 = direct only)
            skip_copulas: Override instance setting for this extraction.
                If None, uses instance default.

        Returns:
            SubgraphResult with word IDs covering the subgraph
        """
        if not seeds:
            return SubgraphResult(
                word_ids=[],
                zoom_level=zoom_level,
                seed_ids=[]
            )

        # Resolve skip_copulas setting
        should_skip = skip_copulas if skip_copulas is not None else self.skip_copulas

        # Identify copula tokens for filtering
        copula_ids = self._find_copula_ids(tokens) if should_skip else set()

        # Collect all seed word IDs as starting points
        seed_word_ids: Set[int] = set()
        for seed in seeds:
            seed_word_ids.update(seed.word_ids)

        # Build adjacency map for efficient traversal
        adjacency = self._build_adjacency(tokens, copula_ids if should_skip else set())

        # Expand from seeds by zoom_level hops
        expanded_ids = self._expand_n_hops(
            seed_word_ids,
            adjacency,
            zoom_level,
            len(tokens)
        )

        # Remove copulas from result if skipping
        if should_skip:
            expanded_ids = expanded_ids - copula_ids

        return SubgraphResult(
            word_ids=sorted(expanded_ids),
            zoom_level=zoom_level,
            seed_ids=[s.id for s in seeds]
        )

    def _find_copula_ids(self, tokens: List[ParsedToken]) -> Set[int]:
        """Find token IDs of copular/auxiliary verbs.

        Identifies forms of "be" that act as linking verbs rather than
        main content verbs. Only marks as copula if:
        - Token is AUX part-of-speech
        - Token is ROOT (main verb position in copular construction)
        - Token text is a form of "be"

        Args:
            tokens: Parsed tokens

        Returns:
            Set of token IDs that are copulas
        """
        copula_ids: Set[int] = set()

        for token in tokens:
            # Only skip if it's:
            # 1. An AUX that is ROOT (copular main verb)
            # 2. A form of "be"
            if (token.pos == "AUX" and
                token.dep == "ROOT" and
                token.text.lower() in COPULA_LEMMAS):
                copula_ids.add(token.id)

        return copula_ids
    
    def _build_adjacency(
        self,
        tokens: List[ParsedToken],
        transparent_ids: Optional[Set[int]] = None
    ) -> dict:
        """Build adjacency map from dependency structure.

        Creates bidirectional links between tokens and their heads/dependents.
        Transparent tokens (like copulas) are traversed through but create
        direct links between their dependents.

        Args:
            tokens: Parsed tokens
            transparent_ids: Token IDs to treat as transparent (skip over)

        Returns:
            Dict mapping token ID to set of adjacent token IDs
        """
        transparent = transparent_ids or set()
        adjacency: dict[int, Set[int]] = {t.id: set() for t in tokens}

        for token in tokens:
            head_id = token.head_id

            # Skip transparent tokens: connect through them
            if head_id in transparent and head_id >= 0:
                # Find the transparent token's dependents (our siblings)
                transparent_token = next(
                    (t for t in tokens if t.id == head_id), None
                )
                if transparent_token:
                    for sibling_id in transparent_token.children_ids:
                        if sibling_id != token.id and sibling_id < len(tokens):
                            adjacency[token.id].add(sibling_id)
                            adjacency[sibling_id].add(token.id)

            # Link to head (if not root and head not transparent)
            if head_id >= 0 and head_id < len(tokens):
                if head_id not in transparent:
                    adjacency[token.id].add(head_id)
                    adjacency[head_id].add(token.id)

            # Link to children (if not transparent)
            for child_id in token.children_ids:
                if child_id < len(tokens) and child_id not in transparent:
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
