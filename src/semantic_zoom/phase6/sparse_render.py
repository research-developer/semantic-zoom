"""Sparse display rendering (NSM-55).

This module provides sparse rendering that:
- Displays subgraph word IDs with visual gaps for omissions
- Uses placeholders to indicate omitted content ([...] or ···)
- Preserves pronouns to maintain referential structure
- Adjusts rendering density with zoom level
"""

from dataclasses import dataclass
from typing import List, Set, Optional

from semantic_zoom.phase1.dependency_parser import ParsedToken
from semantic_zoom.phase6.subgraph_extraction import SubgraphResult


# Common pronouns for preservation
PRONOUNS = {
    # Subject pronouns
    "i", "you", "he", "she", "it", "we", "they",
    # Object pronouns
    "me", "him", "her", "us", "them",
    # Possessive pronouns
    "my", "your", "his", "her", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
    # Reflexive pronouns
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    # Demonstrative pronouns
    "this", "that", "these", "those",
    # Relative pronouns
    "who", "whom", "whose", "which", "that",
    # Interrogative pronouns
    "what", "who", "whom", "whose", "which",
}


class SparseRenderer:
    """Renders subgraphs with visual gaps for omitted content.

    Provides:
    - Gap visualization with configurable placeholders
    - Pronoun preservation for referential structure
    - Density adjustment based on zoom level
    """

    def __init__(self, placeholder: str = "[...]"):
        """Initialize sparse renderer.

        Args:
            placeholder: String to use for indicating omitted content
        """
        self.placeholder = placeholder

    def render(
        self,
        tokens: List[ParsedToken],
        result: SubgraphResult,
        preserve_pronouns: bool = False
    ) -> str:
        """Render subgraph with gaps for omitted tokens.

        Args:
            tokens: All parsed tokens from document
            result: Subgraph extraction result
            preserve_pronouns: Whether to include pronouns even if not in subgraph

        Returns:
            Rendered string with placeholders for gaps
        """
        if not result.word_ids:
            return ""

        # Get IDs to include
        include_ids: Set[int] = set(result.word_ids)

        # Add pronouns if requested
        if preserve_pronouns:
            pronoun_ids = self.find_pronouns(tokens)
            include_ids.update(pronoun_ids)

        # Build rendered output
        parts: List[str] = []
        in_gap = False

        for token in tokens:
            if token.id in include_ids:
                # End any gap
                if in_gap:
                    parts.append(self.placeholder)
                    parts.append(" ")
                    in_gap = False

                # Add token text
                parts.append(token.text)
                parts.append(token.whitespace_after)
            else:
                # Start or continue gap
                in_gap = True

        # Handle trailing gap
        if in_gap and parts:
            parts.append(self.placeholder)

        # Clean up result
        rendered = "".join(parts).strip()

        # Remove duplicate placeholders
        while f"{self.placeholder} {self.placeholder}" in rendered:
            rendered = rendered.replace(
                f"{self.placeholder} {self.placeholder}",
                self.placeholder
            )

        return rendered

    def find_pronouns(self, tokens: List[ParsedToken]) -> Set[int]:
        """Find all pronoun token IDs in the document.

        Args:
            tokens: Parsed tokens

        Returns:
            Set of token IDs that are pronouns
        """
        pronoun_ids: Set[int] = set()

        for token in tokens:
            # Check POS tag for pronoun
            if token.pos == "PRON":
                pronoun_ids.add(token.id)
            # Also check text against common pronouns
            elif token.text.lower() in PRONOUNS:
                pronoun_ids.add(token.id)

        return pronoun_ids

    def render_with_context(
        self,
        tokens: List[ParsedToken],
        result: SubgraphResult,
        context_window: int = 1
    ) -> str:
        """Render subgraph with additional context around included tokens.

        Args:
            tokens: All parsed tokens
            result: Subgraph extraction result
            context_window: Number of tokens to include around each subgraph token

        Returns:
            Rendered string with expanded context
        """
        if not result.word_ids:
            return ""

        # Expand IDs by context window
        expanded_ids: Set[int] = set()
        for word_id in result.word_ids:
            for offset in range(-context_window, context_window + 1):
                new_id = word_id + offset
                if 0 <= new_id < len(tokens):
                    expanded_ids.add(new_id)

        # Create modified result
        expanded_result = SubgraphResult(
            word_ids=sorted(expanded_ids),
            zoom_level=result.zoom_level,
            seed_ids=result.seed_ids
        )

        return self.render(tokens, expanded_result)

    def get_coverage_stats(
        self,
        tokens: List[ParsedToken],
        result: SubgraphResult
    ) -> dict:
        """Get statistics about rendering coverage.

        Args:
            tokens: All parsed tokens
            result: Subgraph extraction result

        Returns:
            Dict with coverage statistics
        """
        total_tokens = len(tokens)
        included_tokens = len(result.word_ids)

        # Count gaps
        gaps = 0
        in_gap = False
        for token in tokens:
            if token.id in result.word_ids:
                if in_gap:
                    gaps += 1
                    in_gap = False
            else:
                in_gap = True
        if in_gap:
            gaps += 1

        return {
            "total_tokens": total_tokens,
            "included_tokens": included_tokens,
            "coverage_percent": (included_tokens / total_tokens * 100) if total_tokens > 0 else 0,
            "gap_count": gaps,
            "zoom_level": result.zoom_level,
        }
