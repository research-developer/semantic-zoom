"""Phase 2 → Phase 3 integration layer.

Connects Phase 2's classified tokens to Phase 3's morphism mapping:
- Maps preposition tokens to categorical symbols
- Identifies focusing adverbs with dependency-based scope resolution
- Maps discourse adverbs to inter-frame relations

Usage:
    from semantic_zoom.phase3.integration import Phase3Processor

    processor = Phase3Processor()
    result = processor.process(tokens)  # tokens from Phase 2
"""

from dataclasses import dataclass, field
from typing import Optional

from semantic_zoom.models import Token
from semantic_zoom.phase3.preposition_symbols import (
    CategoricalSymbol,
    PrepositionMapping,
    map_preposition,
)
from semantic_zoom.phase3.focusing_adverbs import (
    FocusingAdverb,
    ScopeBinding,
    ScopeOperator,
    _ALL_FOCUSING_ADVERBS,
    _get_focus_type,
)
from semantic_zoom.phase3.discourse_adverbs import (
    DiscourseAdverb,
    DiscourseRelation,
    InterFrameMorphism,
    _ALL_MARKERS,
)


@dataclass
class MorphismToken:
    """Token extended with Phase 3 morphism information.

    Extends Phase 2's Token with categorical morphism mappings.
    """
    # Original token reference
    token: Token

    # Preposition mapping (NSM-43)
    prep_mapping: Optional[PrepositionMapping] = None

    # Focusing adverb info (NSM-44)
    is_focusing_adverb: bool = False
    focus_type: Optional[str] = None
    scope_target_idx: Optional[int] = None
    scope_bindings: list[ScopeBinding] = field(default_factory=list)

    # Discourse adverb info (NSM-45)
    is_discourse_adverb: bool = False
    discourse_relation: Optional[DiscourseRelation] = None
    inter_frame_morphism: Optional[InterFrameMorphism] = None


@dataclass
class Phase3Result:
    """Result of Phase 3 processing.

    Contains tokens with morphism annotations plus aggregate collections.
    """
    tokens: list[MorphismToken]
    preposition_mappings: list[PrepositionMapping]
    focusing_adverbs: list[MorphismToken]
    discourse_adverbs: list[MorphismToken]
    inter_frame_morphisms: list[InterFrameMorphism]


class Phase3Processor:
    """Processor that maps Phase 2 tokens to Phase 3 morphisms.

    Uses dependency parsing from Phase 2 for better scope resolution.
    """

    def process(self, tokens: list[Token]) -> Phase3Result:
        """Process Phase 2 tokens through Phase 3 morphism mapping.

        Args:
            tokens: List of Token objects from Phase 2

        Returns:
            Phase3Result with morphism annotations
        """
        if not tokens:
            return Phase3Result(
                tokens=[],
                preposition_mappings=[],
                focusing_adverbs=[],
                discourse_adverbs=[],
                inter_frame_morphisms=[],
            )

        # Create MorphismToken wrappers
        morphism_tokens = [MorphismToken(token=t) for t in tokens]

        # Process each token type
        prep_mappings = self._process_prepositions(morphism_tokens)
        focusing = self._process_focusing_adverbs(morphism_tokens)
        discourse = self._process_discourse_adverbs(morphism_tokens)

        # Extract inter-frame morphisms
        inter_frame = [
            mt.inter_frame_morphism
            for mt in discourse
            if mt.inter_frame_morphism is not None
        ]

        return Phase3Result(
            tokens=morphism_tokens,
            preposition_mappings=prep_mappings,
            focusing_adverbs=focusing,
            discourse_adverbs=discourse,
            inter_frame_morphisms=inter_frame,
        )

    def _process_prepositions(
        self,
        morphism_tokens: list[MorphismToken]
    ) -> list[PrepositionMapping]:
        """Map preposition tokens to categorical symbols.

        Args:
            morphism_tokens: List of MorphismToken wrappers

        Returns:
            List of PrepositionMapping objects
        """
        mappings = []

        for mt in morphism_tokens:
            token = mt.token
            # Check if token is a preposition (ADP = adposition)
            if token.pos == "ADP":
                mapping = map_preposition(token.text)

                # Try to saturate dual-citizenship using context
                if mapping.is_dual_citizen and not mapping.saturated:
                    mapping = self._resolve_dual_citizenship(
                        mapping, mt, morphism_tokens
                    )

                mt.prep_mapping = mapping
                mappings.append(mapping)

        return mappings

    def _resolve_dual_citizenship(
        self,
        mapping: PrepositionMapping,
        current_mt: MorphismToken,
        all_tokens: list[MorphismToken]
    ) -> PrepositionMapping:
        """Resolve dual-citizenship prepositions using context.

        Uses head token and dependency relation to determine category.

        Args:
            mapping: Unsaturated PrepositionMapping
            current_mt: Current MorphismToken
            all_tokens: All tokens for context

        Returns:
            Saturated PrepositionMapping
        """
        token = current_mt.token
        head_idx = token.head_idx

        if 0 <= head_idx < len(all_tokens):
            head = all_tokens[head_idx].token

            # "at" resolution: SPATIAL vs TEMPORAL
            if mapping.original.lower() == "at":
                # Temporal if head is time-related verb or noun
                temporal_indicators = {"time", "hour", "minute", "moment", "o'clock"}
                if head.lemma.lower() in temporal_indicators:
                    return mapping.saturate(CategoricalSymbol.TEMPORAL_AT)
                # Default to spatial
                return mapping.saturate(CategoricalSymbol.SPATIAL_AT)

            # "by" resolution: SPATIAL_PROXIMITY vs AGENT_BY
            if mapping.original.lower() == "by":
                # Agent if in passive construction (dep='agent')
                if token.dep == "agent":
                    return mapping.saturate(CategoricalSymbol.AGENT_BY)
                # Spatial if head is location noun
                return mapping.saturate(CategoricalSymbol.SPATIAL_PROXIMITY)

            # "for" resolution: PURPOSE vs BENEFICIARY
            if mapping.original.lower() == "for":
                # Beneficiary if object is animate (person noun/pronoun)
                # Check the object of the preposition
                for mt in all_tokens:
                    if mt.token.head_idx == token.idx and mt.token.dep == "pobj":
                        if mt.token.pos == "PRON" or (
                            mt.token.pos == "NOUN" and
                            mt.token.tag in ("NN", "NNS", "NNP", "NNPS")
                        ):
                            # Heuristic: proper nouns or pronouns are beneficiaries
                            if mt.token.tag in ("NNP", "NNPS") or mt.token.pos == "PRON":
                                return mapping.saturate(CategoricalSymbol.BENEFICIARY_FOR)
                # Default to purpose
                return mapping.saturate(CategoricalSymbol.PURPOSE_FOR)

        return mapping

    def _process_focusing_adverbs(
        self,
        morphism_tokens: list[MorphismToken]
    ) -> list[MorphismToken]:
        """Identify and process focusing adverbs with scope resolution.

        Uses dependency parsing for accurate scope target identification.

        Args:
            morphism_tokens: List of MorphismToken wrappers

        Returns:
            List of MorphismToken objects that are focusing adverbs
        """
        focusing = []

        for i, mt in enumerate(morphism_tokens):
            token = mt.token
            lemma = token.lemma.lower()
            text = token.text.lower()

            # Check if this is a focusing adverb
            if lemma in _ALL_FOCUSING_ADVERBS or text in _ALL_FOCUSING_ADVERBS:
                mt.is_focusing_adverb = True
                mt.focus_type = _get_focus_type(lemma if lemma in _ALL_FOCUSING_ADVERBS else text)

                # Resolve scope target using dependency parsing
                scope_target = self._resolve_scope_target(i, morphism_tokens)
                mt.scope_target_idx = scope_target

                # Generate scope bindings
                mt.scope_bindings = self._generate_scope_bindings(
                    i, morphism_tokens
                )

                focusing.append(mt)

        return focusing

    def _resolve_scope_target(
        self,
        adverb_idx: int,
        morphism_tokens: list[MorphismToken]
    ) -> Optional[int]:
        """Resolve scope target for a focusing adverb using dependencies.

        Focusing adverbs typically scope over their syntactic sister.

        Args:
            adverb_idx: Index of the focusing adverb
            morphism_tokens: All tokens

        Returns:
            Index of the scope target token, or None
        """
        adverb_token = morphism_tokens[adverb_idx].token

        # Strategy 1: Check syntactic head
        head_idx = adverb_token.head_idx
        if 0 <= head_idx < len(morphism_tokens):
            head = morphism_tokens[head_idx].token

            # If head is a verb, the scope could be the verb's subject or object
            if head.pos == "VERB":
                # Check for adjacent NP (object or subject)
                for mt in morphism_tokens:
                    if mt.token.head_idx == head_idx:
                        if mt.token.dep in ("dobj", "nsubj", "attr"):
                            return mt.token.idx

        # Strategy 2: Immediate right sibling (common pattern)
        next_idx = adverb_idx + 1
        if next_idx < len(morphism_tokens):
            next_token = morphism_tokens[next_idx].token
            # If next token is content word (not function word)
            if next_token.pos in ("NOUN", "PROPN", "VERB", "ADJ", "NUM"):
                return next_idx

        return head_idx if 0 <= head_idx < len(morphism_tokens) else None

    def _generate_scope_bindings(
        self,
        adverb_idx: int,
        morphism_tokens: list[MorphismToken]
    ) -> list[ScopeBinding]:
        """Generate possible scope bindings for a focusing adverb.

        Uses dependency structure to enumerate plausible interpretations.

        Args:
            adverb_idx: Index of the focusing adverb
            morphism_tokens: All tokens

        Returns:
            List of ScopeBinding objects with confidence scores
        """
        bindings = []
        adverb_token = morphism_tokens[adverb_idx].token
        head_idx = adverb_token.head_idx

        # Primary binding: immediate scope target
        if 0 <= head_idx < len(morphism_tokens):
            head = morphism_tokens[head_idx].token

            if head.pos == "VERB":
                # Check dependents of the verb for potential scope targets
                for mt in morphism_tokens:
                    if mt.token.head_idx == head_idx:
                        dep = mt.token.dep

                        # Subject scope
                        if dep == "nsubj":
                            confidence = 0.9 if adverb_idx == 0 else 0.5
                            bindings.append(ScopeBinding(
                                target=mt.token.text,
                                confidence=confidence,
                                position="subject",
                            ))

                        # Object scope
                        elif dep == "dobj":
                            confidence = 0.8 if adverb_idx > 0 else 0.4
                            bindings.append(ScopeBinding(
                                target=mt.token.text,
                                confidence=confidence,
                                position="object",
                            ))

                        # Adjunct scope
                        elif dep in ("prep", "advmod", "npadvmod"):
                            bindings.append(ScopeBinding(
                                target=mt.token.text,
                                confidence=0.3,
                                position="adjunct",
                            ))

                # Verb scope
                bindings.append(ScopeBinding(
                    target=head.text,
                    confidence=0.4,
                    position="verb",
                ))

        # Sort by confidence
        bindings.sort(key=lambda b: b.confidence, reverse=True)
        return bindings

    def _process_discourse_adverbs(
        self,
        morphism_tokens: list[MorphismToken]
    ) -> list[MorphismToken]:
        """Identify discourse adverbs and create inter-frame morphisms.

        Args:
            morphism_tokens: List of MorphismToken wrappers

        Returns:
            List of MorphismToken objects that are discourse adverbs
        """
        discourse = []

        # Build text spans for multi-word matching
        texts = [mt.token.text for mt in morphism_tokens]

        for i, mt in enumerate(morphism_tokens):
            token = mt.token
            text_lower = token.text.lower()
            lemma_lower = token.lemma.lower()

            # Check single-word markers
            marker_key = None
            if text_lower in _ALL_MARKERS:
                marker_key = text_lower
            elif lemma_lower in _ALL_MARKERS:
                marker_key = lemma_lower

            # Check multi-word markers starting at this position
            if marker_key is None:
                for j in range(min(4, len(morphism_tokens) - i), 0, -1):
                    phrase = " ".join(t.lower() for t in texts[i:i+j])
                    if phrase in _ALL_MARKERS:
                        marker_key = phrase
                        break

            if marker_key:
                relation, strength = _ALL_MARKERS[marker_key]
                mt.is_discourse_adverb = True
                mt.discourse_relation = relation

                # Create inter-frame morphism
                # Use sentence position to reference frames
                frame_before = f"FRAME_{i-1}" if i > 0 else "PREVIOUS"
                frame_after = f"FRAME_{i+1}" if i < len(morphism_tokens) - 1 else "FOLLOWING"

                mt.inter_frame_morphism = InterFrameMorphism(
                    source_frame=frame_before,
                    target_frame=frame_after,
                    edge_label=marker_key,
                    relation=relation,
                    strength=strength,
                )

                discourse.append(mt)

        return discourse


def process_tokens_phase3(tokens: list[Token]) -> Phase3Result:
    """Convenience function to process Phase 2 tokens through Phase 3.

    Args:
        tokens: List of Token objects from Phase 2

    Returns:
        Phase3Result with morphism annotations
    """
    processor = Phase3Processor()
    return processor.process(tokens)
