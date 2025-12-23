"""Integration pipeline connecting Phase 1 → Phase 2.

This module provides the adapter layer that:
1. Takes Phase 1's ParsedToken output
2. Extracts lemmas (missing from Phase 1)
3. Converts to Phase 2's Token format
4. Runs Phase 2 grammatical classification

Usage:
    from semantic_zoom.pipeline import Pipeline

    pipeline = Pipeline()
    tokens = pipeline.process("The quick brown fox jumps.")
"""

from typing import List
import spacy

from semantic_zoom.models import Token
from semantic_zoom.phase1.tokenizer import Tokenizer, Token as Phase1Token
from semantic_zoom.phase1.pos_tagger import POSTagger, TaggedToken
from semantic_zoom.phase1.dependency_parser import DependencyParser, ParsedToken
from semantic_zoom.phase2 import (
    classify_tokens_person,
    classify_token_tense,
    classify_tokens_adjectives,
    classify_tokens_adverbs,
)


class Pipeline:
    """End-to-end NLP pipeline from text to classified tokens.

    Integrates Phase 1 (tokenization, POS, dependencies) with
    Phase 2 (grammatical classification).
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize pipeline with spaCy model.

        Args:
            model: spaCy model name
        """
        # Phase 1 components
        self._tokenizer = Tokenizer(model)
        self._pos_tagger = POSTagger(model)
        self._dep_parser = DependencyParser(model)

        # Phase 2 results
        self._adjective_chains = []

        # Shared spaCy instance for lemma extraction
        try:
            self._nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            self._nlp = spacy.load(model)

    def process(self, text: str) -> List[Token]:
        """Process text through full pipeline.

        Args:
            text: Input text to process

        Returns:
            List of Token objects with Phase 2 classifications
        """
        if not text or text.isspace():
            return []

        # Phase 1: Tokenize, POS tag, parse dependencies
        phase1_tokens = self._tokenizer.tokenize(text)
        tagged_tokens = self._pos_tagger.tag(phase1_tokens)
        parsed_tokens = self._dep_parser.parse(phase1_tokens)

        # Convert to Phase 2 Token format with lemmas
        tokens = self._convert_with_lemmas(text, parsed_tokens)

        # Phase 2: Grammatical classification
        tokens = self._run_phase2(tokens)

        return tokens

    def _convert_with_lemmas(
        self,
        text: str,
        parsed_tokens: List[ParsedToken]
    ) -> List[Token]:
        """Convert Phase 1 output to Phase 2 Token format with lemmas.

        Phase 1's ParsedToken lacks lemma extraction. We run spaCy again
        to get lemmas, then merge with Phase 1 output.

        Args:
            text: Original text (for lemma extraction)
            parsed_tokens: Phase 1 parsed tokens

        Returns:
            List of Phase 2 Token objects
        """
        if not parsed_tokens:
            return []

        # Get lemmas from spaCy
        doc = self._nlp(text)

        tokens = []
        for pt, spacy_token in zip(parsed_tokens, doc):
            token = Token(
                text=pt.text,
                lemma=spacy_token.lemma_,
                idx=pt.id,  # Phase 1's 'id' → Phase 2's 'idx'
                pos=pt.pos,
                tag=pt.tag,
                dep=pt.dep,
                head_idx=pt.head_id,  # Phase 1's 'head_id' → Phase 2's 'head_idx'
            )
            tokens.append(token)

        return tokens

    def _run_phase2(self, tokens: List[Token]) -> List[Token]:
        """Run Phase 2 grammatical classification.

        Args:
            tokens: List of Token objects

        Returns:
            Same tokens with Phase 2 fields populated
        """
        # NSM-39: Noun person classification
        classify_tokens_person(tokens)

        # NSM-40: Verb tense/aspect classification
        for token in tokens:
            if token.pos == "VERB" or token.pos == "AUX":
                classify_token_tense(token, tokens)

        # NSM-41: Adjective ordering (modifies tokens in place, returns chains)
        self._adjective_chains = classify_tokens_adjectives(tokens)

        # NSM-42: Adverb tier assignment
        classify_tokens_adverbs(tokens)

        return tokens

    def process_phase1_only(self, text: str) -> List[ParsedToken]:
        """Run Phase 1 only (for debugging/testing).

        Args:
            text: Input text

        Returns:
            List of ParsedToken objects
        """
        phase1_tokens = self._tokenizer.tokenize(text)
        return self._dep_parser.parse(phase1_tokens)

    def convert_parsed_tokens(
        self,
        parsed_tokens: List[ParsedToken],
        text: str = None
    ) -> List[Token]:
        """Convert existing Phase 1 output to Phase 2 format.

        Useful when Phase 1 has already been run separately.

        Args:
            parsed_tokens: Phase 1 parsed tokens
            text: Original text (required for lemma extraction)

        Returns:
            List of Phase 2 Token objects (without classification)
        """
        if not parsed_tokens:
            return []

        if text is None:
            # Reconstruct text from tokens
            text = "".join(pt.text + pt.whitespace_after for pt in parsed_tokens)

        return self._convert_with_lemmas(text, parsed_tokens)


def adapt_phase1_to_phase2(
    parsed_tokens: List[ParsedToken],
    text: str = None,
    model: str = "en_core_web_sm"
) -> List[Token]:
    """Standalone adapter function for Phase 1 → Phase 2 conversion.

    Use this when you have Phase 1 output and want Phase 2 Token format.

    Args:
        parsed_tokens: Phase 1 parsed tokens
        text: Original text (required for lemma extraction)
        model: spaCy model for lemma extraction

    Returns:
        List of Phase 2 Token objects
    """
    if not parsed_tokens:
        return []

    if text is None:
        text = "".join(pt.text + pt.whitespace_after for pt in parsed_tokens)

    try:
        nlp = spacy.load(model)
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", model], check=True)
        nlp = spacy.load(model)

    doc = nlp(text)

    tokens = []
    for pt, spacy_token in zip(parsed_tokens, doc):
        token = Token(
            text=pt.text,
            lemma=spacy_token.lemma_,
            idx=pt.id,
            pos=pt.pos,
            tag=pt.tag,
            dep=pt.dep,
            head_idx=pt.head_id,
        )
        tokens.append(token)

    return tokens
