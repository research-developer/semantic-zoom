"""Basic triple extraction - subject-verb-object (NSM-38).

This module provides triple extraction that:
- Extracts all S-V-O relationships as triples
- Includes word ID ranges for spans
- Handles multiple clauses with linked triples
- Correctly handles passive voice semantic roles
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import spacy

from semantic_zoom.phase1.dependency_parser import ParsedToken


@dataclass
class Triple:
    """A subject-verb-object triple with span information.
    
    Attributes:
        id: Unique identifier for this triple
        subject_text: Text of the subject span
        predicate_text: Text of the predicate (verb)
        object_text: Text of the object span (None for intransitive)
        subject_ids: Token IDs comprising the subject span
        predicate_ids: Token IDs comprising the predicate span
        object_ids: Token IDs comprising the object span
        is_passive: Whether the sentence is in passive voice
        semantic_agent_text: The semantic agent (doer of action)
        semantic_patient_text: The semantic patient (receiver of action)
        parent_triple_id: ID of parent triple for subordinate clauses
    """
    id: int
    subject_text: str
    predicate_text: str
    object_text: Optional[str]
    subject_ids: List[int]
    predicate_ids: List[int]
    object_ids: List[int]
    is_passive: bool = False
    semantic_agent_text: Optional[str] = None
    semantic_patient_text: Optional[str] = None
    parent_triple_id: Optional[int] = None


class TripleExtractor:
    """Extracts subject-verb-object triples from parsed sentences.
    
    Handles:
    - Simple SVO sentences
    - Compound sentences with multiple clauses
    - Passive voice with correct semantic role assignment
    - Subordinate clauses
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize triple extractor.
        
        Args:
            model: spaCy model name (default: en_core_web_sm)
        """
        try:
            self._nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            self._nlp = spacy.load(model)
        
        self._triple_counter = 0
    
    def extract(self, parsed_tokens: List[ParsedToken]) -> List[Triple]:
        """Extract all S-V-O triples from parsed tokens.
        
        Args:
            parsed_tokens: List of ParsedToken objects with dependency info
            
        Returns:
            List of Triple objects
        """
        if not parsed_tokens:
            return []
        
        self._triple_counter = 0
        triples = []
        
        # Reconstruct text for additional spaCy analysis if needed
        text = "".join(t.text + t.whitespace_after for t in parsed_tokens)
        doc = self._nlp(text)
        
        # Find all verbs (potential predicates)
        # Skip auxiliary verbs that are dependents of main verbs
        aux_deps = {"aux", "auxpass"}
        verbs = [
            t for t in parsed_tokens
            if t.pos == "VERB" and t.dep not in aux_deps
        ]

        for verb in verbs:
            triple = self._extract_triple_for_verb(parsed_tokens, verb, doc)
            if triple:
                triples.append(triple)
        
        # Link subordinate clauses
        self._link_triples(triples, parsed_tokens)
        
        return triples
    
    def _extract_triple_for_verb(
        self, 
        tokens: List[ParsedToken], 
        verb: ParsedToken,
        doc
    ) -> Optional[Triple]:
        """Extract a triple for a specific verb.
        
        Args:
            tokens: All parsed tokens
            verb: The verb token to extract triple for
            doc: spaCy Doc for additional analysis
            
        Returns:
            Triple object or None if not a valid triple
        """
        # Check if this is passive voice
        is_passive = self._is_passive(tokens, verb)
        
        # Find subject
        subject = self._find_subject(tokens, verb)
        
        # Find object
        obj = self._find_object(tokens, verb)
        
        # For passive, also find the agent (in by-phrase)
        agent = None
        if is_passive:
            agent = self._find_passive_agent(tokens, verb)
        
        # Skip if no subject found (incomplete clause)
        if subject is None:
            # But still include if we have at least a verb
            pass
        
        # Get span IDs
        subject_ids, subject_text = self._get_span(tokens, subject)
        predicate_ids = [verb.id]
        predicate_text = verb.text
        object_ids, object_text = self._get_span(tokens, obj)
        
        # Handle auxiliary verbs
        aux_verbs = [t for t in tokens if t.head_id == verb.id and t.dep == "aux"]
        for aux in aux_verbs:
            predicate_ids.append(aux.id)
        predicate_ids.sort()
        
        # Determine semantic roles
        if is_passive:
            semantic_agent_text = None
            if agent:
                _, semantic_agent_text = self._get_span(tokens, agent)
            semantic_patient_text = subject_text
        else:
            semantic_agent_text = subject_text
            semantic_patient_text = object_text
        
        triple = Triple(
            id=self._triple_counter,
            subject_text=subject_text or "",
            predicate_text=predicate_text,
            object_text=object_text,
            subject_ids=subject_ids,
            predicate_ids=predicate_ids,
            object_ids=object_ids,
            is_passive=is_passive,
            semantic_agent_text=semantic_agent_text,
            semantic_patient_text=semantic_patient_text,
            parent_triple_id=None,
        )
        
        self._triple_counter += 1
        return triple
    
    def _is_passive(self, tokens: List[ParsedToken], verb: ParsedToken) -> bool:
        """Check if a verb is in passive voice.
        
        Args:
            tokens: All parsed tokens
            verb: The verb token to check
            
        Returns:
            True if passive voice
        """
        # Look for passive auxiliary ("was", "were", "been", etc.)
        for t in tokens:
            if t.head_id == verb.id and t.dep == "auxpass":
                return True
        
        # Also check for nsubjpass dependency
        for t in tokens:
            if t.head_id == verb.id and t.dep == "nsubjpass":
                return True
        
        return False
    
    def _find_subject(self, tokens: List[ParsedToken], verb: ParsedToken) -> Optional[ParsedToken]:
        """Find the subject of a verb.
        
        Args:
            tokens: All parsed tokens
            verb: The verb token
            
        Returns:
            Subject token or None
        """
        subject_deps = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
        for t in tokens:
            if t.head_id == verb.id and t.dep in subject_deps:
                return t
        return None
    
    def _find_object(self, tokens: List[ParsedToken], verb: ParsedToken) -> Optional[ParsedToken]:
        """Find the direct object of a verb.
        
        Args:
            tokens: All parsed tokens
            verb: The verb token
            
        Returns:
            Object token or None
        """
        object_deps = {"dobj", "obj", "attr", "oprd"}
        for t in tokens:
            if t.head_id == verb.id and t.dep in object_deps:
                return t
        return None
    
    def _find_passive_agent(self, tokens: List[ParsedToken], verb: ParsedToken) -> Optional[ParsedToken]:
        """Find the agent in a passive construction (in by-phrase).
        
        Args:
            tokens: All parsed tokens
            verb: The verb token
            
        Returns:
            Agent token or None
        """
        # Look for "by" preposition dependent on verb, then get its object
        for t in tokens:
            if t.head_id == verb.id and t.text.lower() == "by" and t.dep == "agent":
                # Find the object of "by"
                for t2 in tokens:
                    if t2.head_id == t.id and t2.dep == "pobj":
                        return t2
        
        # Alternative: look for agent dependency directly
        for t in tokens:
            if t.head_id == verb.id and t.dep == "agent":
                # The agent token might have a pobj child
                for t2 in tokens:
                    if t2.head_id == t.id and t2.dep == "pobj":
                        return t2
        
        return None
    
    def _get_span(
        self, 
        tokens: List[ParsedToken], 
        head: Optional[ParsedToken]
    ) -> Tuple[List[int], Optional[str]]:
        """Get the full span (with dependents) for a head token.
        
        Args:
            tokens: All parsed tokens
            head: Head token of the span
            
        Returns:
            Tuple of (list of IDs, span text)
        """
        if head is None:
            return [], None
        
        # Get all tokens in the subtree
        span_ids = self._get_subtree_ids(tokens, head.id)
        span_ids.sort()
        
        # Build text from span
        span_tokens = [t for t in tokens if t.id in span_ids]
        span_tokens.sort(key=lambda t: t.id)
        
        text_parts = []
        for i, t in enumerate(span_tokens):
            text_parts.append(t.text)
            # Add whitespace if not last token and there's whitespace after
            if i < len(span_tokens) - 1 and t.whitespace_after:
                text_parts.append(t.whitespace_after)
        
        return span_ids, "".join(text_parts).strip()
    
    def _get_subtree_ids(self, tokens: List[ParsedToken], root_id: int) -> List[int]:
        """Get all token IDs in a subtree.
        
        Args:
            tokens: All parsed tokens
            root_id: ID of subtree root
            
        Returns:
            List of token IDs in the subtree
        """
        ids = [root_id]
        for t in tokens:
            if t.head_id == root_id and t.id != root_id:
                ids.extend(self._get_subtree_ids(tokens, t.id))
        return ids
    
    def _link_triples(self, triples: List[Triple], tokens: List[ParsedToken]) -> None:
        """Link subordinate clause triples to their parent triples.
        
        Args:
            triples: List of extracted triples (modified in place)
            tokens: All parsed tokens
        """
        # Build a mapping from verb ID to triple
        verb_to_triple = {}
        for triple in triples:
            for pred_id in triple.predicate_ids:
                verb_to_triple[pred_id] = triple
        
        # Check each triple for subordination
        for triple in triples:
            main_pred_id = triple.predicate_ids[0] if triple.predicate_ids else None
            if main_pred_id is None:
                continue
            
            pred_token = next((t for t in tokens if t.id == main_pred_id), None)
            if pred_token is None:
                continue
            
            # If this verb's head is another verb, link to that triple
            head_id = pred_token.head_id
            if head_id >= 0:
                head_token = next((t for t in tokens if t.id == head_id), None)
                if head_token and head_token.pos == "VERB" and head_id in verb_to_triple:
                    parent_triple = verb_to_triple[head_id]
                    if parent_triple.id != triple.id:
                        triple.parent_triple_id = parent_triple.id
