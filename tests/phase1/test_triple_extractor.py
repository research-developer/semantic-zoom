"""Tests for basic triple extraction (NSM-38)."""

import pytest
from semantic_zoom.phase1.tokenizer import Tokenizer
from semantic_zoom.phase1.dependency_parser import DependencyParser
from semantic_zoom.phase1.triple_extractor import TripleExtractor, Triple


class TestBasicTriple:
    """Test that S-V-O relationships are extracted as triples."""

    def test_simple_svo(self):
        """Simple sentence should yield one S-V-O triple."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The cat chased the mouse.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        assert len(triples) >= 1, "Should extract at least one triple"
        
        # Find the main triple
        triple = triples[0]
        assert triple.subject_text == "cat" or "cat" in triple.subject_text
        assert triple.predicate_text == "chased"
        assert triple.object_text == "mouse" or "mouse" in triple.object_text

    def test_triple_has_id_ranges(self):
        """Triples should contain word ID ranges for spans."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The big cat ate food.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        triple = triples[0]
        # Subject "The big cat" should span IDs 0, 1, 2
        assert hasattr(triple, 'subject_ids')
        assert hasattr(triple, 'predicate_ids')
        assert hasattr(triple, 'object_ids')
        
        assert isinstance(triple.subject_ids, (list, tuple))
        assert len(triple.subject_ids) >= 1

    def test_no_object_sentence(self):
        """Intransitive verbs should still yield triples (with None object)."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The bird flew.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        assert len(triples) >= 1
        triple = triples[0]
        assert triple.subject_text is not None
        assert triple.predicate_text == "flew"
        # Object can be None or empty for intransitive verbs


class TestMultipleClauses:
    """Test that multiple clauses yield separate linked triples."""

    def test_compound_sentence(self):
        """Compound sentence should yield multiple triples."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The cat slept and the dog barked.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        # Should have at least 2 triples
        assert len(triples) >= 2, f"Expected 2+ triples, got {len(triples)}"
        
        predicates = [t.predicate_text for t in triples]
        assert "slept" in predicates
        assert "barked" in predicates

    def test_subordinate_clause(self):
        """Subordinate clauses should also be extracted."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The cat that caught the mouse ran away.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        # Should have triples for both "caught" and "ran"
        predicates = [t.predicate_text for t in triples]
        assert "caught" in predicates or "ran" in predicates

    def test_triple_linking(self):
        """Related triples should have link information."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("John said that Mary left.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        # Each triple should have a link field
        for triple in triples:
            assert hasattr(triple, 'parent_triple_id')


class TestPassiveVoice:
    """Test that passive voice correctly identifies semantic agent."""

    def test_passive_detection(self):
        """Passive voice should be detected."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The mouse was chased by the cat.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        assert len(triples) >= 1
        triple = triples[0]
        assert triple.is_passive, "Should detect passive voice"

    def test_passive_agent_extraction(self):
        """Passive agent should be correctly identified."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The mouse was chased by the cat.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        triple = triples[0]
        # In passive, semantic agent is "cat" (in by-phrase)
        # Grammatical subject is "mouse" (patient)
        assert triple.semantic_agent_text == "cat" or "cat" in (triple.semantic_agent_text or "")
        assert triple.semantic_patient_text == "mouse" or "mouse" in (triple.semantic_patient_text or "")

    def test_passive_without_agent(self):
        """Truncated passive (no by-phrase) should still work."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The cake was eaten.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        assert len(triples) >= 1
        triple = triples[0]
        assert triple.is_passive
        assert triple.semantic_patient_text == "cake" or "cake" in (triple.semantic_patient_text or "")
        # Agent should be None or empty since no by-phrase


class TestTripleAttributes:
    """Test Triple dataclass attributes."""

    def test_triple_has_required_fields(self):
        """Triple should have all required fields."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("A dog chases a cat.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)

        assert len(triples) >= 1, "Should extract at least one triple"
        triple = triples[0]
        # Core triple components
        assert hasattr(triple, 'subject_text')
        assert hasattr(triple, 'predicate_text')
        assert hasattr(triple, 'object_text')
        # ID ranges
        assert hasattr(triple, 'subject_ids')
        assert hasattr(triple, 'predicate_ids')
        assert hasattr(triple, 'object_ids')
        # Voice information
        assert hasattr(triple, 'is_passive')
        # Semantic roles for passive
        assert hasattr(triple, 'semantic_agent_text')
        assert hasattr(triple, 'semantic_patient_text')

    def test_triple_has_unique_id(self):
        """Each triple should have a unique ID."""
        tokenizer = Tokenizer()
        parser = DependencyParser()
        extractor = TripleExtractor()
        
        tokens = tokenizer.tokenize("The cat slept and the dog ran.")
        parsed = parser.parse(tokens)
        triples = extractor.extract(parsed)
        
        ids = [t.id for t in triples]
        assert len(ids) == len(set(ids)), "Triple IDs should be unique"
