#!/bin/bash
# Batch create GitHub issues for Semantic Zoom project

# Phase 1 issues (NSM-36 to NSM-38)
gh issue create --title 'Part-of-speech tagging' --label 'phase-1' --body 'Parent: https://linear.app/imajn/issue/NSM-36

## Acceptance Criteria
- Each token annotated with grammatical category
- Ambiguous words get context-appropriate tag with confidence
- Queryable by category with word IDs

## Test: `test_pos_tagging`'

gh issue create --title 'Dependency parsing' --label 'phase-1' --body 'Parent: https://linear.app/imajn/issue/NSM-37

## Acceptance Criteria
- Each token has head pointer and dependency relation label
- Root (main predicate) queryable
- Noun dependents (modifiers) retrievable with word IDs

## Test: `test_dependency_parse`'

gh issue create --title 'Basic triple extraction (subject-verb-object)' --label 'phase-1' --body 'Parent: https://linear.app/imajn/issue/NSM-38

## Acceptance Criteria
- All S-V-O relationships extracted as triples
- Triples contain word ID ranges for spans
- Multiple clauses yield separate linked triples
- Passive voice correctly identifies semantic agent

## Test: `test_basic_triple`, `test_passive`'

# Phase 2 issues (NSM-39 to NSM-42)
gh issue create --title 'Noun person classification (1st/2nd/3rd)' --label 'phase-2' --body 'Parent: https://linear.app/imajn/issue/NSM-39

## Acceptance Criteria
- Nouns/pronouns tagged: FIRST, SECOND, THIRD, or NONE
- Generic constructions marked with generic=True flag

## Test: `test_person_classification`'

gh issue create --title 'Verb tense classification (past/present/future)' --label 'phase-2' --body 'Parent: https://linear.app/imajn/issue/NSM-40

## Acceptance Criteria
- Verbs tagged: PAST, PRESENT, FUTURE, or INFINITIVE
- Compound tenses identified from auxiliaries
- Aspect captured: SIMPLE, PROGRESSIVE, PERFECT, PERFECT_PROGRESSIVE

## Test: `test_tense_classification`, `test_aspect`'

gh issue create --title 'Adjective ordering extraction and normalization' --label 'phase-2' --body 'Parent: https://linear.app/imajn/issue/NSM-41

## Acceptance Criteria
- Adjectives ordered: opinion→size→age→shape→color→origin→material→purpose
- Non-canonical order normalized with original positions preserved
- Each adjective classified to one slot

## Test: `test_adjective_ordering`, `test_normalization`'

gh issue create --title 'Adverb tier assignment' --label 'phase-2' --body 'Parent: https://linear.app/imajn/issue/NSM-42

## Acceptance Criteria
- Adverbs assigned: MANNER, PLACE, FREQUENCY, TIME, PURPOSE
- Sentence-level adverbs marked as SENTENCE tier
- Degree adverbs marked with attachment to modified word
- Canonical order: Manner→Place→Frequency→Time→Purpose

## Test: `test_adverb_tiers`'

# Phase 3 issues (NSM-43 to NSM-45)
gh issue create --title 'Preposition → categorical symbol mapping with state' --label 'phase-3' --body 'Parent: https://linear.app/imajn/issue/NSM-43

## Acceptance Criteria
- ~60 prepositions → ~15-20 symbol types with state flags
- Symbol types: DIRECTIONAL (◃▹), CONTAINMENT (∈∉), SPATIAL (⊥), ACCOMPANIMENT (⊕⊖), TEMPORAL (◁▷)
- Dual-citizenship resolved via saturated flag

## Test: `test_preposition_mapping`, `test_dual_citizenship`'

gh issue create --title 'Focusing adverb identification and scope marking' --label 'phase-3' --body 'Parent: https://linear.app/imajn/issue/NSM-44

## Acceptance Criteria
- Focusing adverbs (only, even, just, etc.) marked as SCOPE_OPERATOR
- Scope target identified with possible scope bindings enumerated
- Marked as invertible=False (non-reversible operation)

## Test: `test_scope_identification`'

gh issue create --title 'Discourse adverb → inter-frame relation mapping' --label 'phase-3' --body 'Parent: https://linear.app/imajn/issue/NSM-45

## Acceptance Criteria
- Discourse adverbs marked as INTER_FRAME_MORPHISM
- Relation types: CONTRAST, CONSEQUENCE, ADDITION, CONCESSION, SEQUENCE, EXEMPLIFICATION
- Frames as nodes, discourse adverbs as edges

## Test: `test_discourse_adverb`'

# Phase 4 issues (NSM-46 to NSM-48)
gh issue create --title 'FrameNet frame assignment to verb predicates' --label 'phase-4' --body 'Parent: https://linear.app/imajn/issue/NSM-46

## Acceptance Criteria
- Verbs mapped to candidate FrameNet frames
- Polysemous verbs disambiguated with confidence
- Frame lexical unit and elements available

## Test: `test_frame_assignment`, `test_polysemy`'

gh issue create --title 'Frame element slot filling' --label 'phase-4' --body 'Parent: https://linear.app/imajn/issue/NSM-47

## Acceptance Criteria
- Arguments mapped to frame element slots (Agent, Theme, Goal, etc.)
- Optional elements marked UNFILLED when absent
- Implicit arguments flagged with implicit=True

## Test: `test_slot_filling`, `test_implicit`'

gh issue create --title 'Plan vs Description classification' --label 'phase-4' --body 'Parent: https://linear.app/imajn/issue/NSM-48

## Acceptance Criteria
- Propositions tagged: PLAN, DESCRIPTION, or HYBRID
- Dynamic frames → PLAN, Stative frames → DESCRIPTION
- Aspectual transformations may shift to HYBRID

## Test: `test_plan_classification`, `test_description_classification`, `test_hybrid`'

# Phase 5 issues (NSM-49 to NSM-52)
gh issue create --title 'Node creation (nouns with attributes)' --label 'phase-5' --body 'Parent: https://linear.app/imajn/issue/NSM-49

## Acceptance Criteria
- Noun phrases create nodes with word ID span and attributes
- Adjective vector attached to nodes
- Pronouns marked with antecedent link if resolved
- Proper nouns marked with entity type

## Test: `test_node_creation`'

gh issue create --title 'Edge creation (verbs with modifiers)' --label 'phase-5' --body 'Parent: https://linear.app/imajn/issue/NSM-50

## Acceptance Criteria
- Verb predicates create edges connecting subject to object nodes
- Intransitive verbs connect to NULL/implicit object
- Tiered adverb stack attached to edges

## Test: `test_edge_creation`'

gh issue create --title 'Morphism attachment (prepositions, adverbs)' --label 'phase-5' --body 'Parent: https://linear.app/imajn/issue/NSM-51

## Acceptance Criteria
- Preposition symbols attached at NODE, EDGE, FRAME, or PROPOSITION level
- Focusing adverb scope operators attached with invertible=False
- Adverbs attached at appropriate tier level

## Test: `test_prep_morphism`, `test_focusing_adverb`'

gh issue create --title 'Inter-frame linking' --label 'phase-5' --body 'Parent: https://linear.app/imajn/issue/NSM-52

## Acceptance Criteria
- Discourse adverbs create edges between frame nodes
- Implicit relations detected with lower confidence
- FrameNet frame-to-frame relations represented

## Test: `test_explicit_link`, `test_implicit_link`'

# Phase 6 issues (NSM-53 to NSM-56)
gh issue create --title 'Seed statement selection UI' --label 'phase-6' --body 'Parent: https://linear.app/imajn/issue/NSM-53

## Acceptance Criteria
- Text selection captures word ID range as seed
- Containing sentence/clause identified
- Multiple seeds storable, corresponding graph nodes highlighted

## Test: `test_seed_selection`, `test_multi_seed`'

gh issue create --title 'Subgraph extraction algorithm' --label 'phase-6' --body 'Parent: https://linear.app/imajn/issue/NSM-54

## Acceptance Criteria
- Seeds + zoom level → connected subgraph of related nodes/edges
- Zoom level 1: directly connected only
- Zoom level N: N-hop neighborhood via embedding similarity
- Result: list of word IDs covering subgraph
- Quasi-deterministic: same seeds + level → same subgraph

## Test: `test_subgraph_extraction`, `test_determinism`'

gh issue create --title 'Sparse display rendering' --label 'phase-6' --body 'Parent: https://linear.app/imajn/issue/NSM-55

## Acceptance Criteria
- Subgraph word IDs displayed with visual gaps for omissions
- Placeholders indicate omitted content ([...] or ···)
- Pronouns preserved to maintain referential structure
- Rendering density adjusts with zoom level

## Test: `test_sparse_render`, `test_pronoun_preservation`'

gh issue create --title 'Expansion/recovery mechanics' --label 'phase-6' --body 'Parent: https://linear.app/imajn/issue/NSM-56

## Acceptance Criteria
- Gap expansion retrieves missing word IDs
- Full expansion recovers exact original text
- Partial expansion: only selected region expands
- Granularity controllable: word, phrase, clause, sentence

## Test: `test_full_recovery`, `test_partial_expansion`'

# Phase 7 issues (NSM-57 to NSM-60)
gh issue create --title 'Grammar check integration' --label 'phase-7' --body 'Parent: https://linear.app/imajn/issue/NSM-57

## Acceptance Criteria
- Grammatical errors identified with type and location
- Severity: ERROR, WARNING, INFO
- Suggestions provided where possible
- Both original and corrected versions preserved

## Test: `test_grammar_check`, `test_suggestion`'

gh issue create --title 'Ambiguity detection' --label 'phase-7' --body 'Parent: https://linear.app/imajn/issue/NSM-58

## Acceptance Criteria
- Structural ambiguities identified (PP-attachment, coordination, pronoun, quantifier, negation scope)
- Multiple parse interpretations enumerated with confidence
- Possible antecedents listed for pronoun ambiguity

## Test: `test_pp_attachment`, `test_pronoun_ambiguity`'

gh issue create --title 'User prompt system for clarification' --label 'phase-7' --body 'Parent: https://linear.app/imajn/issue/NSM-59

## Acceptance Criteria
- Grammar errors prompt: accept suggestion, provide alternative, or ignore
- Ambiguities prompt: select intended interpretation
- User response stored and applied to parse
- Batch resolution supported

## Test: `test_correction_prompt`, `test_ambiguity_prompt`'

gh issue create --title 'Original preservation alongside corrections' --label 'phase-7' --body 'Parent: https://linear.app/imajn/issue/NSM-60

## Acceptance Criteria
- Both original and corrected versions stored with word ID mappings
- Original view always recoverable
- Edits changing word count tracked and reversible
- Full version history maintained (git-like DAG)

## Test: `test_preservation`, `test_revert`'

echo "All issues created!"
