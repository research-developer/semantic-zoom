# Phase 2 Linguistic Correctness Audit

**Date:** 2025-12-23
**Auditor:** Phase 2 Lead Agent with NLP Engineer skill
**Scope:** NSM-39 through NSM-42 (Grammatical Classification)

---

## Executive Summary

The Phase 2 implementation demonstrates **solid linguistic foundations** with a few areas for improvement. The code correctly implements core grammatical classification using established frameworks (Penn Treebank tags, Dixon's adjective ordering, Cinque's adverb hierarchy).

**Overall Assessment:** 85/100 - Production-ready with minor refinements recommended.

---

## Module-by-Module Analysis

### 1. Noun Person Classification (NSM-39) - **Score: 90/100**

**Linguistic Framework:** Standard person deixis (Benveniste, 1966)

#### Strengths:
- Correct enumeration of 1st/2nd/3rd person pronouns
- Proper handling of reflexives (myself, yourself, themselves)
- Includes possessives (my/mine, your/yours, their/theirs)
- Handles relative pronouns (who, whom, which) as 3rd person
- Generic "one" correctly classified as grammatically 3rd person

#### Linguistic Accuracy:
| Feature | Status | Notes |
|---------|--------|-------|
| Personal pronouns | Correct | Full paradigm covered |
| Possessives | Correct | Both adjective and pronoun forms |
| Reflexives | Correct | All singular/plural forms |
| Relative pronouns | Correct | who/whom/which/that |
| Proper nouns | Correct | Always 3rd person |
| Generic markers | Correct | one/people/someone/etc. |

#### Issues Identified:
1. **Minor:** "one" appears in both `THIRD_PERSON_LEMMAS` and `GENERIC_MARKERS` - intentional overlap but could be clarified
2. **Edge case:** "we" used editorially ("the royal we") is still 1st person grammatically - correct behavior
3. **Missing:** Archaic forms (thou/thee/thy) - acceptable omission for modern English

### 2. Verb Tense/Aspect Classification (NSM-40) - **Score: 82/100**

**Linguistic Framework:** Reichenbachian tense semantics + traditional aspect classification

#### Strengths:
- Correct Penn Treebank tag mapping (VBD, VBZ, VBP, VB, VBG, VBN)
- Proper compound tense analysis with auxiliary chains
- Handles progressive (be + VBG) and perfect (have + VBN)
- Correct future tense with will/shall

#### Linguistic Accuracy:
| Feature | Status | Notes |
|---------|--------|-------|
| Simple tenses | Correct | Past/present from tags |
| Progressive aspect | Correct | be + -ing |
| Perfect aspect | Correct | have + past participle |
| Perfect progressive | Correct | have been + -ing |
| Future | Correct | will/shall + base |
| Modals | Correct | MD tag excluded from tense |

#### Issues Identified:
1. **Medium:** Auxiliary chain detection uses position-based heuristic (`range(main_verb_idx)`) rather than dependency traversal - may miss discontinuous auxiliaries
2. **Medium:** No handling of "going to" future (gonna) - common in spoken English
3. **Minor:** `'ll` contraction handling is good but `won't` (will not) needs verification
4. **Minor:** Subjunctive mood not distinguished from indicative

#### Recommendations:
- Add "going to" -> FUTURE handling
- Consider dependency-based auxiliary detection for robustness

### 3. Adjective Ordering (NSM-41) - **Score: 88/100**

**Linguistic Framework:** Dixon (1982) / Quirk et al. (1985) adjective ordering

#### Strengths:
- Correct canonical order: opinion > size > age > shape > color > origin > material > purpose
- Comprehensive lexicons for each category
- Morphological heuristics for unknowns (-en endings, capitalization)
- Chain extraction via dependency parsing

#### Linguistic Accuracy:
| Category | Lexicon Coverage | Status |
|----------|------------------|--------|
| Opinion | 28 entries | Good |
| Size | 25 entries | Good |
| Age | 17 entries | Adequate |
| Shape | 17 entries | Adequate |
| Color | 40+ entries | Excellent |
| Origin | 26 entries | Good |
| Material | 33 entries | Excellent |
| Purpose | 20 entries | Adequate |

#### Issues Identified:
1. **Minor:** Some adjectives can span categories (e.g., "dark" could be color or evaluative) - current approach defaults to first match
2. **Minor:** Compound adjectives not explicitly handled (e.g., "light-blue")
3. **Theory note:** Dixon's 8 categories are a simplification; some linguists use 10+ categories

#### Morphological Heuristics Accuracy:
- `-en` ending -> MATERIAL: **Correct** (wooden, golden, silken)
- VBG + amod -> PURPOSE: **Correct** (sleeping bag, running shoes)
- Capitalized ADJ -> ORIGIN: **Correct** (French wine, American car)

### 4. Adverb Tier Assignment (NSM-42) - **Score: 80/100**

**Linguistic Framework:** Cinque (1999) adverb hierarchy + Ernst (2002)

#### Strengths:
- Seven semantic tiers correctly enumerated
- Degree adverb attachment tracking
- Sentence-initial position handling for sentence adverbs
- Good lexicon coverage

#### Linguistic Accuracy:
| Tier | Lexicon Size | Cinque Alignment |
|------|--------------|------------------|
| MANNER | 28 | Correct |
| PLACE | 24 | Correct |
| FREQUENCY | 22 | Correct |
| TIME | 26 | Correct |
| PURPOSE | 7 | Correct |
| SENTENCE | 26 | Correct |
| DEGREE | 30 | Correct |

#### Issues Identified:
1. **Medium:** Cinque's hierarchy is more fine-grained than implemented (his 30+ positions vs. our 7 tiers) - acceptable simplification
2. **Medium:** "so" listed in PURPOSE but is frequently DEGREE ("so big")
3. **Minor:** Some adverbs change tier by context (e.g., "still" = TIME vs. "still" = concessive)
4. **Minor:** `-ly` heuristic is good but can misclassify ("lovely" is ADJ, not ADV)

#### Recommendations:
- Add context-aware disambiguation for polysemous adverbs
- Consider splitting PURPOSE into causal/result subtypes

---

## Cross-Module Integration

### Token Flow Verification:
```
spaCy tokenization → Phase 1 ParsedToken → Phase 2 Token (with classifications)
                                                ↓
                                    Phase 3 MorphismToken (with morphism mappings)
```

**Status:** Integration layer in `pipeline.py` correctly:
- Extracts lemmas (missing from Phase 1)
- Maps field names (id → idx, head_id → head_idx)
- Runs all four classifiers in sequence

### Classification Interactions:
1. **Person + Generic:** Work together correctly for "People say..." constructs
2. **Tense + Aspect:** VerbCompound properly combines auxiliary analysis
3. **Adj Slot + Chains:** Normalization preserves original positions
4. **Adv Tier + Attachment:** Degree adverbs correctly find their targets

---

## Test Coverage Analysis

| Module | Tests | Edge Cases | Linguistic Coverage |
|--------|-------|------------|---------------------|
| noun_person | 20+ | Good | Excellent |
| verb_tense | 25+ | Good | Good |
| adjective_order | 30+ | Good | Good |
| adverb_tier | 25+ | Adequate | Good |

**Total Tests:** 100+ across Phase 2 modules

---

## Recommendations Summary

### High Priority (for v1.0):
1. Add "going to" future tense handling
2. Add context-aware "so" disambiguation (degree vs. purpose)

### Medium Priority (for v1.1):
1. Use dependency-based auxiliary chain detection
2. Add subjunctive mood detection
3. Handle compound adjectives

### Low Priority (future):
1. Extend Cinque hierarchy to 10+ tiers
2. Add archaic pronoun forms
3. Add aspect for stative vs. dynamic verbs

---

## Phase 6 Copula Issue (Cross-Phase)

### Issue Identified
During audit, discovered that copular "is/be" verbs were being overly targeted as seed words in Phase 6 subgraph extraction. Root cause:

1. In copular sentences ("The cat is on the mat"), spaCy correctly identifies "is" as ROOT
2. The `_find_clause_root` method returns "is" as the clause root
3. N-hop expansion from any content word always pulls in "is" (it's the syntactic head)

### Linguistic Analysis
Copular constructions have a syntactic/semantic mismatch:
- **Syntactically:** "is" is the main verb (ROOT)
- **Semantically:** "is" is a linking verb with minimal semantic content

| Sentence Type | Example | Syntactic ROOT | Semantic Focus |
|--------------|---------|----------------|----------------|
| Predicate Adjective | "The cat is happy" | is | happy |
| Predicate Nominal | "John is a doctor" | is | doctor |
| Predicate Locative | "The book is on the shelf" | is | on the shelf |
| Progressive | "She was running" | running | running |

### Fix Implemented
Added `skip_copulas` parameter to `SubgraphExtractor`:

```python
# Option 1: Instance-level setting
extractor = SubgraphExtractor(skip_copulas=True)

# Option 2: Per-call override
extractor.extract(tokens, seeds, zoom_level=1, skip_copulas=True)
```

**Behavior:**
- Copulas (AUX + ROOT + form of "be") are treated as "transparent"
- Expansion passes through copulas to reach content words
- Copulas are excluded from final result
- Auxiliaries in progressives ("was running") are preserved

### Test Coverage
Added 6 tests in `TestCopulaSkipping`:
- Default backward compatibility
- Copula exclusion in predicate constructions
- Sibling connectivity through transparent copulas
- Auxiliary preservation in progressive aspect
- Per-call override functionality
- Predicate nominal handling

---

## Conclusion

Phase 2 implementation is **linguistically sound** and follows established grammatical frameworks correctly. The code demonstrates careful attention to:
- Standard linguistic terminology
- Comprehensive lexicon coverage
- Appropriate fallback heuristics
- Proper handling of edge cases

The identified issues are refinements rather than fundamental errors. The implementation is ready for Phase 3 integration.

**Recommendation:** Proceed with Phase 3 integration. Address high-priority items in maintenance cycle.
