# Semantic Zoom

Quasi-deterministic semantic zoom via NLP triple decomposition with categorical morphisms.

## Overview

This project implements a novel approach to document navigation where users can "zoom" in and out of text semantically - collapsing away irrelevant detail while preserving the structural relationships that matter.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│  Text → Tokenize → POS Tag → Dependency Parse → Triple Extract  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  GRAMMATICAL CLASSIFICATION                      │
│  Person (1/2/3) · Tense · Adjective Order · Adverb Tiers        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     MORPHISM MAPPING                             │
│  Prepositions → Categorical Symbols · Focusing Adverbs · Discourse │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FRAME INTEGRATION                             │
│  FrameNet Assignment · Slot Filling · Plan vs Description       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   GRAPH CONSTRUCTION                             │
│  Nodes (Nouns) · Edges (Verbs) · Morphisms · Inter-frame Links  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ZOOM OPERATIONS                               │
│  Seed Selection · Subgraph Extraction · Sparse Display · Recovery │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  COMPILATION & LINTING                           │
│  Grammar Check · Ambiguity Detection · Clarification · Preservation │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
semantic-zoom/
├── src/
│   ├── ingestion/       # Phase 1: Tokenization, POS, parsing
│   ├── grammar/         # Phase 2: Classification systems
│   ├── morphisms/       # Phase 3: Categorical mappings
│   ├── frames/          # Phase 4: FrameNet integration
│   ├── graph/           # Phase 5: Graph construction
│   ├── zoom/            # Phase 6: Zoom operations
│   └── lint/            # Phase 7: Compilation & linting
├── tests/               # BDD test suites per phase
├── data/                # Sample texts, FrameNet data
└── docs/                # Additional documentation
```

## Development

This project uses git worktrees for parallel development of uncertain approaches:

```bash
# Create a worktree for experimental approach
git worktree add ../semantic-zoom-experiment-name branch-name

# List active worktrees
git worktree list

# Remove when done
git worktree remove ../semantic-zoom-experiment-name
```

## Issues

- **Linear Project:** https://linear.app/imajn/project/semantic-zoom-2bebe946c9e9
- **GitHub Issues:** https://github.com/research-developer/semantic-zoom/issues

## Phases

| Phase | Issues | Status | Dependencies |
|-------|--------|--------|--------------|
| 1. Ingestion | NSM-35→38 | 🟡 In Progress | None |
| 2. Grammar | NSM-39→42 | 🟡 In Progress | Partial on Phase 1 |
| 3. Morphisms | NSM-43→45 | ⚪ Pending | Phase 2 |
| 4. Frames | NSM-46→48 | ⚪ Pending | Phase 2 |
| 5. Graph | NSM-49→52 | ⚪ Pending | Phases 3, 4 |
| 6. Zoom | NSM-53→56 | ⚪ Pending | Phase 5 |
| 7. Lint | NSM-57→60 | ⚪ Pending | Phase 5 |
