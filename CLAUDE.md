# Semantic Zoom - Claude Agent Instructions

## Project Overview
Quasi-deterministic semantic zoom via NLP triple decomposition with categorical morphisms.
- **Linear Project:** https://linear.app/imajn/project/semantic-zoom-2bebe946c9e9
- **GitHub Issues:** https://github.com/research-developer/semantic-zoom/issues
- **PRD:** See `/home/claude/semantic-zoom-spec.md` (or ask orchestrator to share)

## Agent Hierarchy

### Phase Agents (7 total)
Each phase has a dedicated agent coordinating that phase's issues:
- `phase-1-agent` → Ingestion Pipeline (NSM-35→38, GH #1→4)
- `phase-2-agent` → Grammatical Classification (NSM-39→42, GH #5→8)
- `phase-3-agent` → Morphism Mapping (NSM-43→45, GH #9→11)
- `phase-4-agent` → Frame Integration (NSM-46→48, GH #12→14)
- `phase-5-agent` → Graph Construction (NSM-49→52, GH #15→18)
- `phase-6-agent` → Zoom Operations (NSM-53→56, GH #19→22)
- `phase-7-agent` → Compilation & Linting (NSM-57→60, GH #23→26)

### Issue Agents
Phase agents spawn issue-specific agents as needed. Name format: `issue-N-agent` (e.g., `issue-1-agent` for word tokenization).

## Git Worktree Strategy

### When to Use Worktrees
- Multiple implementation approaches being explored
- Parallel development on independent issues
- A/B testing different algorithms
- Uncertainty requiring experimentation

### Worktree Commands
```bash
# Create worktree for an issue
git worktree add ../semantic-zoom-issue-N issue-N

# Create worktree for experimental approach
git worktree add ../semantic-zoom-issue-N-approach-A issue-N-approach-A

# List worktrees
git worktree list

# Remove worktree when done
git worktree remove ../semantic-zoom-issue-N
```

### Branch Naming Convention
- `issue-N` - Main branch for issue N
- `issue-N-approach-A` - Alternative approach A for issue N
- `issue-N-approach-B` - Alternative approach B for issue N
- `phase-N` - Integration branch for phase N

### Context Preservation with `--fork-session`
When spawning agents in worktrees, use `--fork-session` BEFORE moving to worktree:
```bash
# In main repo, fork session to preserve context
claude --fork-session

# THEN move to worktree
cd ../semantic-zoom-issue-N

# Or spawn new agent directly in worktree
cd ../semantic-zoom-issue-N && claude --dangerously-skip-permissions
```

## Directory Structure
```
semantic-zoom/
├── CLAUDE.md           # This file
├── pyproject.toml      # Project config
├── src/
│   └── semantic_zoom/
│       ├── __init__.py
│       ├── phase1/     # Ingestion Pipeline
│       │   ├── tokenizer.py
│       │   ├── pos_tagger.py
│       │   ├── dependency_parser.py
│       │   └── triple_extractor.py
│       ├── phase2/     # Grammatical Classification
│       │   ├── noun_person.py
│       │   ├── verb_tense.py
│       │   ├── adjective_order.py
│       │   └── adverb_tier.py
│       ├── phase3/     # Morphism Mapping
│       │   ├── preposition_symbols.py
│       │   ├── focusing_adverbs.py
│       │   └── discourse_adverbs.py
│       ├── phase4/     # Frame Integration
│       │   ├── framenet_assignment.py
│       │   ├── slot_filling.py
│       │   └── plan_description.py
│       ├── phase5/     # Graph Construction
│       │   ├── nodes.py
│       │   ├── edges.py
│       │   ├── morphisms.py
│       │   └── inter_frame.py
│       ├── phase6/     # Zoom Operations
│       │   ├── seed_selection.py
│       │   ├── subgraph_extraction.py
│       │   ├── sparse_render.py
│       │   └── expansion.py
│       └── phase7/     # Compilation & Linting
│           ├── grammar_check.py
│           ├── ambiguity_detection.py
│           ├── user_prompts.py
│           └── preservation.py
└── tests/
    ├── phase1/
    ├── phase2/
    ├── phase3/
    ├── phase4/
    ├── phase5/
    ├── phase6/
    └── phase7/
```

## TDD/BDD Workflow
1. Read GitHub issue for acceptance criteria
2. Write failing tests FIRST
3. Implement until tests pass
4. Commit with message: `feat(phase-N): description [NSM-XX]`
5. If uncertain, create worktree for alternative approach

## Dependencies
Primary: spaCy, nltk, framenet (via nltk.corpus)
Secondary: networkx (graphs), sentence-transformers (embeddings)

## Communication
- Phase agents report to orchestrator via iTerm MCP
- Use Linear comments for status updates
- Tag issues with `in-progress`, `blocked`, `needs-review`
