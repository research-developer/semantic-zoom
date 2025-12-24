#!/usr/bin/env python3
"""Semantic Zoom Pipeline Demo.

Interactive demonstration of the 7-phase NLP pipeline for quasi-deterministic
semantic zoom via triple decomposition with categorical morphisms.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box

console = Console()

def print_header(text: str, style: str = "bold cyan"):
    """Print a section header."""
    console.print(f"\n{'='*60}", style="dim")
    console.print(text, style=style)
    console.print('='*60, style="dim")

def print_subheader(text: str):
    """Print a subsection header."""
    console.print(f"\n[bold yellow]>>> {text}[/bold yellow]")


# ============================================================================
# PHASE 1: INGESTION PIPELINE
# ============================================================================

def demo_phase1(text: str):
    """Demonstrate Phase 1: Tokenization, POS tagging, dependency parsing."""
    from semantic_zoom.phase1.tokenizer import Tokenizer
    from semantic_zoom.phase1.pos_tagger import POSTagger
    from semantic_zoom.phase1.dependency_parser import DependencyParser

    print_header("PHASE 1: Ingestion Pipeline")
    console.print(Panel(text, title="Input Text", border_style="green"))

    # Tokenization
    print_subheader("Tokenization")
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)

    table = Table(title="Tokens", box=box.ROUNDED)
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Text", style="green")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")

    for t in tokens[:15]:  # Show first 15
        table.add_row(str(t.id), t.text, str(t.start_char), str(t.end_char))
    if len(tokens) > 15:
        table.add_row("...", f"({len(tokens) - 15} more)", "", "")
    console.print(table)

    # POS Tagging
    print_subheader("Part-of-Speech Tagging")
    pos_tagger = POSTagger()
    tagged = pos_tagger.tag(tokens)

    table = Table(title="POS Tags", box=box.ROUNDED)
    table.add_column("Token", style="green")
    table.add_column("POS", style="yellow")
    table.add_column("Tag", style="magenta")

    for t in tagged[:12]:
        table.add_row(t.text, t.pos, t.tag)
    if len(tagged) > 12:
        table.add_row("...", "...", "...")
    console.print(table)

    # Dependency Parsing
    print_subheader("Dependency Parse")
    dep_parser = DependencyParser()
    parsed = dep_parser.parse(tokens)

    table = Table(title="Dependencies", box=box.ROUNDED)
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Token", style="green")
    table.add_column("Dep", style="yellow")
    table.add_column("Head ID", style="blue", justify="center")
    table.add_column("Head", style="blue")

    for t in parsed[:15]:
        head_text = next((p.text for p in parsed if p.id == t.head_id), "ROOT")
        dep_colored = f"[bold red]{t.dep}[/bold red]" if t.dep == "ROOT" else t.dep
        table.add_row(str(t.id), t.text, dep_colored, str(t.head_id), head_text)
    console.print(table)

    # Dependency Tree Visualization
    print_subheader("Dependency Tree")
    root_tokens = [t for t in parsed if t.dep == "ROOT"]

    def build_tree(parent_id, parsed_tokens, tree_node):
        children = [t for t in parsed_tokens if t.head_id == parent_id and t.id != parent_id]
        for child in sorted(children, key=lambda x: x.id):
            child_node = tree_node.add(f"[green]{child.text}[/green] [dim]({child.dep})[/dim]")
            build_tree(child.id, parsed_tokens, child_node)

    for root in root_tokens:
        tree = Tree(f"[bold green]{root.text}[/bold green] [dim](ROOT)[/dim]")
        build_tree(root.id, parsed, tree)
        console.print(tree)

    return parsed


# ============================================================================
# PHASE 2: GRAMMATICAL CLASSIFICATION
# ============================================================================

def demo_phase2(text: str):
    """Demonstrate Phase 2: Grammatical classification."""
    from semantic_zoom.pipeline import Pipeline

    print_header("PHASE 2: Grammatical Classification")

    pipeline = Pipeline()
    tokens = pipeline.process(text)

    # Verb Tense Classification
    print_subheader("Verb Tense & Aspect (NSM-40)")
    verbs = [t for t in tokens if t.pos in ("VERB", "AUX")]

    table = Table(title="Verb Analysis", box=box.ROUNDED)
    table.add_column("Verb", style="green")
    table.add_column("Tense", style="yellow")
    table.add_column("Aspect", style="cyan")
    table.add_column("Tag", style="magenta")

    for v in verbs:
        tense = v.tense.name if v.tense else "-"
        aspect = v.aspect.name if v.aspect else "-"
        table.add_row(v.text, tense, aspect, v.tag)
    console.print(table)

    # Noun Person Classification
    print_subheader("Noun Grammatical Person (NSM-39)")
    nouns = [t for t in tokens if t.pos in ("NOUN", "PROPN", "PRON")]

    table = Table(title="Noun Analysis", box=box.ROUNDED)
    table.add_column("Noun", style="green")
    table.add_column("Person", style="yellow")
    table.add_column("Generic", style="cyan")

    for n in nouns[:10]:
        person = n.person.name if n.person else "-"
        generic = "Yes" if n.generic else "-"
        table.add_row(n.text, person, generic)
    console.print(table)

    # Adjective Ordering
    print_subheader("Adjective Order Categories (NSM-41)")
    adjectives = [t for t in tokens if t.pos == "ADJ"]

    if adjectives:
        table = Table(title="Adjective Categories", box=box.ROUNDED)
        table.add_column("Adjective", style="green")
        table.add_column("Slot", style="yellow")
        table.add_column("Original Pos", style="cyan", justify="center")

        for adj in adjectives:
            slot = adj.adj_slot.name if adj.adj_slot else "-"
            orig_pos = str(adj.adj_original_pos) if adj.adj_original_pos is not None else "-"
            table.add_row(adj.text, slot, orig_pos)
        console.print(table)

        console.print("\n[dim]Order: Opinion → Size → Age → Shape → Color → Origin → Material → Purpose[/dim]")

    # Adverb Tiers
    print_subheader("Adverb Tiers (NSM-42)")
    adverbs = [t for t in tokens if t.pos == "ADV"]

    if adverbs:
        table = Table(title="Adverb Tiers", box=box.ROUNDED)
        table.add_column("Adverb", style="green")
        table.add_column("Tier", style="yellow")
        table.add_column("Attaches To", style="cyan")

        for adv in adverbs:
            tier = adv.adv_tier.name if adv.adv_tier else "-"
            attachment = str(adv.adv_attachment) if adv.adv_attachment is not None else "-"
            table.add_row(adv.text, tier, attachment)
        console.print(table)

    return tokens


# ============================================================================
# PHASE 3: MORPHISM MAPPING
# ============================================================================

def demo_phase3(text: str):
    """Demonstrate Phase 3: Morphism mapping to categorical symbols."""
    from semantic_zoom.pipeline import Pipeline
    from semantic_zoom.phase3 import Phase3Processor

    print_header("PHASE 3: Morphism Mapping")

    # Get Phase 2 tokens first
    pipeline = Pipeline()
    phase2_tokens = pipeline.process(text)

    # Run Phase 3 processing
    processor = Phase3Processor()
    result = processor.process(phase2_tokens)

    # Preposition → Categorical Symbols
    print_subheader("Prepositions → Categorical Symbols (NSM-43)")

    prep_tokens = [t for t in result.tokens if t.prep_mapping is not None]

    if prep_tokens:
        table = Table(title="Preposition Morphisms", box=box.ROUNDED)
        table.add_column("Preposition", style="green")
        table.add_column("Symbol", style="yellow")
        table.add_column("Motion", style="cyan")
        table.add_column("Polarity", style="magenta")

        for t in prep_tokens:
            pm = t.prep_mapping
            symbol = pm.symbol.value if pm.symbol else "-"
            symbol_name = pm.symbol.name if pm.symbol else "-"
            motion = pm.state.motion if pm.state else "-"
            polarity = str(pm.state.polarity) if pm.state else "-"
            table.add_row(
                t.token.text,
                f"{symbol} ({symbol_name})",
                motion,
                polarity
            )
        console.print(table)

        console.print("\n[dim]Symbols: ◃ (toward), ▹ (away), ∈ (containment), ⊤ (on), ⊙ (temporal), etc.[/dim]")
    else:
        console.print("[dim]No prepositions found in this text.[/dim]")

    # Focusing Adverbs → Scope Operators
    print_subheader("Focusing Adverbs → Scope Operators (NSM-44)")

    focus_tokens = [t for t in result.tokens if t.is_focusing_adverb]

    if focus_tokens:
        table = Table(title="Focus Operators", box=box.ROUNDED)
        table.add_column("Adverb", style="green")
        table.add_column("Focus Type", style="yellow")
        table.add_column("Scope Target", style="cyan")

        for t in focus_tokens:
            scope_text = "-"
            if t.scope_target_idx is not None and t.scope_target_idx < len(result.tokens):
                scope_text = result.tokens[t.scope_target_idx].token.text
            table.add_row(
                t.token.text,
                t.focus_type or "-",
                scope_text
            )
        console.print(table)

        # Show scope bindings
        if focus_tokens and focus_tokens[0].scope_bindings:
            console.print("\n[bold]Scope Bindings (ranked by confidence):[/bold]")
            for binding in focus_tokens[0].scope_bindings[:3]:
                conf_bar = "█" * int(binding.confidence * 10)
                console.print(f"  • {binding.target} ({binding.position}): {conf_bar} {binding.confidence:.0%}")

        console.print("\n[dim]Operators: ONLY (restrictive), EVEN (scalar additive), JUST (minimizer)[/dim]")
    else:
        console.print("[dim]No focusing adverbs found (try: 'only', 'even', 'just', 'merely')[/dim]")

    # Discourse Adverbs → Inter-frame Relations
    print_subheader("Discourse Adverbs → Inter-frame Morphisms (NSM-45)")

    discourse_tokens = [t for t in result.tokens if t.is_discourse_adverb]

    if discourse_tokens:
        table = Table(title="Discourse Relations", box=box.ROUNDED)
        table.add_column("Adverb", style="green")
        table.add_column("Relation", style="yellow")
        table.add_column("Frame→Frame", style="cyan")

        for t in discourse_tokens:
            relation = t.discourse_relation.name if t.discourse_relation else "-"
            frame_conn = "-"
            if t.inter_frame_morphism:
                frame_conn = f"{t.inter_frame_morphism.source_frame} → {t.inter_frame_morphism.target_frame}"
            table.add_row(
                t.token.text,
                relation,
                frame_conn
            )
        console.print(table)

        # Show inter-frame morphism details
        if discourse_tokens and discourse_tokens[0].inter_frame_morphism:
            ifm = discourse_tokens[0].inter_frame_morphism
            console.print(f"\n[bold]Inter-frame Morphism:[/bold]")
            console.print(f"  Edge label: [yellow]{ifm.edge_label}[/yellow]")
            console.print(f"  Relation: [cyan]{ifm.relation.name}[/cyan]")
            console.print(f"  Strength: [green]{ifm.strength:.1f}[/green]")
    else:
        console.print("[dim]No discourse adverbs found (try: 'however', 'therefore', 'consequently')[/dim]")

    return result


# ============================================================================
# PHASE 6: ZOOM OPERATIONS
# ============================================================================

def demo_phase6(text: str):
    """Demonstrate Phase 6: Semantic zoom operations."""
    from semantic_zoom.phase1.dependency_parser import DependencyParser
    from semantic_zoom.phase1.tokenizer import Tokenizer
    from semantic_zoom.phase6.seed_selection import SeedSelector
    from semantic_zoom.phase6.subgraph_extraction import SubgraphExtractor

    print_header("PHASE 6: Zoom Operations")

    console.print(Panel(text, title="Input Text", border_style="green"))

    # Parse the text
    tokenizer = Tokenizer()
    dep_parser = DependencyParser()
    tokens = tokenizer.tokenize(text)
    parsed = dep_parser.parse(tokens)

    # Find the main verb as seed
    main_verb = None
    for t in parsed:
        if t.dep == "ROOT":
            main_verb = t
            break

    if not main_verb:
        console.print("[red]No main verb found![/red]")
        return

    print_subheader(f"Seed Selection: Main verb '{main_verb.text}' (ID: {main_verb.id})")

    # Create seed
    selector = SeedSelector()
    seed = selector.select_range(parsed, main_verb.id, main_verb.id)

    console.print(f"  Seed text: [green]{seed.text}[/green]")
    console.print(f"  Sentence bounds: [{seed.sentence_start_id}, {seed.sentence_end_id}]")
    console.print(f"  Clause root: {seed.clause_root_id}")

    # Demo different zoom levels
    extractor = SubgraphExtractor()

    for zoom_level in [1, 3, 5]:
        print_subheader(f"Zoom Level {zoom_level}")

        result = extractor.extract(parsed, [seed], zoom_level)
        subgraph_text = extractor.get_subgraph_text(parsed, result)

        # Highlight the seed
        console.print(f"  Extracted IDs: {result.word_ids}")
        console.print(f"  Token count: [cyan]{len(result.word_ids)}[/cyan]")

        # Show the extracted text with seed highlighted
        displayed = []
        for t in parsed:
            if t.id in result.word_ids:
                if t.id == main_verb.id:
                    displayed.append(f"[bold yellow]{t.text}[/bold yellow]")
                else:
                    displayed.append(f"[green]{t.text}[/green]")
            else:
                displayed.append(f"[dim]{t.text}[/dim]")

        console.print("  Text: " + " ".join(displayed))
        console.print()


# ============================================================================
# PHASE 7: AMBIGUITY DETECTION
# ============================================================================

def demo_phase7_ambiguity(text: str):
    """Demonstrate Phase 7: Ambiguity detection."""
    from semantic_zoom.phase7 import detect_ambiguities, AmbiguityType

    print_header("PHASE 7: Ambiguity Detection")

    console.print(Panel(text, title="Analyzing for Ambiguity", border_style="red"))

    result = detect_ambiguities(text)

    if not result.ambiguities:
        console.print("[green]No structural ambiguities detected![/green]")
        return result

    console.print(f"\n[bold red]Found {len(result.ambiguities)} ambiguities![/bold red]\n")

    for i, amb in enumerate(result.ambiguities, 1):
        # Color based on type
        type_colors = {
            AmbiguityType.PP_ATTACHMENT: "yellow",
            AmbiguityType.COORDINATION: "cyan",
            AmbiguityType.PRONOUN: "magenta",
            AmbiguityType.QUANTIFIER_SCOPE: "green",
            AmbiguityType.NEGATION_SCOPE: "red",
        }
        color = type_colors.get(amb.ambiguity_type, "white")

        console.print(Panel(
            f"[{color}]Type: {amb.ambiguity_type.name}[/{color}]\n"
            f"Text: \"{amb.text}\"\n"
            f"Span: [{amb.span[0]}, {amb.span[1]}]",
            title=f"Ambiguity #{i}",
            border_style=color
        ))

        # Show interpretations
        table = Table(title="Possible Interpretations", box=box.ROUNDED)
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Interpretation", style="white")
        table.add_column("Confidence", style="yellow", justify="center")

        for j, interp in enumerate(amb.interpretations, 1):
            conf_bar = "█" * int(interp.confidence * 10) + "░" * (10 - int(interp.confidence * 10))
            table.add_row(
                str(j),
                interp.description,
                f"{conf_bar} {interp.confidence:.0%}"
            )
        console.print(table)

        # Show antecedents for pronoun ambiguity
        if amb.possible_antecedents:
            console.print("\n[bold]Possible Antecedents:[/bold]")
            for ant in amb.possible_antecedents:
                console.print(f"  • {ant.text} (token {ant.token_id})")

        console.print()

    return result


# ============================================================================
# MAIN DEMO
# ============================================================================

def interactive_mode():
    """Run interactive mode where user can input their own sentences."""
    console.print(Panel.fit(
        "[bold cyan]SEMANTIC ZOOM - Interactive Mode[/bold cyan]\n"
        "[dim]Enter your own sentences to analyze![/dim]",
        border_style="cyan"
    ))

    console.print("""
[bold]Commands:[/bold]
  [green]<any text>[/green]  - Analyze a sentence through all phases
  [green]zoom <text>[/green] - Focus on zoom operations
  [green]ambig <text>[/green] - Focus on ambiguity detection
  [green]examples[/green]    - Show example sentences to try
  [green]quit[/green]        - Exit

[dim]Tip: Try sentences with prepositions, focusing adverbs (only, even, just),
     or discourse markers (however, therefore, consequently)[/dim]
""")

    examples = [
        "However, only the brilliant scientist quickly walked to the observatory.",
        "I saw the man with the telescope.",
        "Every student read a book.",
        "John told Bill that he was wrong.",
        "The old French wooden table was expensive.",
        "She only ate the cake.",
        "Therefore, we must reconsider our approach.",
        "The cat sat on the mat near the window.",
    ]

    while True:
        try:
            console.print("\n[bold cyan]>[/bold cyan] ", end="")
            user_input = input().strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "examples":
                console.print("\n[bold]Example sentences to try:[/bold]")
                for i, ex in enumerate(examples, 1):
                    console.print(f"  {i}. {ex}")
                continue

            # Handle zoom command
            if user_input.lower().startswith("zoom "):
                text = user_input[5:].strip()
                if text:
                    demo_phase6(text)
                continue

            # Handle ambig command
            if user_input.lower().startswith("ambig "):
                text = user_input[6:].strip()
                if text:
                    demo_phase7_ambiguity(text)
                continue

            # Full analysis
            text = user_input
            console.print(Panel(text, title="Analyzing", border_style="green"))

            demo_phase1(text)
            demo_phase2(text)
            demo_phase3(text)
            demo_phase6(text)
            demo_phase7_ambiguity(text)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Run the full pipeline demo."""
    import sys

    console.print(Panel.fit(
        "[bold cyan]SEMANTIC ZOOM[/bold cyan]\n"
        "[dim]Quasi-deterministic semantic zoom via NLP triple decomposition\n"
        "with categorical morphisms[/dim]\n\n"
        "[yellow]Neural-Symbolic Hybrid:[/yellow]\n"
        "  • Phase 1: spaCy neural models (ML-based parsing)\n"
        "  • Phases 2-7: Rule-based symbolic systems (interpretable)",
        border_style="cyan"
    ))

    # Check for interactive mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "-i" or sys.argv[1] == "--interactive":
            interactive_mode()
            return
        else:
            # Process command line argument as text
            text = " ".join(sys.argv[1:])
            console.print(Panel(text, title="Analyzing", border_style="green"))
            demo_phase1(text)
            demo_phase2(text)
            demo_phase3(text)
            demo_phase6(text)
            demo_phase7_ambiguity(text)
            return

    # Default demo mode
    complex_sentence = (
        "However, only the brilliant scientist who discovered the "
        "high-energy particle quickly walked to the ancient observatory."
    )

    ambiguous_sentences = [
        "I saw the man with the telescope.",
        "Every student read a book.",
        "John told Bill that he was wrong.",
        "Old men and women attended the ceremony.",
    ]

    console.print("\n[bold]Demo Sentence:[/bold]")
    console.print(Panel(complex_sentence, border_style="green"))

    console.print("\n[dim]Press Enter to continue through each phase...")
    console.print("Or run with [green]-i[/green] for interactive mode, or pass text as argument.[/dim]")

    # Phase 1: Tokenization & Parsing
    input("\n[Press Enter for Phase 1]")
    demo_phase1(complex_sentence)

    # Phase 2: Grammatical Classification
    input("\n[Press Enter for Phase 2]")
    demo_phase2(complex_sentence)

    # Phase 3: Morphism Mapping
    input("\n[Press Enter for Phase 3]")
    demo_phase3(complex_sentence)

    # Phase 6: Zoom Operations
    input("\n[Press Enter for Phase 6 (Zoom Operations)]")
    demo_phase6(complex_sentence)

    # Phase 7: Ambiguity Detection
    input("\n[Press Enter for Phase 7 (Ambiguity Detection)]")

    console.print("\n[bold]Classic Ambiguous Sentences:[/bold]")
    for sent in ambiguous_sentences:
        console.print(f"  • {sent}")

    for sent in ambiguous_sentences:
        input(f"\n[Press Enter to analyze: '{sent}']")
        demo_phase7_ambiguity(sent)

    # Summary
    print_header("DEMO COMPLETE", "bold green")
    console.print("""
[bold]What we demonstrated:[/bold]

  [cyan]Phase 1:[/cyan] Tokenization, POS tagging, dependency parsing
           → [yellow]ML-based[/yellow]: spaCy's neural models do the heavy lifting

  [cyan]Phase 2:[/cyan] Grammatical classification
           → [green]Rule-based[/green]: Tense/aspect, adjective ordering from linguistic rules

  [cyan]Phase 3:[/cyan] Morphism mapping
           → [green]Rule-based[/green]: Categorical symbols from formal semantics
           → Prepositions → σ symbols, focusing adverbs → scope operators

  [cyan]Phase 6:[/cyan] Semantic zoom operations
           → [green]Rule-based[/green]: Graph traversal via dependency structure
           → N-hop neighborhood expansion is deterministic

  [cyan]Phase 7:[/cyan] Ambiguity detection
           → [green]Rule-based[/green]: Pattern matching for known ambiguity types
           → PP-attachment, quantifier scope, pronoun reference

[bold green]This is neural-symbolic NLP:[/bold green]
  ML for parsing + symbolic reasoning for interpretability!

[dim]Run with -i for interactive mode, or pass text as argument.[/dim]
""")


if __name__ == "__main__":
    main()
