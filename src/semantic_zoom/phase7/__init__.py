"""Phase 7: Compilation & Linting.

NSM-57: Grammar check integration
NSM-58: Ambiguity detection
NSM-59: User prompt system for clarification
NSM-60: Original preservation alongside corrections
"""
from semantic_zoom.phase7.grammar_check import (
    GrammarError,
    GrammarCheckResult,
    Severity,
    check_grammar,
)
from semantic_zoom.phase7.ambiguity_detection import (
    Ambiguity,
    AmbiguityResult,
    AmbiguityType,
    Antecedent,
    Interpretation,
    detect_ambiguities,
)
from semantic_zoom.phase7.user_prompts import (
    AmbiguityPrompt,
    CorrectionPrompt,
    PromptChoice,
    PromptOption,
    ResolvedAmbiguity,
    ResponseStore,
    UserResponse,
    apply_batch_responses,
    apply_response,
    apply_stored_responses,
    create_ambiguity_prompt,
    create_batch_prompts,
    create_correction_prompt,
    resolve_ambiguity,
)
from semantic_zoom.phase7.preservation import (
    Version,
    VersionStore,
    WordID,
    WordMapping,
)

__all__ = [
    # NSM-57: Grammar check
    "GrammarError",
    "GrammarCheckResult",
    "Severity",
    "check_grammar",
    # NSM-58: Ambiguity detection
    "Ambiguity",
    "AmbiguityResult",
    "AmbiguityType",
    "Antecedent",
    "Interpretation",
    "detect_ambiguities",
    # NSM-59: User prompts
    "AmbiguityPrompt",
    "CorrectionPrompt",
    "PromptChoice",
    "PromptOption",
    "ResolvedAmbiguity",
    "ResponseStore",
    "UserResponse",
    "apply_batch_responses",
    "apply_response",
    "apply_stored_responses",
    "create_ambiguity_prompt",
    "create_batch_prompts",
    "create_correction_prompt",
    "resolve_ambiguity",
    # NSM-60: Preservation
    "Version",
    "VersionStore",
    "WordID",
    "WordMapping",
]
