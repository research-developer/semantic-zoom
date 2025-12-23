"""NSM-59: User prompt system for clarification.

Provides prompts for:
- Grammar errors: accept suggestion, provide alternative, or ignore
- Ambiguities: select intended interpretation
- User response storage and application
- Batch resolution support
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid

from semantic_zoom.phase7.grammar_check import GrammarError
from semantic_zoom.phase7.ambiguity_detection import Ambiguity, Interpretation


class PromptOption(Enum):
    """Options for responding to prompts."""
    ACCEPT = auto()      # Accept the suggested correction
    IGNORE = auto()      # Keep original, ignore the error
    ALTERNATIVE = auto() # Provide custom alternative
    SELECT = auto()      # Select an interpretation (for ambiguity)


@dataclass
class PromptChoice:
    """A choice in a prompt.

    Attributes:
        option_type: The type of option
        label: Human-readable label
        value: The value if selected (e.g., correction text)
    """
    option_type: PromptOption
    label: str
    value: Optional[str] = None


@dataclass
class CorrectionPrompt:
    """A prompt for correcting a grammar error.

    Attributes:
        prompt_id: Unique identifier
        error: The grammar error being addressed
        options: Available response options
        message: Human-readable prompt message
    """
    prompt_id: str
    error: GrammarError
    options: list[PromptChoice]
    message: str


@dataclass
class AmbiguityPrompt:
    """A prompt for resolving an ambiguity.

    Attributes:
        prompt_id: Unique identifier
        ambiguity: The ambiguity being addressed
        options: Available interpretation options
        message: Human-readable prompt message
    """
    prompt_id: str
    ambiguity: Ambiguity
    options: list[PromptChoice]
    message: str


@dataclass
class UserResponse:
    """A user's response to a prompt.

    Attributes:
        prompt_id: ID of the prompt being responded to
        option: The chosen option type
        alternative_text: Custom text if ALTERNATIVE chosen
        selected_index: Index of selected interpretation if SELECT chosen
    """
    prompt_id: str
    option: PromptOption
    alternative_text: Optional[str] = None
    selected_index: Optional[int] = None


@dataclass
class ResolvedAmbiguity:
    """An ambiguity with user-selected interpretation.

    Attributes:
        ambiguity: The original ambiguity
        selected_interpretation: The chosen interpretation
    """
    ambiguity: Ambiguity
    selected_interpretation: Interpretation


def _generate_prompt_id() -> str:
    """Generate a unique prompt identifier."""
    return f"prompt_{uuid.uuid4().hex[:12]}"


def _generate_deterministic_prompt_id(error: GrammarError) -> str:
    """Generate a deterministic prompt ID based on error position.

    This ensures the same error always produces the same prompt ID,
    enabling stored responses to be retrieved correctly.
    """
    return f"prompt_error_{error.start_char}_{error.end_char}"


def create_correction_prompt(
    error: GrammarError,
    deterministic: bool = False,
) -> CorrectionPrompt:
    """Create a prompt for a grammar error.

    Args:
        error: The grammar error to create a prompt for
        deterministic: If True, use deterministic ID based on error position.
            This is required for apply_stored_responses to work correctly.

    Returns:
        CorrectionPrompt with options for accept/ignore/alternative
    """
    options = []

    # Accept option (if suggestion available)
    if error.suggestion:
        options.append(PromptChoice(
            option_type=PromptOption.ACCEPT,
            label=f"Accept: '{error.suggestion}'",
            value=error.suggestion,
        ))

    # Alternative option
    options.append(PromptChoice(
        option_type=PromptOption.ALTERNATIVE,
        label="Provide alternative",
        value=None,
    ))

    # Ignore option
    options.append(PromptChoice(
        option_type=PromptOption.IGNORE,
        label="Ignore this error",
        value=error.text,
    ))

    message = f"{error.message}\nOriginal: '{error.text}'"
    if error.suggestion:
        message += f"\nSuggestion: '{error.suggestion}'"

    # Use deterministic ID for stored response matching
    prompt_id = (
        _generate_deterministic_prompt_id(error)
        if deterministic
        else _generate_prompt_id()
    )

    return CorrectionPrompt(
        prompt_id=prompt_id,
        error=error,
        options=options,
        message=message,
    )


def create_ambiguity_prompt(ambiguity: Ambiguity) -> AmbiguityPrompt:
    """Create a prompt for an ambiguity.

    Args:
        ambiguity: The ambiguity to create a prompt for

    Returns:
        AmbiguityPrompt with options for each interpretation
    """
    options = []

    for i, interp in enumerate(ambiguity.interpretations):
        options.append(PromptChoice(
            option_type=PromptOption.SELECT,
            label=interp.description,
            value=str(i),
        ))

    message = f"Ambiguous: '{ambiguity.text}'\nPlease select the intended meaning:"

    return AmbiguityPrompt(
        prompt_id=_generate_prompt_id(),
        ambiguity=ambiguity,
        options=options,
        message=message,
    )


def apply_response(
    text: str,
    error: GrammarError,
    response: UserResponse,
) -> str:
    """Apply a user response to text.

    Args:
        text: Original text
        error: The grammar error
        response: User's response

    Returns:
        Modified text based on response
    """
    if response.option == PromptOption.IGNORE:
        return text

    if response.option == PromptOption.ACCEPT and error.suggestion:
        # Apply the suggestion
        return text[:error.start_char] + error.suggestion + text[error.end_char:]

    if response.option == PromptOption.ALTERNATIVE and response.alternative_text:
        # Apply custom alternative
        return text[:error.start_char] + response.alternative_text + text[error.end_char:]

    return text


def resolve_ambiguity(
    ambiguity: Ambiguity,
    response: UserResponse,
) -> Optional[ResolvedAmbiguity]:
    """Resolve an ambiguity with user response.

    Args:
        ambiguity: The ambiguity to resolve
        response: User's response

    Returns:
        ResolvedAmbiguity with selected interpretation, or None if invalid
    """
    if response.option != PromptOption.SELECT:
        return None

    if response.selected_index is None:
        return None

    if response.selected_index < 0 or response.selected_index >= len(ambiguity.interpretations):
        return None

    return ResolvedAmbiguity(
        ambiguity=ambiguity,
        selected_interpretation=ambiguity.interpretations[response.selected_index],
    )


class ResponseStore:
    """Store for user responses.

    Enables storing responses and applying them to parses.
    """

    def __init__(self):
        """Initialize empty response store."""
        self._responses: dict[str, UserResponse] = {}

    def add(self, response: UserResponse) -> None:
        """Add a response to the store.

        Args:
            response: The response to store
        """
        self._responses[response.prompt_id] = response

    def get(self, prompt_id: str) -> Optional[UserResponse]:
        """Get a response by prompt ID.

        Args:
            prompt_id: The prompt ID to look up

        Returns:
            The stored response, or None if not found
        """
        return self._responses.get(prompt_id)

    def clear(self) -> None:
        """Clear all stored responses."""
        self._responses.clear()

    def __len__(self) -> int:
        """Return number of stored responses."""
        return len(self._responses)


def apply_stored_responses(
    text: str,
    errors: list[GrammarError],
    store: ResponseStore,
) -> str:
    """Apply all stored responses to text.

    Args:
        text: Original text
        errors: List of errors with corresponding prompts
        store: Store containing user responses

    Returns:
        Text with all accepted corrections applied

    Note:
        Prompts must be created with deterministic=True for this to work.
        The deterministic ID is based on error position (start_char, end_char).
    """
    # Sort errors by position (reverse to apply from end)
    sorted_errors = sorted(errors, key=lambda e: e.start_char, reverse=True)

    result = text
    for error in sorted_errors:
        # Use deterministic prompt ID to match stored responses
        prompt = create_correction_prompt(error, deterministic=True)
        response = store.get(prompt.prompt_id)
        if response:
            result = apply_response(result, error, response)

    return result


def create_batch_prompts(errors: list[GrammarError]) -> list[CorrectionPrompt]:
    """Create prompts for multiple errors.

    Args:
        errors: List of grammar errors

    Returns:
        List of correction prompts
    """
    return [create_correction_prompt(error) for error in errors]


def apply_batch_responses(
    text: str,
    errors: list[GrammarError],
    responses: list[UserResponse],
) -> str:
    """Apply batch responses to text.

    Args:
        text: Original text
        errors: List of grammar errors
        responses: List of user responses (same order as errors)

    Returns:
        Text with responses applied
    """
    if len(errors) != len(responses):
        raise ValueError("Number of errors and responses must match")

    # Sort by position (reverse to apply from end)
    error_responses = sorted(
        zip(errors, responses),
        key=lambda x: x[0].start_char,
        reverse=True,
    )

    result = text
    for error, response in error_responses:
        result = apply_response(result, error, response)

    return result
