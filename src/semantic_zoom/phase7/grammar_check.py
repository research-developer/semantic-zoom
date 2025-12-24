"""NSM-57: Grammar check integration.

Identifies grammatical errors with:
- Error type and location (character spans)
- Severity levels: ERROR, WARNING, INFO
- Suggestions for corrections
- Original and corrected version preservation
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import re

# Lazy load spacy
_nlp = None

# Lazy load CMU pronouncing dictionary
_cmudict = None


def _get_nlp():
    """Lazy load spacy model."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_cmudict():
    """Lazy load CMU pronouncing dictionary."""
    global _cmudict
    if _cmudict is None:
        import nltk
        try:
            from nltk.corpus import cmudict
            _cmudict = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict
            _cmudict = cmudict.dict()
    return _cmudict


def _starts_with_vowel_sound(word: str) -> bool:
    """Check if a word starts with a vowel sound using CMU pronouncing dictionary.

    Uses phonological analysis instead of orthographic:
    - 'hour' → vowel sound (AW1) → use 'an'
    - 'university' → consonant sound (Y) → use 'a'

    Falls back to orthographic check if word not in dictionary.

    Args:
        word: Word to check

    Returns:
        True if word starts with vowel sound
    """
    cmudict = _get_cmudict()
    word_lower = word.lower()

    if word_lower in cmudict:
        # Get first pronunciation
        pronunciation = cmudict[word_lower][0]
        if pronunciation:
            first_phoneme = pronunciation[0]
            # CMU vowel phonemes end with a digit (stress marker)
            # e.g., "AH0", "EY1", "IY2"
            return first_phoneme[-1].isdigit()

    # Fallback to orthographic with known exceptions
    first_char = word[0].lower() if word else ""

    # Words that start with vowel letter but consonant sound
    consonant_sound_prefixes = ("uni", "eu", "one", "once", "use")
    if word_lower.startswith(consonant_sound_prefixes):
        return False

    # Words that start with consonant letter but vowel sound
    vowel_sound_prefixes = ("hour", "honest", "honor", "heir")
    if word_lower.startswith(vowel_sound_prefixes):
        return True

    # Default to orthographic check
    return first_char in {"a", "e", "i", "o", "u"}


class Severity(Enum):
    """Severity levels for grammar errors."""
    ERROR = auto()    # Must be fixed (agreement, fragments)
    WARNING = auto()  # Should be fixed (awkward constructions)
    INFO = auto()     # Style suggestions (passive voice)


@dataclass
class GrammarError:
    """A grammatical error with location and suggestion.

    Attributes:
        error_type: Type of error (SUBJECT_VERB_AGREEMENT, ARTICLE, etc.)
        severity: ERROR, WARNING, or INFO
        start_char: Start character position
        end_char: End character position
        text: The erroneous text
        suggestion: Suggested correction (if available)
        message: Human-readable error message
    """
    error_type: str
    severity: Severity
    start_char: int
    end_char: int
    text: str
    suggestion: Optional[str]
    message: str


@dataclass
class GrammarCheckResult:
    """Result of grammar checking.

    Attributes:
        original: The original input text
        corrected: Text with all suggestions applied
        errors: List of detected errors
    """
    original: str
    corrected: str
    errors: list[GrammarError] = field(default_factory=list)


# Subject-verb agreement rules
_SINGULAR_SUBJECTS = {"he", "she", "it", "this", "that", "everyone", "someone", "anyone", "nobody"}
_PLURAL_SUBJECTS = {"they", "we", "these", "those"}

# Article rules - now handled by _starts_with_vowel_sound() using CMU dict

# Common error patterns
_DOUBLE_NEGATIVE_PATTERNS = [
    (r"\bdon't\s+\w*\s*no\b", "double negative"),
    (r"\bwon't\s+\w*\s*no\b", "double negative"),
    (r"\bcan't\s+\w*\s*no\b", "double negative"),
    (r"\bdoesn't\s+\w*\s*no\b", "double negative"),
    (r"\bdidn't\s+\w*\s*no\b", "double negative"),
    (r"\bnever\s+\w*\s*no\b", "double negative"),
]


def _check_subject_verb_agreement(doc) -> list[GrammarError]:
    """Check for subject-verb agreement errors."""
    errors = []

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token
            verb = token.head

            # Check singular subject with plural verb form
            subject_text = subject.text.lower()
            verb_text = verb.text.lower()

            # Detect "The dogs runs" pattern
            if subject.tag_ == "NNS" and verb.tag_ == "VBZ":
                # Plural noun with singular verb
                suggestion = verb.lemma_
                errors.append(GrammarError(
                    error_type="SUBJECT_VERB_AGREEMENT",
                    severity=Severity.ERROR,
                    start_char=verb.idx,
                    end_char=verb.idx + len(verb.text),
                    text=verb.text,
                    suggestion=suggestion,
                    message=f"Plural subject '{subject.text}' requires plural verb form",
                ))

            # Detect "The dog run" pattern
            elif subject.tag_ == "NN" and verb.tag_ == "VBP" and subject_text not in {"i", "you"}:
                # Singular noun with plural verb
                if verb_text not in {"be", "have", "do"}:
                    suggestion = verb_text + "s" if not verb_text.endswith("s") else verb_text
                    errors.append(GrammarError(
                        error_type="SUBJECT_VERB_AGREEMENT",
                        severity=Severity.ERROR,
                        start_char=verb.idx,
                        end_char=verb.idx + len(verb.text),
                        text=verb.text,
                        suggestion=suggestion,
                        message=f"Singular subject '{subject.text}' requires singular verb form",
                    ))

    return errors


def _check_article_errors(doc) -> list[GrammarError]:
    """Check for article errors (a/an) using phonological analysis.

    Uses CMU Pronouncing Dictionary to detect vowel/consonant sounds:
    - 'a hour' → error (hour starts with vowel sound)
    - 'an university' → error (university starts with /j/ consonant)
    """
    errors = []

    for i, token in enumerate(doc[:-1]):
        if token.text.lower() == "a":
            next_token = doc[i + 1]
            # Check if next word starts with vowel sound
            if next_token.text and _starts_with_vowel_sound(next_token.text):
                errors.append(GrammarError(
                    error_type="ARTICLE",
                    severity=Severity.WARNING,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    text=token.text,
                    suggestion="an",
                    message=f"Use 'an' before '{next_token.text}' (vowel sound)",
                ))
        elif token.text.lower() == "an":
            next_token = doc[i + 1]
            # Check if next word starts with consonant sound
            if next_token.text and not _starts_with_vowel_sound(next_token.text):
                errors.append(GrammarError(
                    error_type="ARTICLE",
                    severity=Severity.WARNING,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    text=token.text,
                    suggestion="a",
                    message=f"Use 'a' before '{next_token.text}' (consonant sound)",
                ))

    return errors


def _check_double_negatives(text: str) -> list[GrammarError]:
    """Check for double negatives."""
    errors = []

    for pattern, desc in _DOUBLE_NEGATIVE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            errors.append(GrammarError(
                error_type="DOUBLE_NEGATIVE",
                severity=Severity.ERROR,
                start_char=match.start(),
                end_char=match.end(),
                text=match.group(),
                suggestion=None,  # Complex to suggest
                message="Avoid double negatives",
            ))

    return errors


def _check_fragment(doc) -> list[GrammarError]:
    """Check for sentence fragments."""
    errors = []

    # Check if sentence has a root verb
    has_root = any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in doc)
    has_subject = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)

    # Fragment: starts with subordinating conjunction but no main clause
    if doc and doc[0].text.lower() in ("because", "although", "if", "when", "while"):
        if not has_root or not has_subject:
            errors.append(GrammarError(
                error_type="FRAGMENT",
                severity=Severity.WARNING,
                start_char=0,
                end_char=len(doc.text),
                text=doc.text,
                suggestion=None,
                message="Sentence fragment: subordinate clause without main clause",
            ))

    return errors


def _check_comma_after_introductory(doc) -> list[GrammarError]:
    """Check for missing comma after introductory elements."""
    errors = []

    introductory_adverbs = {"however", "therefore", "moreover", "furthermore",
                           "nevertheless", "consequently", "additionally"}

    if doc and doc[0].text.lower() in introductory_adverbs:
        # Check if followed by comma
        if len(doc) > 1 and doc[1].text != ",":
            errors.append(GrammarError(
                error_type="MISSING_COMMA",
                severity=Severity.INFO,
                start_char=doc[0].idx + len(doc[0].text),
                end_char=doc[0].idx + len(doc[0].text),
                text="",
                suggestion=",",
                message=f"Add comma after introductory '{doc[0].text}'",
            ))

    return errors


def check_grammar(text: str) -> GrammarCheckResult:
    """Check text for grammatical errors.

    Args:
        text: Input text to check

    Returns:
        GrammarCheckResult with original, corrected text, and errors
    """
    if not text or not text.strip():
        return GrammarCheckResult(original=text, corrected=text, errors=[])

    nlp = _get_nlp()
    doc = nlp(text)

    errors = []

    # Run all checks
    errors.extend(_check_subject_verb_agreement(doc))
    errors.extend(_check_article_errors(doc))
    errors.extend(_check_double_negatives(text))
    errors.extend(_check_fragment(doc))
    errors.extend(_check_comma_after_introductory(doc))

    # Sort errors by position
    errors.sort(key=lambda e: e.start_char)

    # Generate corrected version by applying suggestions
    corrected = _apply_corrections(text, errors)

    return GrammarCheckResult(
        original=text,
        corrected=corrected,
        errors=errors,
    )


def _apply_corrections(text: str, errors: list[GrammarError]) -> str:
    """Apply all suggestions to create corrected text."""
    if not errors:
        return text

    # Sort by position in reverse to apply from end to start
    sorted_errors = sorted(errors, key=lambda e: e.start_char, reverse=True)

    result = text
    for error in sorted_errors:
        if error.suggestion is not None:
            result = result[:error.start_char] + error.suggestion + result[error.end_char:]

    return result
