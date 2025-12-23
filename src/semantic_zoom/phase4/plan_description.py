"""NSM-48: Plan vs Description classification.

Classifies propositions as PLAN (dynamic/eventive), DESCRIPTION (stative),
or HYBRID based on frame semantics and aspectual properties.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PropositionType(Enum):
    """Classification of proposition type."""
    PLAN = "plan"
    DESCRIPTION = "description"
    HYBRID = "hybrid"


@dataclass
class ClassificationResult:
    """Result of plan/description classification."""
    proposition_type: PropositionType
    confidence: float
    reasoning: str


# Frames that are inherently dynamic/eventive (favor PLAN)
DYNAMIC_FRAMES = {
    # Motion frames
    "Motion", "Self_motion", "Motion_directional", "Traversing",
    "Arriving", "Departing", "Fleeing", "Escaping", "Traveling",

    # Causation frames
    "Causation", "Cause_motion", "Cause_change", "Cause_harm",
    "Cause_to_start", "Cause_to_end", "Cause_change_of_phase",

    # Action frames
    "Activity", "Activity_start", "Activity_finish", "Activity_pause",
    "Intentionally_act", "Intentionally_create", "Creating",
    "Building", "Cooking_creation", "Manufacturing",

    # Communication frames
    "Communication", "Statement", "Telling", "Request", "Questioning",

    # Transfer frames
    "Giving", "Getting", "Commerce_buy", "Commerce_sell",

    # Manipulation frames
    "Manipulation", "Body_movement", "Placing", "Removing",

    # Change frames
    "Change_position_on_a_scale", "Becoming", "Change_of_phase",

    # Perception (active)
    "Perception_active", "Scrutiny", "Seeking",

    # Other dynamic
    "Attack", "Impact", "Apply_heat", "Opening", "Closure",
}

# Frames that are inherently stative (favor DESCRIPTION)
STATIVE_FRAMES = {
    # Cognition frames
    "Awareness", "Certainty", "Opinion", "Remembering",
    "Expectation", "Mental_property",

    # Perception (passive)
    "Perception_experience", "Perception_body",

    # Possession frames
    "Possession", "Have_associated", "Inclusion",

    # Relation frames
    "Similarity", "Compatibility", "Relation",
    "Being_in_category", "Categorization",

    # State frames
    "State", "Being_located", "Locative_relation",
    "Measurable_attributes", "Dimension",

    # Existence frames
    "Existence", "Entity", "Coming_to_be",

    # Attribute frames
    "Attributes", "Color", "Age", "Size",

    # Emotion state
    "Experiencer_focus", "Emotions_by_stimulus",
}

# Stative verbs (lexical aspect)
STATIVE_VERBS = {
    "be", "have", "know", "understand", "believe", "think",
    "want", "need", "like", "love", "hate", "prefer",
    "see", "hear", "feel", "smell", "taste",
    "seem", "appear", "look", "sound",
    "own", "possess", "belong", "contain", "include",
    "cost", "weigh", "measure", "equal",
    "resemble", "differ", "agree", "disagree",
    "exist", "matter", "depend", "consist",
}

# Aspect indicators that can shift classification
ASPECT_EFFECTS = {
    "progressive": ("dynamic_shift", 0.2),  # Stative + progressive -> more dynamic
    "habitual": ("stative_shift", 0.3),     # Dynamic + habitual -> more stative
    "inchoative": ("dynamic_shift", 0.4),   # Beginning of state -> dynamic
    "resultative": ("stative_shift", 0.3),  # Result of event -> stative
    "perfective": ("neutral", 0.0),
    "imperfective": ("neutral", 0.0),
}

# Mood effects
MOOD_EFFECTS = {
    "imperative": PropositionType.PLAN,
    "subjunctive": PropositionType.PLAN,
    "indicative": None,  # No effect
}

# Tense effects
TENSE_EFFECTS = {
    "future": PropositionType.PLAN,
    "past": None,  # Neutral
    "present": None,  # Neutral
}


def _is_frame_dynamic(frame_name: str) -> Optional[bool]:
    """Check if a frame is inherently dynamic or stative.

    Returns:
        True if dynamic, False if stative, None if unknown
    """
    if frame_name in DYNAMIC_FRAMES:
        return True
    if frame_name in STATIVE_FRAMES:
        return False
    return None


def _is_verb_stative(verb: str) -> bool:
    """Check if a verb is lexically stative."""
    return verb.lower() in STATIVE_VERBS


def _compute_base_classification(
    verb: str,
    frame_name: str
) -> tuple[PropositionType, float, str]:
    """Compute base classification from verb and frame."""
    reasons = []

    # Check frame type
    frame_dynamic = _is_frame_dynamic(frame_name)
    verb_stative = _is_verb_stative(verb)

    if frame_dynamic is True:
        base_type = PropositionType.PLAN
        confidence = 0.8
        reasons.append(f"Frame '{frame_name}' is dynamic/eventive")
    elif frame_dynamic is False:
        base_type = PropositionType.DESCRIPTION
        confidence = 0.8
        reasons.append(f"Frame '{frame_name}' is stative")
    elif verb_stative:
        base_type = PropositionType.DESCRIPTION
        confidence = 0.7
        reasons.append(f"Verb '{verb}' is lexically stative")
    else:
        # Default to PLAN for unknown frames with non-stative verbs
        base_type = PropositionType.PLAN
        confidence = 0.6
        reasons.append(f"Default classification for verb '{verb}'")

    return base_type, confidence, "; ".join(reasons)


def _apply_aspect_modulation(
    base_type: PropositionType,
    confidence: float,
    aspect: Optional[str]
) -> tuple[PropositionType, float, str]:
    """Apply aspectual modulation to classification."""
    if aspect is None or aspect not in ASPECT_EFFECTS:
        return base_type, confidence, ""

    effect_type, shift_amount = ASPECT_EFFECTS[aspect]

    if effect_type == "dynamic_shift":
        if base_type == PropositionType.DESCRIPTION:
            # Shift toward HYBRID or PLAN
            if shift_amount > 0.3:
                return PropositionType.HYBRID, confidence * 0.9, f"{aspect} aspect shifts toward dynamic"
            else:
                return PropositionType.HYBRID, confidence * 0.8, f"{aspect} aspect creates hybrid"
    elif effect_type == "stative_shift":
        if base_type == PropositionType.PLAN:
            # Shift toward HYBRID or DESCRIPTION
            if shift_amount > 0.3:
                return PropositionType.HYBRID, confidence * 0.9, f"{aspect} aspect shifts toward stative"
            else:
                return PropositionType.HYBRID, confidence * 0.8, f"{aspect} aspect creates hybrid"

    return base_type, confidence, ""


def _apply_mood_tense_modulation(
    base_type: PropositionType,
    confidence: float,
    mood: Optional[str],
    tense: Optional[str]
) -> tuple[PropositionType, float, str]:
    """Apply mood and tense effects."""
    reasons = []

    # Mood effects
    if mood and mood in MOOD_EFFECTS:
        mood_effect = MOOD_EFFECTS[mood]
        if mood_effect is not None:
            if mood_effect == PropositionType.PLAN and base_type != PropositionType.PLAN:
                base_type = PropositionType.PLAN
                confidence = min(confidence + 0.1, 1.0)
                reasons.append(f"{mood} mood indicates PLAN")

    # Tense effects
    if tense and tense in TENSE_EFFECTS:
        tense_effect = TENSE_EFFECTS[tense]
        if tense_effect is not None:
            if tense_effect == PropositionType.PLAN:
                if base_type == PropositionType.DESCRIPTION:
                    base_type = PropositionType.HYBRID
                    reasons.append(f"{tense} tense shifts toward PLAN")
                elif base_type == PropositionType.HYBRID:
                    base_type = PropositionType.PLAN
                    reasons.append(f"{tense} tense confirms PLAN")

    return base_type, confidence, "; ".join(reasons)


def classify_proposition(
    verb: str,
    frame_name: str,
    context: Optional[str] = None,
    aspect: Optional[str] = None,
    tense: Optional[str] = None,
    mood: Optional[str] = None
) -> ClassificationResult:
    """Classify a proposition as PLAN, DESCRIPTION, or HYBRID.

    Args:
        verb: The verb lemma
        frame_name: The assigned FrameNet frame name
        context: Optional sentence context
        aspect: Optional aspectual information (progressive, habitual, etc.)
        tense: Optional tense (past, present, future)
        mood: Optional mood (indicative, imperative, subjunctive)

    Returns:
        ClassificationResult with type, confidence, and reasoning
    """
    all_reasons = []

    # Step 1: Base classification from frame and verb
    prop_type, confidence, reason = _compute_base_classification(verb, frame_name)
    if reason:
        all_reasons.append(reason)

    # Step 2: Apply aspectual modulation
    prop_type, confidence, reason = _apply_aspect_modulation(prop_type, confidence, aspect)
    if reason:
        all_reasons.append(reason)

    # Step 3: Apply mood/tense modulation
    prop_type, confidence, reason = _apply_mood_tense_modulation(
        prop_type, confidence, mood, tense
    )
    if reason:
        all_reasons.append(reason)

    return ClassificationResult(
        proposition_type=prop_type,
        confidence=min(confidence, 1.0),
        reasoning="; ".join(all_reasons) if all_reasons else "Default classification"
    )
