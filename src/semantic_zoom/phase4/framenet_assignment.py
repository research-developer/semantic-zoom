"""NSM-46: FrameNet frame assignment to verb predicates.

Maps verbs to candidate FrameNet frames and disambiguates polysemous verbs
using context and semantic similarity.
"""
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

from nltk.corpus import framenet as fn


@dataclass
class FrameElement:
    """A frame element with its properties."""
    name: str
    core_type: str  # "Core", "Peripheral", "Extra-Thematic"
    definition: Optional[str] = None


@dataclass
class FrameCandidate:
    """A candidate frame for a verb with confidence score."""
    frame_name: str
    frame_id: int
    lexical_unit: str
    frame_elements: list[FrameElement]
    confidence: float
    definition: str


@dataclass
class FrameAssignment:
    """Result of frame assignment for a verb."""
    verb: str
    candidates: list[FrameCandidate]
    best_frame: Optional[FrameCandidate] = None

    def __post_init__(self):
        if self.candidates and self.best_frame is None:
            # Default to highest confidence candidate
            self.best_frame = max(self.candidates, key=lambda c: c.confidence)


# Cache for sentence embeddings model (lazy load)
_embedding_model = None


def _get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            _embedding_model = None
    return _embedding_model


@lru_cache(maxsize=1000)
def _get_frames_for_verb(verb: str) -> list:
    """Get all FrameNet frames that contain a lexical unit for this verb."""
    try:
        frames = fn.frames_by_lemma(verb)
        return list(frames)
    except Exception:
        return []


def _extract_frame_elements(frame) -> list[FrameElement]:
    """Extract frame elements from a FrameNet frame object."""
    elements = []
    for fe_name, fe in frame.FE.items():
        elements.append(FrameElement(
            name=fe_name,
            core_type=fe.coreType,
            definition=getattr(fe, 'definition', None)
        ))
    return elements


def _find_lexical_unit(frame, verb: str) -> str:
    """Find the lexical unit in the frame that matches the verb."""
    for lu_name in frame.lexUnit.keys():
        # Lexical units are formatted as "word.pos"
        if lu_name.lower().startswith(verb.lower() + "."):
            return lu_name
    # Fallback: return first matching or generic
    return f"{verb}.v"


def _compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using embeddings."""
    model = _get_embedding_model()
    if model is None:
        return 0.5  # Default similarity when no model available

    try:
        embeddings = model.encode([text1, text2])
        # Cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        # Clamp to [0, 1] range (cosine can be negative)
        return max(0.0, min(1.0, float(similarity)))
    except Exception:
        return 0.5


def _score_frame_for_context(frame, verb: str, context: Optional[str]) -> float:
    """Score how well a frame matches the given context."""
    if context is None:
        # No context: use frame frequency/salience heuristic
        # Frames with more lexical units tend to be more common
        base_score = min(len(frame.lexUnit) / 20.0, 0.5)
        return base_score

    # Use frame definition for semantic matching
    frame_def = frame.definition if hasattr(frame, 'definition') else ""

    # Compute similarity between context and frame definition
    similarity = _compute_semantic_similarity(context, frame_def)

    # Boost if frame name contains relevant keywords from context
    context_lower = context.lower()
    frame_name_lower = frame.name.lower().replace("_", " ")

    keyword_boost = 0.0
    for word in frame_name_lower.split():
        if len(word) > 3 and word in context_lower:
            keyword_boost += 0.1

    return min(similarity + keyword_boost, 1.0)


def assign_frame(verb: str, context: Optional[str] = None) -> FrameAssignment:
    """Assign FrameNet frames to a verb predicate.

    Args:
        verb: The verb lemma to look up
        context: Optional sentence context for disambiguation

    Returns:
        FrameAssignment with candidate frames and best match
    """
    frames = _get_frames_for_verb(verb)

    if not frames:
        return FrameAssignment(verb=verb, candidates=[], best_frame=None)

    candidates = []
    for frame in frames:
        try:
            confidence = _score_frame_for_context(frame, verb, context)
            candidate = FrameCandidate(
                frame_name=frame.name,
                frame_id=frame.ID,
                lexical_unit=_find_lexical_unit(frame, verb),
                frame_elements=_extract_frame_elements(frame),
                confidence=confidence,
                definition=frame.definition[:500] if frame.definition else ""
            )
            candidates.append(candidate)
        except Exception:
            continue

    # Sort by confidence
    candidates.sort(key=lambda c: c.confidence, reverse=True)

    return FrameAssignment(
        verb=verb,
        candidates=candidates,
        best_frame=candidates[0] if candidates else None
    )


def disambiguate_polysemous(
    verb: str,
    candidates: Optional[list[FrameCandidate]],
    context: str
) -> FrameAssignment:
    """Disambiguate a polysemous verb using context.

    Args:
        verb: The verb lemma
        candidates: Optional pre-fetched candidates (will fetch if None)
        context: Sentence context for disambiguation

    Returns:
        FrameAssignment with re-ranked candidates
    """
    if candidates is None:
        # Fetch candidates from FrameNet
        initial = assign_frame(verb, context=None)
        candidates = initial.candidates

    if not candidates:
        return FrameAssignment(verb=verb, candidates=[], best_frame=None)

    # Re-score all candidates with context
    rescored = []
    for candidate in candidates:
        # Get the original frame for full definition
        try:
            frame = fn.frame(candidate.frame_name)
            new_confidence = _score_frame_for_context(frame, verb, context)
            rescored.append(FrameCandidate(
                frame_name=candidate.frame_name,
                frame_id=candidate.frame_id,
                lexical_unit=candidate.lexical_unit,
                frame_elements=candidate.frame_elements,
                confidence=new_confidence,
                definition=candidate.definition
            ))
        except Exception:
            rescored.append(candidate)

    rescored.sort(key=lambda c: c.confidence, reverse=True)

    return FrameAssignment(
        verb=verb,
        candidates=rescored,
        best_frame=rescored[0] if rescored else None
    )
