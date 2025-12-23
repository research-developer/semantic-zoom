"""Phase 4: Frame Integration.

NSM-46: FrameNet frame assignment to verb predicates
NSM-47: Frame element slot filling
NSM-48: Plan vs Description classification
"""
from semantic_zoom.phase4.framenet_assignment import (
    FrameAssignment,
    FrameCandidate,
    FrameElement,
    assign_frame,
    disambiguate_polysemous,
)
from semantic_zoom.phase4.slot_filling import (
    FilledSlot,
    FrameInstance,
    SlotStatus,
    fill_slots,
)
from semantic_zoom.phase4.plan_description import (
    ClassificationResult,
    PropositionType,
    classify_proposition,
)

__all__ = [
    # NSM-46
    "FrameAssignment",
    "FrameCandidate",
    "FrameElement",
    "assign_frame",
    "disambiguate_polysemous",
    # NSM-47
    "FilledSlot",
    "FrameInstance",
    "SlotStatus",
    "fill_slots",
    # NSM-48
    "ClassificationResult",
    "PropositionType",
    "classify_proposition",
]
