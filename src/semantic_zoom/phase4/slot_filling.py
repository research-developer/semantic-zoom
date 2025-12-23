"""NSM-47: Frame element slot filling.

Maps sentence arguments to frame element slots (Agent, Theme, Goal, etc.)
with support for unfilled and implicit arguments.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from semantic_zoom.phase4.framenet_assignment import (
    assign_frame,
    FrameCandidate,
    FrameElement,
)


class SlotStatus(Enum):
    """Status of a frame element slot."""
    FILLED = "filled"
    UNFILLED = "unfilled"
    IMPLICIT = "implicit"


@dataclass
class FilledSlot:
    """A frame element slot with its filling status."""
    element_name: str
    core_type: str  # "Core", "Peripheral", "Extra-Thematic"
    status: SlotStatus
    value: Optional[str] = None
    implicit: bool = False

    def __post_init__(self):
        if self.status == SlotStatus.IMPLICIT:
            self.implicit = True


@dataclass
class FrameInstance:
    """An instantiated frame with filled slots."""
    frame_name: str
    frame_id: int
    verb: str
    filled_slots: list[FilledSlot]
    all_slots: list[FilledSlot] = field(default_factory=list)

    def __post_init__(self):
        if not self.all_slots:
            self.all_slots = self.filled_slots.copy()


# Mapping from syntactic roles to likely frame element names
SUBJECT_ROLES = {
    "Agent", "Actor", "Cause", "Experiencer", "Donor", "Giver",
    "Speaker", "Cognizer", "Creator", "Author", "Ingestor",
    "Theme", "Breaker", "Killer", "Cook", "Builder", "Writer",
    "Self_mover", "Perceiver", "Owner"
}

OBJECT_ROLES = {
    "Theme", "Patient", "Undergoer", "Phenomenon", "Content",
    "Message", "Created_entity", "Food", "Building", "Text",
    "Broken_entity", "Victim", "Ingestibles", "Entity",
    "Information", "Item", "Goods", "Resource"
}

INDIRECT_OBJECT_ROLES = {
    "Recipient", "Goal", "Beneficiary", "Addressee", "Receiver"
}


def _match_argument_to_element(
    arg_value: Optional[str],
    arg_type: str,  # "subject", "object", "indirect_object"
    frame_elements: list[FrameElement]
) -> tuple[Optional[str], float]:
    """Match an argument to the best frame element.

    Returns:
        Tuple of (element_name, confidence)
    """
    if arg_value is None:
        return None, 0.0

    # Get the role set based on argument type
    if arg_type == "subject":
        preferred_roles = SUBJECT_ROLES
    elif arg_type == "object":
        preferred_roles = OBJECT_ROLES
    elif arg_type == "indirect_object":
        preferred_roles = INDIRECT_OBJECT_ROLES
    else:
        preferred_roles = set()

    # First pass: look for Core elements matching preferred roles
    for fe in frame_elements:
        if fe.core_type == "Core" and fe.name in preferred_roles:
            return fe.name, 0.9

    # Second pass: any element matching preferred roles
    for fe in frame_elements:
        if fe.name in preferred_roles:
            return fe.name, 0.7

    # Third pass: first Core element
    core_elements = [fe for fe in frame_elements if fe.core_type == "Core"]
    if core_elements:
        return core_elements[0].name, 0.5

    return None, 0.0


def _detect_implicit_arguments(
    frame_elements: list[FrameElement],
    filled_element_names: set[str],
    subject: Optional[str],
    verb: str
) -> list[FilledSlot]:
    """Detect implicit (unexpressed but understood) arguments."""
    implicit_slots = []

    # Check for inchoative/anticausative alternations
    # e.g., "The door opened" - implicit opener
    if subject is not None:
        # Look for unfilled agent-like Core elements
        for fe in frame_elements:
            if fe.core_type == "Core" and fe.name not in filled_element_names:
                if fe.name in SUBJECT_ROLES:
                    implicit_slots.append(FilledSlot(
                        element_name=fe.name,
                        core_type=fe.core_type,
                        status=SlotStatus.IMPLICIT,
                        value=None,
                        implicit=True
                    ))

    # Pro-drop detection (subject is None but expected)
    if subject is None:
        for fe in frame_elements:
            if fe.core_type == "Core" and fe.name in SUBJECT_ROLES:
                if fe.name not in filled_element_names:
                    implicit_slots.append(FilledSlot(
                        element_name=fe.name,
                        core_type=fe.core_type,
                        status=SlotStatus.IMPLICIT,
                        value=None,
                        implicit=True
                    ))

    return implicit_slots


def fill_slots(
    verb: str,
    subject: Optional[str] = None,
    object: Optional[str] = None,
    indirect_object: Optional[str] = None,
    context: Optional[str] = None,
    frame: Optional[FrameCandidate] = None
) -> FrameInstance:
    """Fill frame element slots with sentence arguments.

    Args:
        verb: The verb lemma
        subject: The syntactic subject
        object: The direct object
        indirect_object: The indirect object
        context: Full sentence for frame assignment
        frame: Optional pre-assigned frame

    Returns:
        FrameInstance with filled and unfilled slots
    """
    # Get frame if not provided
    if frame is None:
        assignment = assign_frame(verb, context)
        if assignment.best_frame is None:
            return FrameInstance(
                frame_name="Unknown",
                frame_id=-1,
                verb=verb,
                filled_slots=[],
                all_slots=[]
            )
        frame = assignment.best_frame

    frame_elements = frame.frame_elements
    filled_slots = []
    filled_element_names = set()

    # Map subject to element
    if subject is not None:
        element_name, confidence = _match_argument_to_element(
            subject, "subject", frame_elements
        )
        if element_name:
            fe = next((fe for fe in frame_elements if fe.name == element_name), None)
            if fe:
                filled_slots.append(FilledSlot(
                    element_name=element_name,
                    core_type=fe.core_type,
                    status=SlotStatus.FILLED,
                    value=subject,
                    implicit=False
                ))
                filled_element_names.add(element_name)

    # Map direct object to element
    if object is not None:
        element_name, confidence = _match_argument_to_element(
            object, "object", frame_elements
        )
        if element_name and element_name not in filled_element_names:
            fe = next((fe for fe in frame_elements if fe.name == element_name), None)
            if fe:
                filled_slots.append(FilledSlot(
                    element_name=element_name,
                    core_type=fe.core_type,
                    status=SlotStatus.FILLED,
                    value=object,
                    implicit=False
                ))
                filled_element_names.add(element_name)

    # Map indirect object to element
    if indirect_object is not None:
        element_name, confidence = _match_argument_to_element(
            indirect_object, "indirect_object", frame_elements
        )
        if element_name and element_name not in filled_element_names:
            fe = next((fe for fe in frame_elements if fe.name == element_name), None)
            if fe:
                filled_slots.append(FilledSlot(
                    element_name=element_name,
                    core_type=fe.core_type,
                    status=SlotStatus.FILLED,
                    value=indirect_object,
                    implicit=False
                ))
                filled_element_names.add(element_name)

    # Build all_slots with unfilled elements
    all_slots = filled_slots.copy()

    for fe in frame_elements:
        if fe.name not in filled_element_names:
            all_slots.append(FilledSlot(
                element_name=fe.name,
                core_type=fe.core_type,
                status=SlotStatus.UNFILLED,
                value=None,
                implicit=False
            ))

    # Detect implicit arguments
    implicit_slots = _detect_implicit_arguments(
        frame_elements, filled_element_names, subject, verb
    )
    for slot in implicit_slots:
        if slot.element_name not in filled_element_names:
            # Update the unfilled slot to implicit
            for i, s in enumerate(all_slots):
                if s.element_name == slot.element_name and s.status == SlotStatus.UNFILLED:
                    all_slots[i] = slot
                    break

    return FrameInstance(
        frame_name=frame.frame_name,
        frame_id=frame.frame_id,
        verb=verb,
        filled_slots=filled_slots,
        all_slots=all_slots
    )
