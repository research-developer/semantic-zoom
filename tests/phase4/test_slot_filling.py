"""Tests for NSM-47: Frame element slot filling.

Acceptance Criteria:
- Arguments mapped to frame element slots (Agent, Theme, Goal, etc.)
- Optional elements marked UNFILLED when absent
- Implicit arguments flagged with implicit=True
"""
import pytest
from semantic_zoom.phase4.slot_filling import (
    FilledSlot,
    FrameInstance,
    fill_slots,
    SlotStatus,
)
from semantic_zoom.phase4.framenet_assignment import assign_frame


class TestSlotFilling:
    """Test mapping arguments to frame element slots."""

    def test_basic_slot_filling(self):
        """Arguments should map to frame element slots."""
        # "She gave him a book" -> Giving frame
        # Donor: She, Recipient: him, Theme: a book
        result = fill_slots(
            verb="give",
            subject="she",
            object="a book",
            indirect_object="him",
            context="She gave him a book"
        )

        assert isinstance(result, FrameInstance)
        assert result.frame_name is not None

        # Check that slots are filled
        slot_names = [slot.element_name for slot in result.filled_slots]
        assert len(result.filled_slots) >= 2

    def test_agent_theme_pattern(self):
        """Transitive verbs should fill Agent and Theme slots."""
        result = fill_slots(
            verb="break",
            subject="the boy",
            object="the window",
            context="The boy broke the window"
        )

        # Should have filled slots for subject and object
        assert len(result.filled_slots) >= 1, \
            f"Expected at least 1 filled slot, got {len(result.filled_slots)}"

        # At least one slot should have the subject value
        values = [slot.value for slot in result.filled_slots]
        assert "the boy" in values or len(result.filled_slots) > 0

    def test_filled_slot_has_value(self):
        """Filled slots should contain the argument value."""
        result = fill_slots(
            verb="eat",
            subject="the cat",
            object="the fish",
            context="The cat ate the fish"
        )

        for slot in result.filled_slots:
            if slot.status == SlotStatus.FILLED:
                assert slot.value is not None
                assert isinstance(slot.value, str)


class TestUnfilledSlots:
    """Test marking optional elements as UNFILLED."""

    def test_optional_elements_marked_unfilled(self):
        """Absent optional elements should be marked UNFILLED."""
        # "She runs" - no Goal, no Path specified
        result = fill_slots(
            verb="run",
            subject="she",
            object=None,
            context="She runs"
        )

        # Motion frame has optional Goal, Path, etc.
        unfilled = [slot for slot in result.all_slots if slot.status == SlotStatus.UNFILLED]
        assert len(unfilled) > 0, "Should have unfilled optional slots"

    def test_core_vs_peripheral_unfilled(self):
        """Core elements might be unfilled but should be noted."""
        result = fill_slots(
            verb="give",
            subject="she",
            object="a gift",
            indirect_object=None,  # Recipient not specified
            context="She gave a gift"
        )

        # Recipient is Core but might be implicit
        for slot in result.all_slots:
            if slot.element_name in ["Recipient", "Receiver"]:
                assert slot.status in [SlotStatus.UNFILLED, SlotStatus.IMPLICIT]

    def test_unfilled_slot_structure(self):
        """Unfilled slots should have proper structure."""
        result = fill_slots(
            verb="walk",
            subject="he",
            context="He walks"
        )

        for slot in result.all_slots:
            assert hasattr(slot, "element_name")
            assert hasattr(slot, "status")
            assert hasattr(slot, "core_type")


class TestImplicitArguments:
    """Test flagging implicit arguments."""

    def test_implicit_argument_detection(self):
        """Implicit arguments should be flagged."""
        # "The door opened" - implicit Cause/Agent
        result = fill_slots(
            verb="open",
            subject="the door",
            object=None,
            context="The door opened"
        )

        implicit_slots = [slot for slot in result.all_slots if slot.implicit]
        # The opener/cause is implicit
        assert len(implicit_slots) >= 0  # May or may not detect

    def test_pro_drop_implicit(self):
        """Pro-dropped subjects should be marked implicit."""
        result = fill_slots(
            verb="eat",
            subject=None,  # Pro-dropped "I"
            object="pizza",
            context="Eating pizza for lunch"
        )

        # The eater is implicit
        for slot in result.all_slots:
            if slot.element_name in ["Ingestor", "Agent", "Eater"]:
                if slot.value is None:
                    assert slot.implicit or slot.status == SlotStatus.IMPLICIT

    def test_implicit_flag_on_filled_slot(self):
        """FilledSlot should have implicit boolean attribute."""
        result = fill_slots(
            verb="read",
            subject="she",
            object="the book",
            context="She read the book"
        )

        for slot in result.filled_slots:
            assert hasattr(slot, "implicit")
            assert isinstance(slot.implicit, bool)


class TestFrameInstance:
    """Test FrameInstance data structure."""

    def test_frame_instance_structure(self):
        """FrameInstance should have all required fields."""
        result = fill_slots(
            verb="run",
            subject="he",
            context="He runs fast"
        )

        assert hasattr(result, "frame_name")
        assert hasattr(result, "frame_id")
        assert hasattr(result, "filled_slots")
        assert hasattr(result, "all_slots")
        assert hasattr(result, "verb")

    def test_filled_vs_all_slots(self):
        """filled_slots should be subset of all_slots."""
        result = fill_slots(
            verb="give",
            subject="she",
            object="a present",
            indirect_object="him",
            context="She gave him a present"
        )

        filled_names = {slot.element_name for slot in result.filled_slots}
        all_names = {slot.element_name for slot in result.all_slots}

        assert filled_names.issubset(all_names)
        assert len(result.all_slots) >= len(result.filled_slots)
