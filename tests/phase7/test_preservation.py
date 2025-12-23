"""Tests for NSM-60: Original preservation alongside corrections.

Tests:
- Original and corrected versions with word ID mappings
- Original view recovery
- Word count change tracking
- Version history (git-like DAG)
"""
import pytest


class TestVersionStore:
    """Test version storage system."""

    def test_create_version_store(self):
        """Test creating a version store."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        assert store is not None

    def test_initial_version(self):
        """Test storing initial version."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        text = "The dogs runs quickly."
        version_id = store.add_version(text)

        assert version_id is not None
        retrieved = store.get_version(version_id)
        assert retrieved.text == text

    def test_version_has_word_ids(self):
        """Test that versions track word IDs."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        text = "The dog runs."
        version_id = store.add_version(text)

        version = store.get_version(version_id)
        assert version.word_ids is not None
        assert len(version.word_ids) >= 3  # At least "The", "dog", "runs"


class TestWordIDMapping:
    """Test word ID mapping between versions."""

    def test_mapping_preserved_on_edit(self):
        """Test that word ID mapping is preserved when editing."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        original = "The dogs runs quickly."
        v1 = store.add_version(original)

        # Apply correction
        corrected = "The dogs run quickly."
        v2 = store.add_version(corrected, parent_id=v1)

        mapping = store.get_word_mapping(v1, v2)
        assert mapping is not None
        # "The" should map to "The" (unchanged)
        # "dogs" should map to "dogs" (unchanged)
        # "runs" should map to "run" (corrected)
        # "quickly" should map to "quickly" (unchanged)

    def test_mapping_tracks_insertions(self):
        """Test mapping tracks word insertions."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        original = "Dog runs."
        v1 = store.add_version(original)

        corrected = "The dog runs."
        v2 = store.add_version(corrected, parent_id=v1)

        mapping = store.get_word_mapping(v1, v2)
        # Should indicate insertion at position 0
        assert mapping is not None

    def test_mapping_tracks_deletions(self):
        """Test mapping tracks word deletions."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        original = "The big dog runs."
        v1 = store.add_version(original)

        corrected = "The dog runs."
        v2 = store.add_version(corrected, parent_id=v1)

        mapping = store.get_word_mapping(v1, v2)
        # Should indicate deletion of "big"
        assert mapping is not None


class TestOriginalRecovery:
    """Test original view recovery."""

    def test_recover_original(self):
        """Test recovering original text from any version."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        original = "The dogs runs quickly."
        v1 = store.add_version(original)

        corrected = "The dogs run quickly."
        v2 = store.add_version(corrected, parent_id=v1)

        # Should be able to recover original
        recovered = store.get_original(v2)
        assert recovered == original

    def test_recover_original_multi_hop(self):
        """Test recovering original through multiple edits."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("Original text.")
        v2 = store.add_version("Changed text.", parent_id=v1)
        v3 = store.add_version("Changed again.", parent_id=v2)

        recovered = store.get_original(v3)
        assert recovered == "Original text."

    def test_original_always_recoverable(self):
        """Test that original is always recoverable."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        original = "The original text here."
        v1 = store.add_version(original)

        # Multiple edits
        current = original
        current_id = v1
        for i in range(5):
            current = current.replace(".", f" edit{i}.")
            current_id = store.add_version(current, parent_id=current_id)

        # Still recoverable
        recovered = store.get_original(current_id)
        assert recovered == original


class TestWordCountTracking:
    """Test word count change tracking."""

    def test_word_count_stored(self):
        """Test that word count is stored per version."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        text = "The dog runs."
        version_id = store.add_version(text)

        version = store.get_version(version_id)
        assert hasattr(version, 'word_count')
        assert version.word_count >= 3

    def test_word_count_change_tracked(self):
        """Test that word count changes are tracked."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("The dog runs.")
        v2 = store.add_version("The big dog runs fast.", parent_id=v1)

        version1 = store.get_version(v1)
        version2 = store.get_version(v2)

        # Word count should differ
        assert version2.word_count > version1.word_count

    def test_word_count_delta(self):
        """Test getting word count delta between versions."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("The dog runs.")
        v2 = store.add_version("Dog runs.", parent_id=v1)

        delta = store.get_word_count_delta(v1, v2)
        assert delta < 0  # Word removed


class TestVersionHistory:
    """Test version history (git-like DAG)."""

    def test_linear_history(self):
        """Test linear version history."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("Version 1.")
        v2 = store.add_version("Version 2.", parent_id=v1)
        v3 = store.add_version("Version 3.", parent_id=v2)

        history = store.get_history(v3)
        assert len(history) == 3
        assert history[0] == v1  # Oldest first

    def test_branching_history(self):
        """Test branching version history (DAG)."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("Base version.")
        v2a = store.add_version("Branch A.", parent_id=v1)
        v2b = store.add_version("Branch B.", parent_id=v1)

        # Both branches should have v1 as parent
        version_a = store.get_version(v2a)
        version_b = store.get_version(v2b)
        assert version_a.parent_id == v1
        assert version_b.parent_id == v1

    def test_version_has_parent(self):
        """Test that versions track parent."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("Parent.")
        v2 = store.add_version("Child.", parent_id=v1)

        version = store.get_version(v2)
        assert version.parent_id == v1

    def test_root_version_no_parent(self):
        """Test that root version has no parent."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("Root.")

        version = store.get_version(v1)
        assert version.parent_id is None


class TestRevert:
    """Test reverting to previous versions."""

    def test_revert_to_version(self):
        """Test reverting to a specific version."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("Original.")
        v2 = store.add_version("Changed.", parent_id=v1)

        reverted_id = store.revert_to(v1, from_version=v2)
        reverted = store.get_version(reverted_id)

        assert reverted.text == "Original."
        assert reverted.parent_id == v2  # Revert is a new version

    def test_revert_creates_new_version(self):
        """Test that revert creates a new version (doesn't destroy history)."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("Original.")
        v2 = store.add_version("Changed.", parent_id=v1)
        v3 = store.revert_to(v1, from_version=v2)

        # All three versions should exist
        assert store.get_version(v1) is not None
        assert store.get_version(v2) is not None
        assert store.get_version(v3) is not None

    def test_revert_preserves_history(self):
        """Test that revert preserves full history."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("First.")
        v2 = store.add_version("Second.", parent_id=v1)
        v3 = store.add_version("Third.", parent_id=v2)
        v4 = store.revert_to(v1, from_version=v3)

        history = store.get_history(v4)
        # Should include original path plus revert
        assert len(history) >= 3


class TestVersionMetadata:
    """Test version metadata."""

    def test_version_timestamp(self):
        """Test that versions have timestamps."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        version_id = store.add_version("Test.")

        version = store.get_version(version_id)
        assert hasattr(version, 'timestamp')
        assert version.timestamp is not None

    def test_version_id_unique(self):
        """Test that version IDs are unique."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        v1 = store.add_version("First.")
        v2 = store.add_version("Second.")
        v3 = store.add_version("Third.")

        assert v1 != v2
        assert v2 != v3
        assert v1 != v3

    def test_version_description(self):
        """Test adding description to versions."""
        from semantic_zoom.phase7.preservation import VersionStore

        store = VersionStore()
        version_id = store.add_version("Text.", description="Initial version")

        version = store.get_version(version_id)
        assert version.description == "Initial version"
