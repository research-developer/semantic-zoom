"""NSM-60: Original preservation alongside corrections.

Provides version control for text with:
- Original and corrected versions with word ID mappings
- Original view always recoverable
- Word count change tracking
- Full version history (git-like DAG)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

# Lazy load spacy for tokenization
_nlp = None


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


@dataclass
class WordID:
    """A word with its ID and position.

    Attributes:
        word_id: Unique identifier for this word instance
        text: The word text
        position: Position in the version (0-indexed)
    """
    word_id: str
    text: str
    position: int


@dataclass
class WordMapping:
    """Mapping between words in two versions.

    Attributes:
        source_id: Word ID in source version
        target_id: Word ID in target version (None if deleted)
        change_type: 'unchanged', 'modified', 'inserted', 'deleted'
    """
    source_id: Optional[str]
    target_id: Optional[str]
    change_type: str


@dataclass
class Version:
    """A version of the text.

    Attributes:
        version_id: Unique identifier
        text: The full text
        word_ids: List of word IDs with positions
        word_count: Number of words
        parent_id: ID of parent version (None for root)
        timestamp: When this version was created
        description: Optional description of the change
    """
    version_id: str
    text: str
    word_ids: list[WordID]
    word_count: int
    parent_id: Optional[str]
    timestamp: datetime
    description: Optional[str] = None


def _generate_version_id() -> str:
    """Generate a unique version identifier."""
    return f"v_{uuid.uuid4().hex[:12]}"


def _generate_word_id() -> str:
    """Generate a unique word identifier."""
    return f"w_{uuid.uuid4().hex[:8]}"


def _tokenize_text(text: str) -> list[str]:
    """Tokenize text into words."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]


def _create_word_ids(text: str) -> list[WordID]:
    """Create word IDs for text."""
    words = _tokenize_text(text)
    return [
        WordID(
            word_id=_generate_word_id(),
            text=word,
            position=i,
        )
        for i, word in enumerate(words)
    ]


class VersionStore:
    """Store for text versions with DAG history.

    Provides version control functionality:
    - Add new versions with parent tracking
    - Get versions by ID
    - Compute word mappings between versions
    - Recover original text from any version
    - Revert to previous versions
    """

    def __init__(self):
        """Initialize empty version store."""
        self._versions: dict[str, Version] = {}
        self._root_id: Optional[str] = None

    def add_version(
        self,
        text: str,
        parent_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Add a new version.

        Args:
            text: The text content
            parent_id: ID of parent version (None for root)
            description: Optional description

        Returns:
            ID of the new version
        """
        version_id = _generate_version_id()
        word_ids = _create_word_ids(text)

        version = Version(
            version_id=version_id,
            text=text,
            word_ids=word_ids,
            word_count=len(word_ids),
            parent_id=parent_id,
            timestamp=datetime.now(),
            description=description,
        )

        self._versions[version_id] = version

        if parent_id is None:
            self._root_id = version_id

        return version_id

    def get_version(self, version_id: str) -> Optional[Version]:
        """Get a version by ID.

        Args:
            version_id: The version ID

        Returns:
            The version, or None if not found
        """
        return self._versions.get(version_id)

    def get_word_mapping(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[list[WordMapping]]:
        """Get word mapping between two versions.

        Uses LCS-based diff with modification detection.
        Adjacent deletion+insertion pairs are merged into 'modified'
        if the words are similar (share a common lemma).

        Args:
            source_id: Source version ID
            target_id: Target version ID

        Returns:
            List of word mappings, or None if versions not found
        """
        source = self.get_version(source_id)
        target = self.get_version(target_id)

        if source is None or target is None:
            return None

        mappings = []
        source_words = source.word_ids
        target_words = target.word_ids

        # Simple LCS-based mapping
        source_texts = [w.text for w in source_words]
        target_texts = [w.text for w in target_words]

        # Build mapping using longest common subsequence approach
        lcs = _compute_lcs(source_texts, target_texts)

        source_idx = 0
        target_idx = 0
        lcs_idx = 0

        while source_idx < len(source_words) or target_idx < len(target_words):
            if lcs_idx < len(lcs):
                # Check if current positions match LCS
                source_matches = (
                    source_idx < len(source_words) and
                    source_texts[source_idx] == lcs[lcs_idx]
                )
                target_matches = (
                    target_idx < len(target_words) and
                    target_texts[target_idx] == lcs[lcs_idx]
                )

                if source_matches and target_matches:
                    # Unchanged word
                    mappings.append(WordMapping(
                        source_id=source_words[source_idx].word_id,
                        target_id=target_words[target_idx].word_id,
                        change_type="unchanged",
                    ))
                    source_idx += 1
                    target_idx += 1
                    lcs_idx += 1
                elif not source_matches and source_idx < len(source_words):
                    # Deleted from source
                    mappings.append(WordMapping(
                        source_id=source_words[source_idx].word_id,
                        target_id=None,
                        change_type="deleted",
                    ))
                    source_idx += 1
                elif not target_matches and target_idx < len(target_words):
                    # Inserted in target
                    mappings.append(WordMapping(
                        source_id=None,
                        target_id=target_words[target_idx].word_id,
                        change_type="inserted",
                    ))
                    target_idx += 1
            else:
                # Past LCS
                if source_idx < len(source_words):
                    mappings.append(WordMapping(
                        source_id=source_words[source_idx].word_id,
                        target_id=None,
                        change_type="deleted",
                    ))
                    source_idx += 1
                if target_idx < len(target_words):
                    mappings.append(WordMapping(
                        source_id=None,
                        target_id=target_words[target_idx].word_id,
                        change_type="inserted",
                    ))
                    target_idx += 1

        # Post-process to detect modifications (deletion+insertion of similar words)
        mappings = _detect_modifications(mappings, source_words, target_words)

        return mappings

    def get_original(self, version_id: str) -> Optional[str]:
        """Get the original (root) text from any version.

        Args:
            version_id: ID of a version in the chain

        Returns:
            The original root text, or None if not found
        """
        version = self.get_version(version_id)
        if version is None:
            return None

        # Walk back to root
        current = version
        while current.parent_id is not None:
            parent = self.get_version(current.parent_id)
            if parent is None:
                break
            current = parent

        return current.text

    def get_word_count_delta(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[int]:
        """Get word count difference between versions.

        Args:
            source_id: Source version ID
            target_id: Target version ID

        Returns:
            Delta (target - source), or None if versions not found
        """
        source = self.get_version(source_id)
        target = self.get_version(target_id)

        if source is None or target is None:
            return None

        return target.word_count - source.word_count

    def get_history(self, version_id: str) -> list[str]:
        """Get version history (path to root).

        Args:
            version_id: ID of a version

        Returns:
            List of version IDs from root to version_id
        """
        version = self.get_version(version_id)
        if version is None:
            return []

        history = []
        current = version

        while current is not None:
            history.append(current.version_id)
            if current.parent_id:
                current = self.get_version(current.parent_id)
            else:
                current = None

        # Reverse to get root-first order
        return list(reversed(history))

    def revert_to(
        self,
        target_id: str,
        from_version: str,
    ) -> str:
        """Revert to a previous version.

        Creates a new version with the target's text but
        with from_version as parent (preserves history).

        Args:
            target_id: ID of version to revert to
            from_version: ID of current version (new parent)

        Returns:
            ID of the new revert version
        """
        target = self.get_version(target_id)
        if target is None:
            raise ValueError(f"Target version {target_id} not found")

        return self.add_version(
            text=target.text,
            parent_id=from_version,
            description=f"Reverted to {target_id}",
        )

    def get_branches(self, version_id: str) -> list[str]:
        """Get all child versions (branches) from a version.

        Args:
            version_id: ID of the version

        Returns:
            List of child version IDs
        """
        return [
            v.version_id
            for v in self._versions.values()
            if v.parent_id == version_id
        ]


def _compute_lcs(seq1: list[str], seq2: list[str]) -> list[str]:
    """Compute longest common subsequence of two sequences."""
    m, n = len(seq1), len(seq2)

    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(lcs))


def _are_words_similar(word1: str, word2: str) -> bool:
    """Check if two words are morphologically related (same lemma).

    Uses spaCy lemmatization to determine if words share a common root.
    Examples: "runs" and "run", "dogs" and "dog".
    """
    if word1.lower() == word2.lower():
        return True

    nlp = _get_nlp()
    doc1 = nlp(word1)
    doc2 = nlp(word2)

    if doc1 and doc2:
        lemma1 = doc1[0].lemma_.lower()
        lemma2 = doc2[0].lemma_.lower()
        return lemma1 == lemma2

    return False


def _detect_modifications(
    mappings: list[WordMapping],
    source_words: list[WordID],
    target_words: list[WordID],
) -> list[WordMapping]:
    """Post-process mappings to detect modifications.

    Identifies contiguous blocks of deletions/insertions and matches
    morphologically related words (same lemma) as 'modified'.

    Args:
        mappings: Initial mappings from LCS algorithm
        source_words: Source version word IDs
        target_words: Target version word IDs

    Returns:
        Mappings with modifications detected
    """
    if len(mappings) < 2:
        return mappings

    # Build lookup dictionaries for word texts
    source_id_to_text = {w.word_id: w.text for w in source_words}
    target_id_to_text = {w.word_id: w.text for w in target_words}

    result = []
    i = 0

    while i < len(mappings):
        current = mappings[i]

        # If not a deletion, just add and continue
        if current.change_type != "deleted":
            result.append(current)
            i += 1
            continue

        # Collect contiguous change block (deletions followed by insertions)
        deletions = []
        insertions = []

        # Collect all consecutive deletions
        j = i
        while j < len(mappings) and mappings[j].change_type == "deleted":
            deletions.append(mappings[j])
            j += 1

        # Collect all consecutive insertions
        while j < len(mappings) and mappings[j].change_type == "inserted":
            insertions.append(mappings[j])
            j += 1

        # If no insertions, just emit deletions
        if not insertions:
            result.extend(deletions)
            i = j
            continue

        # Match deletions to insertions by lemma similarity
        used_deletions = set()
        used_insertions = set()
        modifications = []

        for d_idx, deletion in enumerate(deletions):
            source_text = source_id_to_text.get(deletion.source_id, "")
            if not source_text:
                continue

            for ins_idx, insertion in enumerate(insertions):
                if ins_idx in used_insertions:
                    continue

                target_text = target_id_to_text.get(insertion.target_id, "")
                if not target_text:
                    continue

                if _are_words_similar(source_text, target_text):
                    modifications.append(WordMapping(
                        source_id=deletion.source_id,
                        target_id=insertion.target_id,
                        change_type="modified",
                    ))
                    used_deletions.add(d_idx)
                    used_insertions.add(ins_idx)
                    break

        # Add unmatched deletions
        for d_idx, deletion in enumerate(deletions):
            if d_idx not in used_deletions:
                result.append(deletion)

        # Add modifications
        result.extend(modifications)

        # Add unmatched insertions
        for ins_idx, insertion in enumerate(insertions):
            if ins_idx not in used_insertions:
                result.append(insertion)

        i = j

    return result
