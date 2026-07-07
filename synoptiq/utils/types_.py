"""Shared type aliases, TypedDict definitions, and Protocols.

All types used across the SynoptiQ codebase are defined here
to ensure consistency and enable static type checking (mypy strict).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, Protocol, TypeAlias, TypedDict, runtime_checkable

# ── Core identifiers ──────────────────────────────────────────────────────────

Book: TypeAlias = Literal["Matthew", "Mark", "Luke", "John"]

Tradition: TypeAlias = Literal["triple", "double", "mark_unique", "matthean_unique", "lukan_unique"]

Genre: TypeAlias = Literal["narrative", "discourse", "wisdom", "passion", "other"]

# ── Token-level data ──────────────────────────────────────────────────────────


class TokenRecord(TypedDict):
    """A single token with full morphological annotation.

    Each token in the corpus carries its surface form, lemma,
    part-of-speech tag, morphological parsing, and book/verse location.
    This is the atomic unit of all SynoptiQ analyses.
    """

    token_id: str  # e.g., "Matt.1.1.3"
    book: Book
    chapter: int
    verse: int
    position: int  # 0-indexed token position within the verse
    text: str  # Surface form (polytonic Greek)
    normalized: str  # De-accented, lowercased form
    lemma: str  # Dictionary headword form
    pos: str  # Part-of-speech code (MorphGNT CCAT tagset)
    morph: str  # Full morphological string (person-tense-voice-mood-case-number-gender-degree)
    pericope_id: str | None  # Aland Synopsis pericope number, None if unaligned
    is_punctuation: bool  # True for punctuation tokens


# ── Pericope alignment ────────────────────────────────────────────────────────


class PericopeAlignment(TypedDict):
    """An aligned set of parallel Gospel passages for one pericope.

    Contains the tokens for each gospel containing this pericope,
    plus pairwise token alignment matrices.
    """

    pericope_id: str
    tradition: Tradition
    genre: Genre
    books: list[Book]
    tokens: dict[Book, list[TokenRecord]]
    # (idx_in_A, idx_in_B) — None means gap
    alignment: dict[tuple[Book, Book], list[tuple[int | None, int | None]]]


# ── Data split results ────────────────────────────────────────────────────────


class SplitResult(TypedDict):
    """Result of a stratified train/val/test split."""

    train_ids: list[str]  # Pericope IDs in train set
    val_ids: list[str]  # Pericope IDs in validation set
    test_ids: list[str]  # Pericope IDs in test set


# ── Morphological parsing ─────────────────────────────────────────────────────


class MorphRecord(TypedDict):
    """Morphological annotation for a single token (from MorphGNT)."""

    book: Book
    chapter: int
    verse: int
    position: int
    pos: str  # CCAT POS code
    parsing: str  # Full 8-char parsing string
    text: str  # Original text (with punctuation)
    word: str  # Word with punctuation stripped
    normalized: str  # Normalized inflected form
    lemma: str  # Dictionary form


class ConlluToken(TypedDict):
    """A single token in CoNLL-U format (from PROIEL UD)."""

    id: int  # Token ID within sentence
    form: str  # Surface form
    lemma: str  # Lemma
    upos: str  # Universal POS tag
    xpos: str  # Language-specific POS tag
    feats: dict[str, str]  # Morphological features
    head: int  # Head token ID (0 = root)
    deprel: str  # Dependency relation
    deps: str  # Enhanced dependencies
    misc: str  # Miscellaneous


# ── Protocols ─────────────────────────────────────────────────────────────────


@runtime_checkable
class TrainableModel(Protocol):
    """Protocol for any trainable SynoptiQ model."""

    def train(self) -> dict[str, list[float]]:
        """Run training loop, return metric history."""
        ...

    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation/test set."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint from disk."""
        ...


@runtime_checkable
class CorpusLike(Protocol):
    """Protocol for any corpus-like object."""

    @property
    def n_tokens(self) -> int:
        """Total token count."""
        ...

    @property
    def n_pericopes(self) -> int:
        """Total pericope count."""
        ...

    def iter_pericopes(
        self,
        *,
        tradition: Tradition | None = None,
        books: tuple[Book, ...] | None = None,
    ) -> Iterator[PericopeAlignment]:
        """Iterate over pericopes."""
        ...
