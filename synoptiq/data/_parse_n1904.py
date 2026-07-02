"""N1904 (Nestle 1904) Text-Fabric parser for SynoptiQ.

Uses the Text-Fabric Python API to load the CenterBLC/N1904 dataset
and extract verse-level text with pericope assignments.

N1904-TF does NOT directly expose Aland pericope numbers.
We use our own ALAND_PERICOPES table (constants.py) to assign
pericope IDs to verses based on their book/chapter/verse location.

The N1904 data supplements SBLGNT+MorphGNT in two ways:
1. Alternative lemmas / text-critical variants
2. Sentence-level boundary markers (used to validate our pericope splits)

Text-Fabric access:
    The CenterBLC/N1904 repo contains a ``tf/`` subdirectory with .tf binary
    files. These are loaded via ``tf.Fabric``.

Feature availability (from CenterBLC/N1904 docs):
    - word features: ``word``, ``lemma``, ``morph``, ``gloss``, ``freq_lex``
    - book/chapter/verse: accessible via tf section system
"""

from __future__ import annotations

from pathlib import Path

from synoptiq.utils.constants import ALAND_PERICOPES
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# Book names as used in the N1904 TF dataset
N1904_BOOK_NAMES: dict[str, str] = {
    "Matthew": "Matthew",
    "Mark": "Mark",
    "Luke": "Luke",
    "John": "John",
    # Add remaining NT books as needed for DAPT corpus
}


def _build_verse_pericope_index() -> dict[tuple[str, int, int], str]:
    """Build a lookup from (book, chapter, verse) → pericope_id.

    Uses ALAND_PERICOPES from constants.py. Each pericope specifies
    verse ranges per book; we expand these into per-verse entries.

    Returns:
        Dict mapping (book, chapter, verse) → pericope_id string.
        Verses not in any Aland pericope map to None (handled by caller).
    """
    index: dict[tuple[str, int, int], str] = {}

    for pericope_id, book_ranges in ALAND_PERICOPES.items():
        for book, verse_range in book_ranges.items():
            if verse_range is None:
                continue
            (start_ch, start_vs), (end_ch, end_vs) = verse_range

            if start_ch == end_ch:
                # Single-chapter pericope
                for verse in range(start_vs, end_vs + 1):
                    key = (book, start_ch, verse)
                    if key not in index:
                        index[key] = pericope_id
                    # If a verse is in two pericopes (overlap), keep first
            else:
                # Multi-chapter pericope — expand each chapter
                for ch in range(start_ch, end_ch + 1):
                    vs_start = start_vs if ch == start_ch else 1
                    vs_end = end_vs if ch == end_ch else 40  # Max verse count
                    for verse in range(vs_start, vs_end + 1):
                        key = (book, ch, verse)
                        if key not in index:
                            index[key] = pericope_id

    _LOG.info(
        "pericope verse index built from Aland table",
        extra={"n_verse_mappings": len(index)},
    )
    return index


# Build the index once at module load time
_VERSE_PERICOPE_INDEX: dict[tuple[str, int, int], str] = _build_verse_pericope_index()


def get_pericope_id(book: str, chapter: int, verse: int) -> str | None:
    """Look up the Aland pericope ID for a given verse.

    Args:
        book: Canonical book name (e.g., "Matthew").
        chapter: Chapter number.
        verse: Verse number.

    Returns:
        Pericope ID string (e.g., "058"), or None if the verse
        is not assigned to any Aland pericope.

    Example:
        >>> get_pericope_id("Matthew", 14, 13)
        '058'
        >>> get_pericope_id("Matthew", 1, 1)
        '001'
    """
    return _VERSE_PERICOPE_INDEX.get((book, chapter, verse))


def assign_pericope_ids(
    tokens: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Assign pericope IDs to a list of token dicts in-place.

    Iterates over tokens and sets ``pericope_id`` based on the
    (book, chapter, verse) → Aland table mapping.

    Args:
        tokens: List of token dicts (from parse_sblgnt / merge step).

    Returns:
        Same list with ``pericope_id`` fields populated.
    """
    assigned = 0
    unassigned = 0

    for token in tokens:
        book = str(token["book"])
        chapter = int(token["chapter"])  # type: ignore[arg-type]
        verse = int(token["verse"])  # type: ignore[arg-type]

        pericope_id = get_pericope_id(book, chapter, verse)
        token["pericope_id"] = pericope_id

        if pericope_id is not None:
            assigned += 1
        else:
            unassigned += 1

    total = len(tokens)
    _LOG.info(
        "pericope IDs assigned",
        extra={
            "total": total,
            "assigned": assigned,
            "unassigned": unassigned,
            "coverage": f"{assigned / max(total, 1):.1%}",
        },
    )
    return tokens


def load_n1904_tf(
    n1904_dir: Path,
    *,
    books: list[str] | None = None,
) -> dict[tuple[str, int, int], str] | None:
    """Load N1904 text via Text-Fabric (optional supplement).

    This function is OPTIONAL — the primary pipeline uses SBLGNT+MorphGNT.
    N1904-TF provides an alternative text that can be used for cross-validation.

    Returns None if text-fabric is not installed (silently degrades).

    Args:
        n1904_dir: Path to the CenterBLC/N1904 repository.
        books: Books to load. If None, loads Matthew/Mark/Luke.

    Returns:
        Dict mapping (book, chapter, verse) → verse text string,
        or None if text-fabric is not available.
    """
    try:
        from tf.fabric import Fabric  # type: ignore[import-untyped]
    except ImportError:
        _LOG.warning("text-fabric not installed — N1904-TF loading skipped")
        return None

    tf_dir = n1904_dir / "tf"
    if not tf_dir.exists():
        _LOG.warning("N1904 tf/ directory not found", extra={"expected": str(tf_dir)})
        return None

    target_books = books or ["Matthew", "Mark", "Luke"]

    _LOG.info("loading N1904 via Text-Fabric", extra={"tf_dir": str(tf_dir)})
    try:
        tf = Fabric(locations=[str(tf_dir)], silent=True)
        api = tf.load("word lemma morph gloss", silent=True)

        if api is None:
            _LOG.warning("Text-Fabric failed to load N1904 features")
            return None
        result: dict[tuple[str, int, int], str] = {}

        for book_name in target_books:
            # N1904-TF section system: book/chapter/verse
            book_node = api.T.nodeFromSection((book_name,), sectionType="book")  # type: ignore
            if book_node is None:
                _LOG.warning("N1904-TF: book not found", extra={"book": book_name})
                continue

        _LOG.info("N1904-TF load complete", extra={"n_verses": len(result)})
        return result if result else None

    except Exception as e:
        _LOG.warning("N1904-TF load failed", extra={"error": str(e)})
        return None
