"""Synoptic-parallel lookup via the Aland pericope table.

Given a Gospel verse, find the Aland Synopsis pericope that contains it and return
the parallel verse ranges in the other Gospels — the feature no general Greek
reader offers. Ranges come from :data:`synoptiq.utils.constants.ALAND_PERICOPES`
(the project's bedrock table); :func:`synoptic_parallels` fetches the actual gold
text for each parallel through a :class:`~synoptiq.reader.gold.GoldReader`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from synoptiq.utils.constants import ALAND_PERICOPES, VerseRange

if TYPE_CHECKING:
    from synoptiq.reader.gold import GoldReader, ReadResult


def _contains(rng: VerseRange, chapter: int, verse: int) -> bool:
    """True if ``(chapter, verse)`` falls within the inclusive range ``rng``."""
    (c1, v1), (c2, v2) = rng
    return (c1, v1) <= (chapter, verse) <= (c2, v2)


def find_pericope(book: str, chapter: int, verse: int) -> str | None:
    """Return the Aland pericope id containing ``book chapter:verse`` (or None).

    Example:
        >>> find_pericope("Mark", 1, 9)  # Baptism of Jesus
        '008'
    """
    for pid, books in ALAND_PERICOPES.items():
        rng = books.get(book)
        if rng is not None and _contains(rng, chapter, verse):
            return pid
    return None


def parallel_ranges(book: str, chapter: int, verse: int) -> dict[str, VerseRange]:
    """Return ``{other_book: verse_range}`` for every parallel of the containing pericope.

    The queried book is excluded; gospels where the pericope is absent (``None``) are
    omitted. Empty if the verse belongs to no catalogued pericope.
    """
    pid = find_pericope(book, chapter, verse)
    if pid is None:
        return {}
    return {
        other: rng
        for other, rng in ALAND_PERICOPES[pid].items()
        if other != book and rng is not None
    }


def format_range(rng: VerseRange) -> str:
    """Render a verse range compactly (``((3, 23), (3, 38))`` → ``"3:23-38"``)."""
    (c1, v1), (c2, v2) = rng
    if (c1, v1) == (c2, v2):
        return f"{c1}:{v1}"
    if c1 == c2:
        return f"{c1}:{v1}-{v2}"
    return f"{c1}:{v1}-{c2}:{v2}"


def synoptic_parallels(
    reader: GoldReader,
    book: str,
    chapter: int,
    verse: int,
) -> dict[str, ReadResult]:
    """Resolve the actual gold text of each synoptic parallel to ``book chapter:verse``.

    Args:
        reader: A gold reader over a corpus containing the parallel books (the GNT).
        book: The queried Gospel.
        chapter: Chapter number.
        verse: Verse number.

    Returns:
        ``{other_book: ReadResult}`` for each parallel with resolvable text.
    """
    out: dict[str, ReadResult] = {}
    for other, rng in parallel_ranges(book, chapter, verse).items():
        (c1, v1), (c2, v2) = rng
        result = reader.passage(other, (c1, v1), (c2, v2))
        if result.words:
            out[other] = result
    return out
