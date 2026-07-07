"""Pericope classification and Aland number handling.

Maps token sequences to pericope IDs from the Aland Synopsis table
(constants.ALAND_PERICOPES) and determines tradition type for each pericope.

Tradition classification rules:
  - "triple": pericope appears in Matthew, Mark, AND Luke
  - "double": appears in exactly Matthew + Luke (no Mark) — potential Q material
  - "mark_unique": appears only in Mark (Markan Sondergut)
  - "matthean_unique": appears only in Matthew (M-material)
  - "lukan_unique": appears only in Luke (L-material)
  - "johannine": appears only in John (not used in 3-gospel analysis)
  - Combinations like (Matthew+Mark only) are uncommon and labeled by books.

The tradition classification is foundational to the research:
  - Triple tradition → Matthew/Mark/Luke parallels (Q-reconstruction supervision)
  - Double tradition → Matthew/Luke parallels (proto-Q material)
  - Unique material → authorship/editorial style baselines
"""

from __future__ import annotations

from synoptiq.utils.constants import ALAND_PERICOPES, PERICOPE_GENRES
from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import Book, Genre, PericopeAlignment, TokenRecord, Tradition

_LOG = get_logger(__name__)

# ── Tradition classification ───────────────────────────────────────────────────


def classify_tradition(books: frozenset[Book]) -> Tradition:
    """Classify the tradition type of a pericope given which books contain it.

    Args:
        books: Frozenset of canonical book names that contain this pericope.

    Returns:
        Tradition type string.

    Example:
        >>> classify_tradition(frozenset({"Matthew", "Mark", "Luke"}))
        'triple'
        >>> classify_tradition(frozenset({"Matthew", "Luke"}))
        'double'
        >>> classify_tradition(frozenset({"Mark"}))
        'mark_unique'
    """
    b: set[str] = set(books)
    if b >= {"Matthew", "Mark", "Luke"}:
        return "triple"
    if b == {"Matthew", "Luke"}:
        return "double"
    if b == {"Mark"}:
        return "mark_unique"
    if b == {"Matthew"}:
        return "matthean_unique"
    if b == {"Luke"}:
        return "lukan_unique"
    if b == {"Matthew", "Mark"} or b == {"Mark", "Luke"}:
        return "triple"
    if b == {"Matthew", "Mark", "John"} or b == {"Mark", "Luke", "John"}:
        return "triple"
    # Johannine or complex overlap
    return "lukan_unique"  # Fallback — should not occur for NT synoptics


def get_pericope_books(pericope_id: str) -> frozenset[Book]:
    """Return the set of books that contain a given Aland pericope.

    Args:
        pericope_id: Aland pericope number string (e.g., "058").

    Returns:
        Frozenset of canonical book names present in this pericope.

    Raises:
        KeyError: If pericope_id is not in the Aland table.
    """
    entry = ALAND_PERICOPES[pericope_id]
    return frozenset(
        book  # type: ignore[misc]
        for book, verse_range in entry.items()
        if verse_range is not None
    )


def get_genre(pericope_id: str) -> Genre:
    """Return the genre classification for a pericope.

    Args:
        pericope_id: Aland pericope number string.

    Returns:
        Genre string from PERICOPE_GENRES table, or "other" if not classified.
    """
    return PERICOPE_GENRES.get(pericope_id, "other")  # type: ignore[return-value]


# ── Grouping tokens into pericopes ───────────────────────────────────────────


def group_tokens_by_pericope(
    tokens: list[TokenRecord],
) -> dict[str, dict[Book, list[TokenRecord]]]:
    """Group a flat token list into a nested pericope→book→token structure.

    Tokens with ``pericope_id = None`` are placed into a special
    ``"__unassigned__"`` bucket.

    Args:
        tokens: Flat list of TokenRecord dicts (all books mixed together).

    Returns:
        Dict: pericope_id → {book → [tokens in order]}.
        Tokens with pericope_id=None → in ``"__unassigned__"``.
    """
    result: dict[str, dict[Book, list[TokenRecord]]] = {}

    for token in tokens:
        pid = token.get("pericope_id") or "__unassigned__"
        book: Book = token["book"]  # type: ignore[assignment]

        if pid not in result:
            result[pid] = {}
        if book not in result[pid]:
            result[pid][book] = []
        result[pid][book].append(token)

    _LOG.info(
        "tokens grouped by pericope",
        extra={
            "n_pericopes": len(result),
            "n_unassigned": len(result.get("__unassigned__", {})),
        },
    )
    return result


def build_pericope_alignments(
    grouped: dict[str, dict[Book, list[TokenRecord]]],
    alignment_pairs: dict[tuple[str, Book, Book], list[tuple[int | None, int | None]]],
) -> list[PericopeAlignment]:
    """Build PericopeAlignment dicts from grouped tokens and pre-computed alignments.

    Args:
        grouped: Output of group_tokens_by_pericope().
        alignment_pairs: Dict of (pericope_id, book_a, book_b) → alignment pairs.

    Returns:
        List of PericopeAlignment dicts ready for Corpus construction.
    """
    alignments: list[PericopeAlignment] = []

    for pericope_id, book_tokens in grouped.items():
        if pericope_id == "__unassigned__":
            continue

        books = list(book_tokens.keys())
        books_frozen = frozenset(books)  # type: ignore[arg-type]
        tradition = classify_tradition(books_frozen)
        genre = get_genre(pericope_id)

        # Collect alignment pairs for this pericope
        pericope_alignments: dict[tuple[Book, Book], list[tuple[int | None, int | None]]] = {}
        for (pid, book_a, book_b), pairs in alignment_pairs.items():
            if pid == pericope_id:
                pericope_alignments[(book_a, book_b)] = pairs

        alignment = PericopeAlignment(
            pericope_id=pericope_id,
            tradition=tradition,
            genre=genre,
            books=books,
            tokens=book_tokens,
            alignment=pericope_alignments,
        )
        alignments.append(alignment)

    # Sort by pericope ID for deterministic ordering
    alignments.sort(key=lambda a: a["pericope_id"])

    _LOG.info(
        "pericope alignments built",
        extra={
            "n_pericopes": len(alignments),
            "n_triple": sum(1 for a in alignments if a["tradition"] == "triple"),
            "n_double": sum(1 for a in alignments if a["tradition"] == "double"),
        },
    )
    return alignments


# ── Pericope classifier convenience class ────────────────────────────────────


class PericopeClassifier:
    """Assigns tradition type and genre to pericopes via the Aland table.

    Provides a higher-level interface over the module-level functions
    for use in the corpus builder pipeline.

    Example:
        >>> clf = PericopeClassifier()
        >>> clf.classify("058")
        ('triple', 'narrative')
        >>> clf.get_books("058")
        frozenset({'Matthew', 'Mark', 'Luke', 'John'})
    """

    def classify(self, pericope_id: str) -> tuple[Tradition, Genre]:
        """Return (tradition, genre) for a pericope ID.

        Args:
            pericope_id: Aland pericope number.

        Returns:
            Tuple of (Tradition, Genre).
        """
        books = get_pericope_books(pericope_id)
        tradition = classify_tradition(books)
        genre = get_genre(pericope_id)
        return tradition, genre

    def get_books(self, pericope_id: str) -> frozenset[Book]:
        """Return the books containing a pericope.

        Args:
            pericope_id: Aland pericope number.

        Returns:
            Frozenset of canonical book names.
        """
        return get_pericope_books(pericope_id)

    def all_triple_tradition(self) -> list[str]:
        """Return sorted list of all triple tradition pericope IDs."""
        return sorted(
            pid
            for pid in ALAND_PERICOPES
            if classify_tradition(get_pericope_books(pid)) == "triple"
        )

    def all_double_tradition(self) -> list[str]:
        """Return sorted list of all double tradition pericope IDs."""
        return sorted(
            pid
            for pid in ALAND_PERICOPES
            if classify_tradition(get_pericope_books(pid)) == "double"
        )
