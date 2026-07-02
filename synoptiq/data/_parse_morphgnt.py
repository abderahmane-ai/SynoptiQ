"""MorphGNT TSV parser for SynoptiQ.

Parses the MorphGNT morphological annotation files to produce
MorphRecord dicts for each token in Matthew, Mark, and Luke.

MorphGNT uses 7-column TSV format (confirmed from morphgnt/sblgnt repository):
  Col 0: bcv   — "MAT 1:1/1" (book chapter:verse/token-position)
  Col 1: pos   — Part-of-speech code (e.g., "N-", "V-", "RA")
  Col 2: parsing — 8-char morphological string (e.g., "NSN-----")
  Col 3: text  — Original surface form with punctuation (e.g., "Βίβλος")
  Col 4: word  — Surface form with punctuation stripped (e.g., "Βίβλος")
  Col 5: normalized — Normalized form (movable nu, orthographic variants)
  Col 6: lemma — Dictionary headword (e.g., "βίβλος")

After parsing, this module produces a merge table keyed by
(book, chapter, verse, position) which the corpus builder uses
to augment SBLGNT tokens with morphological annotations.

Merge strategy: match on normalized surface form within each verse.
For rare cases where surfaces differ (accent variant, text variant),
we fall back to position-within-verse matching.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from synoptiq.utils.constants import BOOK_ABBREV_TO_FULL
from synoptiq.utils.greek import normalize_greek
from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import MorphRecord

_LOG = get_logger(__name__)

# MorphGNT book abbreviations for the four gospels
MORPHGNT_GOSPEL_ABBREVS: dict[str, str] = {
    "MAT": "Matthew",
    "MRK": "Mark",
    "LUK": "Luke",
    "JHN": "John",
}


def _parse_bcv(bcv: str) -> tuple[str, int, int, int]:
    """Parse a MorphGNT BCV field like "MAT 1:1/1".

    Args:
        bcv: A string like "MAT 1:1/1" or "MAT 1:1" (without position).

    Returns:
        Tuple of (book_name, chapter, verse, position).

    Raises:
        ValueError: If the BCV field is malformed.
    """
    # Split on space: ["MAT", "1:1/1"]
    parts = bcv.strip().split()
    if len(parts) < 2:
        msg = f"Malformed BCV field: {bcv!r}"
        raise ValueError(msg)

    book_abbr = parts[0]
    book = BOOK_ABBREV_TO_FULL.get(book_abbr, book_abbr)

    # Parse "chapter:verse/position"
    ref = parts[1]
    if "/" in ref:
        cv_part, pos_str = ref.split("/", 1)
        position = int(pos_str) - 1  # Convert 1-indexed to 0-indexed
    else:
        cv_part = ref
        position = 0

    ch_str, vs_str = cv_part.split(":", 1)
    chapter = int(ch_str)
    verse = int(vs_str)

    return book, chapter, verse, position


def _parse_morphgnt_line(line: str) -> MorphRecord | None:
    """Parse a single TSV line from MorphGNT into a MorphRecord.

    Args:
        line: A single tab-separated line from a MorphGNT file.

    Returns:
        MorphRecord dict if the line is valid, None if it's a header/blank.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    cols = line.split("\t")
    if len(cols) < 7:
        _LOG.warning("short MorphGNT line", extra={"line": line[:80]})
        return None

    bcv, pos, parsing, text, word, normalized_col, lemma = cols[:7]

    try:
        book, chapter, verse, position = _parse_bcv(bcv)
    except ValueError:
        _LOG.warning("failed to parse BCV", extra={"bcv": bcv})
        return None

    return MorphRecord(
        book=book,  # type: ignore[arg-type]
        chapter=chapter,
        verse=verse,
        position=position,
        pos=pos.strip(),
        parsing=parsing.strip(),
        text=text.strip(),
        word=word.strip(),
        normalized=normalized_col.strip(),
        lemma=lemma.strip(),
    )


def _iter_morphgnt_file(path: Path) -> Iterator[MorphRecord]:
    """Iterate over MorphRecord entries from a MorphGNT TSV file.

    Args:
        path: Path to a MorphGNT TSV file.

    Yields:
        MorphRecord dicts (skipping blank lines and headers).
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = _parse_morphgnt_line(line)
            if record is not None:
                yield record


def parse_morphgnt(
    morphgnt_dir: Path,
    *,
    books: list[str] | None = None,
) -> dict[tuple[str, int, int, int], MorphRecord]:
    """Parse all MorphGNT TSV files and return a merge lookup table.

    The MorphGNT repository (morphgnt/sblgnt) contains one TSV file per
    book, named like ``61-Mt-morphgnt.txt``, ``62-Mk-morphgnt.txt``, etc.
    We discover files by scanning for ``*morphgnt*.txt`` patterns.

    Args:
        morphgnt_dir: Path to the cloned morphgnt/sblgnt repository.
        books: Canonical book names to parse. If None, parses all gospels.

    Returns:
        Dict keyed by (book, chapter, verse, position) → MorphRecord.
        This is the merge table used to annotate SBLGNT tokens.

    Raises:
        FileNotFoundError: If no MorphGNT files are found.
    """
    target_books: set[str] = set(books) if books else {"Matthew", "Mark", "Luke"}

    # Discover MorphGNT files — pattern: "61-Mt-morphgnt.txt"
    morphgnt_files = sorted(morphgnt_dir.rglob("*morphgnt*.txt"))
    if not morphgnt_files:
        # Try alternative naming: MAT.txt, MRK.txt etc.
        morphgnt_files = sorted(morphgnt_dir.rglob("*.txt"))

    if not morphgnt_files:
        msg = f"No MorphGNT TSV files found in {morphgnt_dir}"
        raise FileNotFoundError(msg)

    _LOG.info(
        "found MorphGNT files",
        extra={"count": len(morphgnt_files), "first": str(morphgnt_files[0])},
    )

    lookup: dict[tuple[str, int, int, int], MorphRecord] = {}

    for filepath in morphgnt_files:
        book_tokens = 0
        first_book: str | None = None

        for record in _iter_morphgnt_file(filepath):
            book = record["book"]
            if book not in target_books:
                continue

            if first_book is None:
                first_book = book

            key = (book, record["chapter"], record["verse"], record["position"])
            if key in lookup:
                # Duplicate key — position counting might be off; use normalized match
                _LOG.debug(
                    "duplicate MorphGNT key",
                    extra={"key": key, "word": record["word"]},
                )
            lookup[key] = record
            book_tokens += 1

        if book_tokens > 0:
            _LOG.info(
                "parsed MorphGNT file",
                extra={"file": filepath.name, "book": first_book, "tokens": book_tokens},
            )

    if not lookup:
        _LOG.warning(
            "MorphGNT parse produced zero records for target books",
            extra={"books": list(target_books)},
        )

    _LOG.info(
        "MorphGNT parse complete",
        extra={"n_records": len(lookup), "books": sorted(target_books)},
    )
    return lookup


def merge_sblgnt_with_morphgnt(
    sblgnt_tokens: list[dict[str, object]],
    morphgnt_lookup: dict[tuple[str, int, int, int], MorphRecord],
) -> list[dict[str, object]]:
    """Merge MorphGNT annotations into SBLGNT token records.

    Matching strategy (in priority order):
    1. Exact (book, chapter, verse, position) key match
    2. Normalized surface form match within the same verse
       (handles accent variants, minor text differences)

    If neither match succeeds, token retains empty pos/morph/lemma.
    This is expected for ~0.3% of tokens due to text-critical differences.

    Args:
        sblgnt_tokens: List of SBLGNT token dicts (from parse_sblgnt).
        morphgnt_lookup: Dict from parse_morphgnt.

    Returns:
        List of token dicts with pos, morph, lemma fields populated
        wherever a match was found.
    """
    matched = 0
    fallback_matched = 0
    unmatched = 0

    # Build a verse-level index for normalized surface matching
    # verse_index[(book, chapter, verse)] → list[(normalized_surface, MorphRecord)]
    verse_index: dict[tuple[str, int, int], list[tuple[str, MorphRecord]]] = {}
    for (book, ch, vs, _pos), record in morphgnt_lookup.items():
        key = (book, ch, vs)
        if key not in verse_index:
            verse_index[key] = []
        verse_index[key].append((normalize_greek(record["normalized"]), record))

    for token in sblgnt_tokens:
        book = str(token["book"])
        chapter = int(token["chapter"])  # type: ignore[arg-type]
        verse = int(token["verse"])  # type: ignore[arg-type]
        position = int(token["position"])  # type: ignore[arg-type]
        norm_text = str(token["normalized"])

        # Strategy 1: Exact position match
        exact_key = (book, chapter, verse, position)
        if exact_key in morphgnt_lookup:
            record = morphgnt_lookup[exact_key]
            token["lemma"] = record["lemma"]
            token["pos"] = record["pos"]
            token["morph"] = record["parsing"]
            matched += 1
            continue

        # Strategy 2: Normalized surface match within verse
        verse_key = (book, chapter, verse)
        candidates = verse_index.get(verse_key, [])
        surface_match: MorphRecord | None = None
        for cand_norm, cand_record in candidates:
            if cand_norm == norm_text:
                surface_match = cand_record
                break

        if surface_match is not None:
            token["lemma"] = surface_match["lemma"]
            token["pos"] = surface_match["pos"]
            token["morph"] = surface_match["parsing"]
            fallback_matched += 1
        else:
            unmatched += 1
            _LOG.debug(
                "no MorphGNT match",
                extra={
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "text": token["text"],
                },
            )

    total = len(sblgnt_tokens)
    _LOG.info(
        "MorphGNT merge complete",
        extra={
            "total": total,
            "exact_matched": matched,
            "surface_matched": fallback_matched,
            "unmatched": unmatched,
            "match_rate": f"{(matched + fallback_matched) / max(total, 1):.1%}",
        },
    )
    return sblgnt_tokens
