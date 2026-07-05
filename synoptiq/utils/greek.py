"""Ancient Greek text processing utilities.

Handles normalization, accent stripping, and Koine-specific features like
movable nu and sigma variants. (Nomina sacra special tokens are handled by
``synoptiq.utils.tokenization``.)

Koine Greek differs from classical Attic in several ways:
- Simpler sentence structure (more paratactic kai)
- Reduced use of the optative mood
- Semantic shift in particles (e.g., hina + subjunctive replacing infinitive)
- Nomina sacra abbreviations in manuscript tradition
- More consistent use of movable nu

Our normalization pipeline strips diacritics for comparison
while preserving lemma-level information for linguistic analysis.
"""

from __future__ import annotations

import re
from typing import Final
import unicodedata

# ── Constants ─────────────────────────────────────────────────────────────────

# Matches Greek characters including polytonic diacritics.
# Covers Greek and Coptic (U+0370–U+03FF) + Greek Extended (U+1F00–U+1FFF).
_GREEK_PATTERN: Final = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]+")

# Unicode combining marks that appear as diacritics on Greek letters.
# These are separated out by NFD normalization and filtered below.
_COMBINING_MARKS: Final = re.compile(r"[\u0300-\u036f\u0345]")

# Punctuation characters that appear adjacent to Greek words in SBLGNT.
_PUNCT_CHARS: Final[frozenset[str]] = frozenset(".,·;·!?\"'()[]{}—–-\u00b7\u037e\u0387")


# ── Public API ────────────────────────────────────────────────────────────────


def is_greek(text: str) -> bool:
    """Check whether a string contains Greek characters.

    Args:
        text: Any text string.

    Returns:
        True if the string contains at least one Greek character
        (basic Greek, Greek extended, or Greek/Coptic block).

    Example:
        >>> is_greek("λόγος")
        True
        >>> is_greek("word")
        False
    """
    return bool(_GREEK_PATTERN.search(text))


def is_punctuation_token(text: str) -> bool:
    """Return True if all non-whitespace characters are punctuation.

    Args:
        text: A token string, possibly with surrounding whitespace.

    Returns:
        True if the stripped text consists entirely of punctuation characters.

    Example:
        >>> is_punctuation_token("·")
        True
        >>> is_punctuation_token("λόγος·")
        False
    """
    stripped = text.strip()
    return bool(stripped) and all(ch in _PUNCT_CHARS for ch in stripped)


def strip_accents(text: str) -> str:
    """Remove all Greek diacritics: accents, breathings, iota subscript.

    Uses Unicode NFD decomposition to separate base characters from
    combining diacritical marks, then discards the marks.

    Args:
        text: Polytonic Greek text with full diacritics.

    Returns:
        De-accented text retaining only base Greek characters.

    Example:
        >>> strip_accents("ὁ λόγος τοῦ θεοῦ")
        'ο λογος του θεου'
        >>> strip_accents("ἐγένετο")
        'εγενετο'
    """
    nfd = unicodedata.normalize("NFD", text)
    return _COMBINING_MARKS.sub("", nfd)


def normalize_greek(
    text: str,
    *,
    lower: bool = True,
    strip_diacritics: bool = True,
    normalize_sigma: bool = True,
    normalize_movable_nu: bool = False,
) -> str:
    """Normalize Greek text for comparison and alignment.

    The default normalization pipeline:
    1. Strip leading/trailing whitespace
    2. Lowercase (Koine manuscripts were uncial; case is a modern editorial choice)
    3. Strip diacritics (breathings, accents, iota subscript)
    4. Normalize sigma variants: final sigma (ς) → medial sigma (σ)

    Args:
        text: Greek text string (polytonic or stripped).
        lower: If True, convert to lowercase.
        strip_diacritics: If True, remove breathings, accents, iota subscript.
        normalize_sigma: If True, convert final sigma (ς) to medial sigma (σ).
        normalize_movable_nu: If True, strip trailing nu from movable-nu words
            to collapse the two forms to one canonical representation.

    Returns:
        Normalized string suitable for lemma comparison and alignment scoring.

    Example:
        >>> normalize_greek("Ἐν ἀρχῇ ἦν ὁ λόγος")
        'εν αρχη ην ο λογος'
        >>> normalize_greek("ς")
        'σ'
    """
    result = text.strip()
    if lower:
        result = result.lower()
    if strip_diacritics:
        result = strip_accents(result)
    if normalize_sigma:
        result = result.replace("ς", "σ")
    if normalize_movable_nu:
        # Collapse movable nu: "εἶπεν" → "εἶπε" (de-accented already)
        for with_nu, without_nu in {
            "ειπεν": "ειπε",
            "ελεγεν": "ελεγε",
            "εστιν": "εστι",
            "φησιν": "φησι",
        }.items():
            result = result.replace(with_nu, without_nu)
    return result


def strip_punctuation(word: str) -> str:
    """Strip leading and trailing punctuation from a Greek word token.

    Args:
        word: A token string that may have attached punctuation (e.g., "λόγος,").

    Returns:
        The word with leading/trailing punctuation removed.

    Example:
        >>> strip_punctuation("λόγος,")
        'λόγος'
        >>> strip_punctuation("·εἶπεν·")
        'εἶπεν'
    """
    return word.strip("".join(_PUNCT_CHARS))


def extract_greek_words(text: str) -> list[str]:
    """Extract a list of Greek word tokens from a whitespace-separated string.

    Used to parse MorphGNT lemma columns and plain-text Apostolic Fathers
    files, which contain one or more Greek words per line mixed with
    punctuation tokens.

    Args:
        text: Whitespace-separated string, possibly containing punctuation
            tokens and non-Greek characters.

    Returns:
        List of Greek-only tokens (punctuation stripped, in order).

    Example:
        >>> extract_greek_words("Ἐν ἀρχῇ ἦν · ὁ λόγος")
        ['Ἐν', 'ἀρχῇ', 'ἦν', 'ὁ', 'λόγος']
    """
    return [token for token in text.split() if is_greek(token)]


def parse_verse_ref(ref: str) -> tuple[str, int, int]:
    """Parse a verse reference like 'Matt 14:1' or 'Luke 19:11-27'.

    Converts standard scholarly verse references into a canonical tuple
    of (book_name, start_verse_id, end_verse_id), where verse IDs are
    computed as ``chapter * 1000 + verse`` for easy range comparison.

    Args:
        ref: Verse reference string in standard scholarly format.
            Supports abbreviated book names (Matt, Mk, Lk) and
            verse ranges (e.g., 14:1-12).

    Returns:
        Tuple of (canonical_book_name, start_verse_id, end_verse_id).
        For single verses, start_id == end_id.

    Raises:
        ValueError: If the reference cannot be parsed.

    Example:
        >>> parse_verse_ref("Matt 14:1-12")
        ('Matthew', 14001, 14012)
        >>> parse_verse_ref("Luke 19:27")
        ('Luke', 19027, 19027)
        >>> parse_verse_ref("Mk 6:45")
        ('Mark', 6045, 6045)
    """
    _BOOK_MAP: dict[str, str] = {
        "Matt": "Matthew",
        "Mt": "Matthew",
        "Matthew": "Matthew",
        "Mark": "Mark",
        "Mk": "Mark",
        "Marc": "Mark",
        "Luke": "Luke",
        "Lk": "Luke",
        "Luk": "Luke",
        "John": "John",
        "Jn": "John",
    }

    pattern = re.compile(
        r"(?P<book>\w+)\s+"
        r"(?P<chapter>\d+):"
        r"(?P<start_v>\d+)"
        r"(?:-(?P<end_v>\d+))?"
    )
    match = pattern.match(ref.strip())
    if not match:
        msg = f"Cannot parse verse reference: {ref!r}"
        raise ValueError(msg)

    book_abbr = match.group("book")
    book = _BOOK_MAP.get(book_abbr, book_abbr)
    chapter = int(match.group("chapter"))
    start_v = int(match.group("start_v"))
    end_v = int(match.group("end_v") or start_v)

    start_id = chapter * 1000 + start_v
    end_id = chapter * 1000 + end_v
    return book, start_id, end_id
