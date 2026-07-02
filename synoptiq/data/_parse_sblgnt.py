"""SBLGNT XML parser for SynoptiQ.

Parses the Faithlife SBLGNT XML files to produce a list of token
records for each synoptic book (Matthew, Mark, Luke).

SBLGNT XML structure (from Faithlife/SBLGNT repository):
    <sblgnt>
      <book id="Matthew">
        <title>...</title>
        <p>
          <verse-number id="1:1"/>
          <w>Βίβλος</w>
          <suffix>,</suffix>
          <w>γενέσεως</w>
          ...
        </p>
      </book>
    </sblgnt>

Each <w> tag is a word; <suffix> contains punctuation after the word.
<verse-number id="chapter:verse"> marks verse boundaries (cumulative).

Output: list[dict] with standardized keys matching TokenRecord schema.
The position field is 0-indexed within each verse.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from synoptiq.utils.greek import normalize_greek
from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import Book

_LOG = get_logger(__name__)

# Mapping: SBLGNT XML <book id="..."> attribute → canonical Book name
_SBLGNT_ID_TO_BOOK: dict[str, Book] = {
    "Mt": "Matthew",
    "Mk": "Mark",
    "Lk": "Luke",
    "Lu": "Luke",
    "Jn": "John",
}

# Mapping: canonical Book name → XML filename in the SBLGNT repo
_BOOK_TO_FILENAME: dict[Book, str] = {
    "Matthew": "Matt.xml",
    "Mark": "Mark.xml",
    "Luke": "Luke.xml",
    "John": "John.xml",
}


def _parse_sblgnt_book(
    book_elem: ET.Element,
    *,
    book_name: Book,
) -> list[dict[str, object]]:
    """Parse a single <book> element from SBLGNT XML.

    Args:
        book_elem: The <book> XML element.
        book_name: Canonical book name (e.g., "Matthew").

    Returns:
        List of token dicts with keys: book, chapter, verse, position,
        text, suffix, is_punctuation.
    """
    records: list[dict[str, object]] = []
    current_chapter = 1
    current_verse = 1
    verse_position = 0  # 0-indexed position within current verse

    for para in book_elem.findall("p"):
        for child in para:
            tag = child.tag
            text = (child.text or "").strip()

            if tag == "verse-number":
                # id attribute is like "1:1", "Matt 1:1", or "Mark 1:1"
                verse_id = child.get("id", "1:1")
                try:
                    # Handle both "chapter:verse" and "Book chapter:verse" formats
                    parts = verse_id.split()
                    ref = parts[-1] if parts else verse_id  # take last token
                    ch_str, vs_str = ref.split(":")
                    current_chapter = int(ch_str)
                    current_verse = int(vs_str)
                except (ValueError, IndexError):
                    _LOG.warning(
                        "malformed verse-number id",
                        extra={"book": book_name, "id": verse_id},
                    )
                verse_position = 0  # Reset position counter at each verse
                continue

            if tag == "w" and text:
                # Generate stable token ID
                token_id = f"{book_name}.{current_chapter}.{current_verse}.{verse_position}"
                records.append(
                    {
                        "token_id": token_id,
                        "book": book_name,
                        "chapter": current_chapter,
                        "verse": current_verse,
                        "position": verse_position,
                        "text": text,
                        "normalized": normalize_greek(text),
                        # lemma and morphology added during MorphGNT merge step
                        "lemma": "",
                        "pos": "",
                        "morph": "",
                        "pericope_id": None,
                        "is_punctuation": False,
                    }
                )
                verse_position += 1

            # Punctuation attached via <suffix> — we skip as standalone tokens
            # (SBLGNT punctuation is encoded as suffix text, not separate tokens)

    return records


def parse_sblgnt(
    sblgnt_dir: Path,
    *,
    books: list[Book] | None = None,
) -> dict[Book, list[dict[str, object]]]:
    """Parse SBLGNT XML files for the specified books.

    The Faithlife/SBLGNT repo stores each book as a separate XML file
    in ``data/sblgnt/xml/`` (e.g., ``Matt.xml``, ``Mark.xml``, ``Luke.xml``).
    Each file contains a single ``<book id="Mt">`` element with the text.

    Args:
        sblgnt_dir: Path to the cloned Faithlife/SBLGNT repository.
        books: List of canonical book names. Defaults to Matt, Mark, Luke.

    Returns:
        Dict mapping canonical book name → list of token dicts.

    Raises:
        FileNotFoundError: If a requested book's XML file cannot be found.
    """
    target_books: list[Book] = books or ["Matthew", "Mark", "Luke"]

    # Look for individual book files in data/sblgnt/xml/
    xml_dir = sblgnt_dir / "data" / "sblgnt" / "xml"
    if not xml_dir.exists():
        xml_dir = sblgnt_dir / "xml"  # some layouts

    if not xml_dir.exists():
        msg = f"SBLGNT XML directory not found in {sblgnt_dir}"
        raise FileNotFoundError(msg)

    result: dict[Book, list[dict[str, object]]] = {}
    found_book_ids: set[str] = set()

    # Walk all XML files in the directory
    for xml_file in sorted(xml_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        root = tree.getroot()
        if root.tag != "book":
            continue

        book_id = root.get("id", "")
        book_name = _SBLGNT_ID_TO_BOOK.get(book_id)
        if book_name is None or book_name not in target_books:
            continue

        _LOG.info("parsing book", extra={"book": book_name, "file": str(xml_file.name)})
        records = _parse_sblgnt_book(root, book_name=book_name)
        result[book_name] = records
        found_book_ids.add(book_id)
        _LOG.info("book parsed", extra={"book": book_name, "n_tokens": len(records)})

    # Validate all requested books were found
    missing = set(target_books) - set(result)
    if missing:
        _LOG.warning("books not found in SBLGNT", extra={"missing": sorted(missing)})

    total = sum(len(v) for v in result.values())
    _LOG.info("SBLGNT parse complete", extra={"n_books": len(result), "n_tokens": total})
    return result
