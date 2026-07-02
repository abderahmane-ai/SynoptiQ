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

from synoptiq.utils.constants import SBLGNT_BOOK_IDS
from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.greek import normalize_greek
from synoptiq.utils.types_ import Book

_LOG = get_logger(__name__)

# Books to parse for synoptic analysis (only synoptic gospels in core pipeline)
SYNOPTIC_BOOK_IDS: frozenset[str] = frozenset(SBLGNT_BOOK_IDS.keys())


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
                # id attribute is like "1:1" or "14:1"
                verse_id = child.get("id", "1:1")
                try:
                    ch_str, vs_str = verse_id.split(":")
                    current_chapter = int(ch_str)
                    current_verse = int(vs_str)
                except ValueError:
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

    Looks for ``sblgnt.xml`` in ``sblgnt_dir/`` (the main file in
    the Faithlife/SBLGNT repository contains all books in one XML file).
    Falls back to per-book files if the monolithic file is absent.

    Args:
        sblgnt_dir: Path to the cloned Faithlife/SBLGNT repository.
        books: List of books to parse. If None, parses Matthew, Mark, Luke.

    Returns:
        Dict mapping book name → list of token dicts.

    Raises:
        FileNotFoundError: If no SBLGNT XML file can be found.
        ET.ParseError: If the XML is malformed.
    """
    target_books: list[Book] = books or ["Matthew", "Mark", "Luke"]

    # The Faithlife repo has sblgnt.xml at the root
    xml_file = sblgnt_dir / "sblgnt.xml"
    if not xml_file.exists():
        # Some forks split by book; try per-book file
        xml_file = sblgnt_dir / "text" / "sblgnt.xml"
    if not xml_file.exists():
        msg = f"SBLGNT XML file not found in {sblgnt_dir}. Expected sblgnt.xml or text/sblgnt.xml"
        raise FileNotFoundError(msg)

    _LOG.info("parsing SBLGNT XML", extra={"file": str(xml_file)})
    tree = ET.parse(xml_file)
    root = tree.getroot()

    result: dict[Book, list[dict[str, object]]] = {}

    for book_elem in root.findall("book"):
        book_id = book_elem.get("id", "")
        if book_id not in {b: b for b in target_books}:
            continue

        book_name = book_id  # In SBLGNT XML, id is the full name
        if book_name not in SYNOPTIC_BOOK_IDS:
            continue

        _LOG.info("parsing book", extra={"book": book_name})
        records = _parse_sblgnt_book(book_elem, book_name=book_name)  # type: ignore[arg-type]
        result[book_name] = records  # type: ignore[index]
        _LOG.info(
            "book parsed",
            extra={"book": book_name, "n_tokens": len(records)},
        )

    # Validate all requested books were found
    missing = set(target_books) - set(result)
    if missing:
        _LOG.warning("books not found in SBLGNT", extra={"missing": sorted(missing)})

    total = sum(len(v) for v in result.values())
    _LOG.info("SBLGNT parse complete", extra={"n_books": len(result), "n_tokens": total})
    return result
