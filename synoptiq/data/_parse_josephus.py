"""Josephus parser for SynoptiQ DAPT corpus.

Parses Josephus texts from the First1KGreek TEI XML repository.
First1KGreek is cleaner than scraping PACE McMaster HTML.

Josephus texts in First1KGreek (TEI XML, OpenGreekAndLatin):
  - Jewish War (Bellum Judaicum) — tlg0526.tlg004 and related files
  - Jewish Antiquities (Antiquitates Judaicae) — tlg0526.tlg001
  - Against Apion — tlg0526.tlg002
  - Life (Vita) — tlg0526.tlg003

TEI XML structure: <text><body><div><p><milestone/><w/> or text nodes
We use a permissive parser that extracts all text nodes under <body>
and filters for Greek tokens.

Usage: DAPT corpus only (same as LXX). ~300K tokens from Josephus
provides koine prose narrative essential for KoineFormer pre-training.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from synoptiq.utils.greek import is_greek, normalize_greek, strip_punctuation
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# TLG identifiers for Josephus in First1KGreek
JOSEPHUS_TLG_IDS: frozenset[str] = frozenset(
    {
        "tlg0526",  # All Josephus texts share this author ID
    }
)

# TEI XML namespaces used in First1KGreek
TEI_NS: dict[str, str] = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "": "http://www.tei-c.org/ns/1.0",  # Default namespace
}


def _extract_text_nodes(element: ET.Element) -> list[str]:
    """Recursively extract all text content from an XML element tree.

    Handles both element.text (before first child) and element.tail
    (text after a child tag) — standard ET text extraction.

    Args:
        element: Root XML element to extract from.

    Returns:
        List of text strings (non-empty, non-whitespace-only).
    """
    texts: list[str] = []
    if element.text:
        stripped = element.text.strip()
        if stripped:
            texts.append(stripped)
    for child in element:
        texts.extend(_extract_text_nodes(child))
        if child.tail:
            stripped = child.tail.strip()
            if stripped:
                texts.append(stripped)
    return texts


def _parse_josephus_file(xml_path: Path) -> list[str]:
    """Parse a single Josephus TEI XML file and return Greek tokens.

    Args:
        xml_path: Path to a ``*.xml`` file from First1KGreek.

    Returns:
        List of normalized Greek tokens from this file.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        _LOG.warning(
            "XML parse error in Josephus file",
            extra={"file": str(xml_path), "error": str(e)},
        )
        return []

    root = tree.getroot()

    # Strip namespace prefix for easier navigation
    # ET uses Clark notation: {http://www.tei-c.org/ns/1.0}text
    ns_prefix = "{http://www.tei-c.org/ns/1.0}"

    body = root.find(f".//{ns_prefix}body")
    if body is None:
        # Try without namespace
        body = root.find(".//body")
    if body is None:
        _LOG.debug("no body element found", extra={"file": str(xml_path)})
        return []

    raw_texts = _extract_text_nodes(body)

    tokens: list[str] = []
    for text_chunk in raw_texts:
        for raw_token in text_chunk.split():
            cleaned = strip_punctuation(raw_token)
            if cleaned and is_greek(cleaned):
                tokens.append(normalize_greek(cleaned))

    return tokens


def parse_josephus(
    first1k_dir: Path,
    *,
    max_tokens: int | None = 350_000,
) -> list[str]:
    """Parse Josephus texts from the First1KGreek TEI XML repository.

    Scans for XML files with ``tlg0526`` in the filename (Josephus TLG ID).
    Falls back to scanning any Greek XML files if no Josephus-specific
    files are found.

    Args:
        first1k_dir: Path to the cloned OpenGreekAndLatin/First1KGreek repo.
        max_tokens: Maximum tokens to collect (default 350K).

    Returns:
        List of normalized Greek tokens for DAPT pre-training.

    Raises:
        FileNotFoundError: If no XML files are found in the repository.
    """
    # First1KGreek organizes files in data/ or similar subdirectory
    # File naming convention: tlg{author_id}.tlg{work_id}.perseus-grc*.xml
    josephus_files = sorted(first1k_dir.rglob("*tlg0526*.xml"))

    if not josephus_files:
        # Broader search: all Greek XML files
        josephus_files = sorted(first1k_dir.rglob("*.xml"))
        _LOG.info(
            "No Josephus-specific files found; scanning all XML files",
            extra={"count": len(josephus_files)},
        )

    if not josephus_files:
        msg = f"No XML files found in First1KGreek dir: {first1k_dir}"
        raise FileNotFoundError(msg)

    _LOG.info(
        "parsing Josephus from First1KGreek",
        extra={"n_files": len(josephus_files)},
    )

    all_tokens: list[str] = []

    for xml_path in josephus_files:
        file_tokens = _parse_josephus_file(xml_path)
        if not file_tokens:
            continue

        all_tokens.extend(file_tokens)
        _LOG.info(
            "parsed Josephus file",
            extra={"file": xml_path.name, "n_tokens": len(file_tokens)},
        )

        if max_tokens is not None and len(all_tokens) >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            _LOG.info("Josephus token limit reached", extra={"limit": max_tokens})
            break

    _LOG.info(
        "Josephus parse complete",
        extra={"n_tokens": len(all_tokens)},
    )
    return all_tokens
