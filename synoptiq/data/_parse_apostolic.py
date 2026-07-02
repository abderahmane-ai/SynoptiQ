"""Apostolic Fathers plain text parser for SynoptiQ DAPT corpus.

Parses the jtauber/apostolic-fathers repository, which provides
corrected Greek texts of the Apostolic Fathers in plain ``.txt`` format.

Format confirmed (online research): plain text files, one per author/work.
No XML — just Greek prose. One word per line or space-separated.

Texts included (post-NT Koine Greek, 1st–2nd century CE):
  - 1 Clement
  - 2 Clement
  - Ignatius (7 letters)
  - Polycarp
  - Didache
  - Epistle of Barnabas
  - Shepherd of Hermas
  - Martyrdom of Polycarp
  - Epistle to Diognetus

~35K tokens total — small but linguistically valuable as Koine
produced by Christians directly influenced by the NT vocabulary.

Usage: DAPT corpus supplement (after SBLGNT, LXX, Josephus).
"""

from __future__ import annotations

from pathlib import Path

from synoptiq.utils.greek import is_greek, normalize_greek, strip_punctuation
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)


def parse_apostolic_fathers(
    apostolic_dir: Path,
    *,
    max_tokens: int | None = 50_000,
) -> list[str]:
    """Parse all Apostolic Fathers plain text files.

    Walks all ``*.txt`` files in the repository and extracts Greek tokens.

    Args:
        apostolic_dir: Path to the cloned jtauber/apostolic-fathers repo.
        max_tokens: Maximum tokens to collect (None = no limit).

    Returns:
        List of normalized Greek tokens for DAPT pre-training.

    Raises:
        FileNotFoundError: If no text files are found.
    """
    txt_files = sorted(apostolic_dir.rglob("*.txt"))

    if not txt_files:
        msg = f"No .txt files found in Apostolic Fathers dir: {apostolic_dir}"
        raise FileNotFoundError(msg)

    _LOG.info(
        "found Apostolic Fathers text files",
        extra={"count": len(txt_files)},
    )

    all_tokens: list[str] = []

    for filepath in txt_files:
        # Skip README and non-Greek files
        if filepath.name.lower() in {"readme.txt", "readme.md", "notes.txt"}:
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = filepath.read_text(encoding="latin-1")
            except Exception:
                _LOG.warning("failed to read file", extra={"file": str(filepath)})
                continue

        file_tokens: list[str] = []
        for raw_token in text.split():
            cleaned = strip_punctuation(raw_token.strip())
            if cleaned and is_greek(cleaned):
                file_tokens.append(normalize_greek(cleaned))

        if not file_tokens:
            _LOG.debug("no Greek tokens found in file", extra={"file": filepath.name})
            continue

        all_tokens.extend(file_tokens)
        _LOG.info(
            "parsed Apostolic Fathers file",
            extra={"file": filepath.name, "n_tokens": len(file_tokens)},
        )

        if max_tokens is not None and len(all_tokens) >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            _LOG.info("Apostolic Fathers token limit reached", extra={"limit": max_tokens})
            break

    _LOG.info(
        "Apostolic Fathers parse complete",
        extra={"n_tokens": len(all_tokens), "n_files_parsed": len(txt_files)},
    )
    return all_tokens
