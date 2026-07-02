"""LXX (Septuagint) plain text parser for SynoptiQ DAPT corpus.

Parses the biblicalhumanities/Septuagint repository, which provides
the Rahlfs LXX text as plain text files (one per book).

Usage: This corpus is used EXCLUSIVELY for Domain-Adaptive Pre-Training
(DAPT) of KoineFormer. It is NOT used for pericope alignment or source
criticism analysis. LXX provides ~600K tokens of Koine Greek text
to extend KoineFormer's language model pre-training beyond the NT.

File format: plain text, one verse per line (or continuous prose).
We tokenize on whitespace and filter for Greek-only tokens.
"""

from __future__ import annotations

from pathlib import Path

from synoptiq.utils.greek import is_greek, normalize_greek, strip_punctuation
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# LXX books that are closest to Koine Greek (most relevant for DAPT)
# These are post-classical and Hellenistic texts; classical poetry less useful.
PRIORITY_LXX_BOOKS: frozenset[str] = frozenset(
    {
        # Historical books (prose narrative — closest to NT Koine)
        "genesis",
        "exodus",
        "deuteronomy",
        "joshua",
        "judges",
        "1kingdoms",
        "2kingdoms",
        "3kingdoms",
        "4kingdoms",
        "1chronicles",
        "2chronicles",
        "1maccabees",
        "2maccabees",
        # Wisdom/Prophets (mixed prose)
        "proverbs",
        "sirach",
        "wisdom",
        "tobit",
        "isaiah",
        "jeremiah",
        "ezekiel",
        "daniel",
        "1esdras",
        "judith",
        "baruch",
    }
)


def parse_lxx(
    lxx_dir: Path,
    *,
    priority_only: bool = False,
    max_tokens: int | None = 700_000,
) -> list[str]:
    """Parse LXX plain text files and return a flat list of Greek tokens.

    The biblicalhumanities/Septuagint repository has plain text files
    in a ``texts/`` subdirectory, organized by book. We walk all ``.txt``
    files and extract Greek tokens.

    Args:
        lxx_dir: Path to the cloned biblicalhumanities/Septuagint repository.
        priority_only: If True, only parse Koine-priority books.
        max_tokens: Stop after this many tokens (None = no limit).

    Returns:
        List of normalized Greek token strings for DAPT pre-training.

    Raises:
        FileNotFoundError: If no text files are found.
    """
    # Find text files — the repo may have them in various locations
    text_files = sorted(lxx_dir.rglob("*.txt"))
    if not text_files:
        text_files = sorted(lxx_dir.rglob("*.text"))

    if not text_files:
        msg = f"No LXX text files found in {lxx_dir}"
        raise FileNotFoundError(msg)

    _LOG.info(
        "found LXX text files",
        extra={"count": len(text_files)},
    )

    all_tokens: list[str] = []

    for filepath in text_files:
        if priority_only:
            # Check if this book is in priority list
            stem = filepath.stem.lower()
            if not any(priority in stem for priority in PRIORITY_LXX_BOOKS):
                continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = filepath.read_text(encoding="latin-1")
            except Exception:
                _LOG.warning("failed to read LXX file", extra={"file": str(filepath)})
                continue

        # Tokenize on whitespace and filter for Greek tokens
        file_tokens: list[str] = []
        for raw_token in text.split():
            cleaned = strip_punctuation(raw_token.strip())
            if cleaned and is_greek(cleaned):
                file_tokens.append(normalize_greek(cleaned))

        all_tokens.extend(file_tokens)
        _LOG.info(
            "parsed LXX file",
            extra={"file": filepath.name, "n_tokens": len(file_tokens)},
        )

        if max_tokens is not None and len(all_tokens) >= max_tokens:
            _LOG.info(
                "LXX token limit reached",
                extra={"limit": max_tokens, "collected": len(all_tokens)},
            )
            all_tokens = all_tokens[:max_tokens]
            break

    _LOG.info(
        "LXX parse complete",
        extra={"n_tokens": len(all_tokens), "n_files": len(text_files)},
    )
    return all_tokens
