"""Pytest fixtures for SynoptiQ tests.

All fixtures are deterministic (no real network calls, no file I/O).
The ``tiny_corpus`` fixture provides a minimal in-memory corpus
with 3 pericopes and ~60 tokens for fast unit testing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

# ── Helper builders ───────────────────────────────────────────────────────────


def _make_token(
    token_id: str,
    book: str,
    chapter: int,
    verse: int,
    position: int,
    text: str,
    lemma: str,
    pos: str = "N-",
    pericope_id: str | None = None,
) -> dict[str, Any]:
    """Build a minimal TokenRecord dict for testing."""
    return {
        "token_id": token_id,
        "book": book,
        "chapter": chapter,
        "verse": verse,
        "position": position,
        "text": text,
        "normalized": text.lower(),
        "lemma": lemma,
        "pos": pos,
        "morph": "NSN-----",
        "pericope_id": pericope_id,
        "is_punctuation": False,
    }


# ── Core fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_morphgnt_line() -> str:
    """A single valid MorphGNT TSV line (Matthew 1:1, word 1)."""
    # Format: bcv\tpos\tparsing\ttext\tword\tnormalized\tlemma
    return "MAT 1:1/1\tN-\tNSN-----\tΒίβλος\tΒίβλος\tβίβλος\tβίβλος"


@pytest.fixture
def sample_triple_tokens() -> dict[str, list[dict[str, Any]]]:
    """A minimal set of tokens for triple tradition pericope 020 (Cleansing of a Leper).

    Matthew: Matt 8:2-4 (~6 tokens)
    Mark: Mark 1:40-45 (~8 tokens)
    Luke: Luke 5:12-16 (~7 tokens)
    """
    matt = [
        _make_token("Matt.8.2.0", "Matthew", 8, 2, 0, "καὶ", "καί", "C-", "020"),
        _make_token("Matt.8.2.1", "Matthew", 8, 2, 1, "ἰδοὺ", "ἰδού", "X-", "020"),
        _make_token("Matt.8.2.2", "Matthew", 8, 2, 2, "λεπρὸς", "λεπρός", "A-", "020"),
        _make_token("Matt.8.2.3", "Matthew", 8, 2, 3, "προσελθὼν", "προσέρχομαι", "V-", "020"),
        _make_token("Matt.8.3.0", "Matthew", 8, 3, 0, "καὶ", "καί", "C-", "020"),
        _make_token("Matt.8.3.1", "Matthew", 8, 3, 1, "ἐκτείνας", "ἐκτείνω", "V-", "020"),
    ]
    mark = [
        _make_token("Mark.1.40.0", "Mark", 1, 40, 0, "καὶ", "καί", "C-", "020"),
        _make_token("Mark.1.40.1", "Mark", 1, 40, 1, "ἔρχεται", "ἔρχομαι", "V-", "020"),
        _make_token("Mark.1.40.2", "Mark", 1, 40, 2, "πρὸς", "πρός", "P-", "020"),
        _make_token("Mark.1.40.3", "Mark", 1, 40, 3, "λεπρὸς", "λεπρός", "A-", "020"),
        _make_token("Mark.1.41.0", "Mark", 1, 41, 0, "καὶ", "καί", "C-", "020"),
        _make_token("Mark.1.41.1", "Mark", 1, 41, 1, "σπλαγχνισθεὶς", "σπλαγχνίζομαι", "V-", "020"),
        _make_token("Mark.1.41.2", "Mark", 1, 41, 2, "ἐκτείνας", "ἐκτείνω", "V-", "020"),
        _make_token("Mark.1.42.0", "Mark", 1, 42, 0, "καὶ", "καί", "C-", "020"),
    ]
    luke = [
        _make_token("Luke.5.12.0", "Luke", 5, 12, 0, "καὶ", "καί", "C-", "020"),
        _make_token("Luke.5.12.1", "Luke", 5, 12, 1, "ἰδοὺ", "ἰδού", "X-", "020"),
        _make_token("Luke.5.12.2", "Luke", 5, 12, 2, "λεπρὸς", "λεπρός", "A-", "020"),
        _make_token("Luke.5.12.3", "Luke", 5, 12, 3, "πλήρης", "πλήρης", "A-", "020"),
        _make_token("Luke.5.13.0", "Luke", 5, 13, 0, "καὶ", "καί", "C-", "020"),
        _make_token("Luke.5.13.1", "Luke", 5, 13, 1, "ἐκτείνας", "ἐκτείνω", "V-", "020"),
        _make_token("Luke.5.13.2", "Luke", 5, 13, 2, "τὴν", "ὁ", "RA", "020"),
    ]
    return {"Matthew": matt, "Mark": mark, "Luke": luke}


@pytest.fixture
def sample_double_tokens() -> dict[str, list[dict[str, Any]]]:
    """Minimal tokens for double tradition pericope 088 (Lord's Prayer)."""
    matt = [
        _make_token("Matt.6.9.0", "Matthew", 6, 9, 0, "οὕτως", "οὕτως", "D-", "088"),
        _make_token("Matt.6.9.1", "Matthew", 6, 9, 1, "οὖν", "οὖν", "D-", "088"),
        _make_token("Matt.6.9.2", "Matthew", 6, 9, 2, "προσεύχεσθε", "προσεύχομαι", "V-", "088"),
        _make_token("Matt.6.9.3", "Matthew", 6, 9, 3, "ὑμεῖς", "σύ", "RP", "088"),
        _make_token("Matt.6.9.4", "Matthew", 6, 9, 4, "Πάτερ", "πατήρ", "N-", "088"),
    ]
    luke = [
        _make_token("Luke.11.2.0", "Luke", 11, 2, 0, "εἶπεν", "λέγω", "V-", "088"),
        _make_token("Luke.11.2.1", "Luke", 11, 2, 1, "αὐτοῖς", "αὐτός", "RP", "088"),
        _make_token("Luke.11.2.2", "Luke", 11, 2, 2, "προσεύχεσθε", "προσεύχομαι", "V-", "088"),
        _make_token("Luke.11.2.3", "Luke", 11, 2, 3, "Πάτερ", "πατήρ", "N-", "088"),
    ]
    return {"Matthew": matt, "Luke": luke}


@pytest.fixture
def tiny_corpus_dfs(
    sample_triple_tokens: dict[str, list[dict[str, Any]]],
    sample_double_tokens: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal token and pericope DataFrames for unit testing."""
    all_tokens: list[dict[str, Any]] = []

    # Triple tradition pericope
    for book_tokens in sample_triple_tokens.values():
        all_tokens.extend(book_tokens)

    # Double tradition pericope
    for book_tokens in sample_double_tokens.values():
        all_tokens.extend(book_tokens)

    # Mark-unique token (no pericope assignment — tests unassigned handling)
    all_tokens.append(_make_token("Mark.4.26.0", "Mark", 4, 26, 0, "καὶ", "καί", "C-", "044"))

    token_df = pd.DataFrame(all_tokens)

    pericope_records = [
        {
            "pericope_id": "020",
            "tradition": "triple",
            "genre": "narrative",
            "books": '["Matthew", "Mark", "Luke"]',
        },
        {"pericope_id": "044", "tradition": "mark_unique", "genre": "wisdom", "books": '["Mark"]'},
        {
            "pericope_id": "088",
            "tradition": "double",
            "genre": "discourse",
            "books": '["Matthew", "Luke"]',
        },
    ]
    pericope_df = pd.DataFrame(pericope_records)

    return token_df, pericope_df


@pytest.fixture
def tiny_corpus(tiny_corpus_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> Corpus:  # noqa: F821
    """Minimal in-memory Corpus with 3 pericopes for fast unit testing."""
    from synoptiq.data.corpus import Corpus

    token_df, pericope_df = tiny_corpus_dfs
    return Corpus(token_df, pericope_df, alignments={})


@pytest.fixture
def dummy_alignment() -> list[tuple[int | None, int | None]]:
    """A pre-computed alignment for a 4-token × 4-token pair."""
    # Matthew: [καί, ἰδού, λεπρός, προσέρχομαι]
    # Mark:    [καί, ἔρχομαι, λεπρός, ἐκτείνω]
    # Expected: (0,0) kai-kai, (1,None) idou-gap, (2,2) lepros-lepros, (None,1) gap-erchomai, (3,3)
    return [(0, 0), (1, None), (2, 2), (None, 1), (3, 3)]
