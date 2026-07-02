"""Tests for synoptiq/data/corpus.py — the Corpus class.

Uses the tiny_corpus fixture from conftest.py (3 pericopes, ~31 tokens).
"""

from __future__ import annotations

from typing import Any


class TestCorpusBasics:
    """Construction and basic properties."""

    def test_corpus_built(self, tiny_corpus: Any) -> None:
        assert tiny_corpus is not None

    def test_n_tokens(self, tiny_corpus: Any) -> None:
        assert tiny_corpus.n_tokens == 31

    def test_n_pericopes(self, tiny_corpus: Any) -> None:
        assert tiny_corpus.n_pericopes == 3


class TestIterPericopes:
    """Pericope iteration."""

    def test_all_pericopes(self, tiny_corpus: Any) -> None:
        pericopes = list(tiny_corpus.iter_pericopes())
        assert len(pericopes) == 3

    def test_triple_filter(self, tiny_corpus: Any) -> None:
        pericopes = list(tiny_corpus.iter_pericopes(tradition="triple"))
        assert len(pericopes) == 1
        assert pericopes[0]["pericope_id"] == "020"

    def test_double_filter(self, tiny_corpus: Any) -> None:
        pericopes = list(tiny_corpus.iter_pericopes(tradition="double"))
        assert len(pericopes) == 1
        assert pericopes[0]["pericope_id"] == "088"

    def test_mark_unique_filter(self, tiny_corpus: Any) -> None:
        pericopes = list(tiny_corpus.iter_pericopes(tradition="mark_unique"))
        assert len(pericopes) == 1
        assert pericopes[0]["pericope_id"] == "044"

    def test_pericope_has_tokens(self, tiny_corpus: Any) -> None:
        for p in tiny_corpus.iter_pericopes(tradition="triple"):
            assert len(p["tokens"]["Matthew"]) > 0
            assert len(p["tokens"]["Mark"]) > 0
            assert len(p["tokens"]["Luke"]) > 0


class TestDirectionPairs:
    """Direction pair generation."""

    def test_triple_has_pairs(self, tiny_corpus: Any) -> None:
        pairs = list(tiny_corpus.direction_pairs(tradition="triple"))
        assert len(pairs) > 0

    def test_pairs_have_alignment(self, tiny_corpus: Any) -> None:
        for book_a, tokens_a, book_b, tokens_b, alignment in tiny_corpus.direction_pairs(
            tradition="triple"
        ):
            assert isinstance(book_a, str)
            assert isinstance(book_b, str)
            assert len(tokens_a) > 0
            assert len(tokens_b) > 0
            assert isinstance(alignment, list)


class TestTokenAccess:
    """Token access methods."""

    def test_get_tokens_by_book(self, tiny_corpus: Any) -> None:
        tokens = tiny_corpus.get_tokens("Matthew")
        assert len(tokens) > 0

    def test_get_tokens_by_pericope(self, tiny_corpus: Any) -> None:
        tokens = tiny_corpus.get_tokens(pericope_id="020")
        assert len(tokens) > 0

    def test_get_tokens_by_book_and_pericope(self, tiny_corpus: Any) -> None:
        tokens = tiny_corpus.get_tokens("Matthew", pericope_id="020")
        assert len(tokens) == 6  # 6 tokens in the fixture

    def test_get_verse(self, tiny_corpus: Any) -> None:
        tokens = tiny_corpus.get_verse("Mark", 1, 40)
        assert len(tokens) == 4
        assert tokens[2]["lemma"] == "πρός"
