"""Tests for synoptiq/data/alignment.py — token-level alignment."""

from __future__ import annotations

from synoptiq.data.alignment import _make_token_key, align_tokens, alignment_score


def _tk(lemma: str, text: str = "", pos: str = "N-", pericope_id: str = "020") -> dict:
    return {
        "token_id": f"t.{lemma}",
        "book": "Matthew",
        "chapter": 1,
        "verse": 1,
        "position": 0,
        "text": text or lemma,
        "normalized": (text or lemma).lower(),
        "lemma": lemma,
        "pos": pos,
        "morph": "",
        "pericope_id": pericope_id,
        "is_punctuation": False,
    }


class TestMakeTokenKey:
    """Token key extraction for alignment."""

    def test_key_tuple(self) -> None:
        t = _tk("καί", pos="C-", text="καί")
        lemma, pos, norm = _make_token_key(t)
        # _make_token_key strips accents
        assert lemma == "και"
        assert pos == "C-"

    def test_punctuation_lemma_is_surface(self) -> None:
        t = _tk(".", pos="U-")
        t["text"] = "."
        lemma, pos, _ = _make_token_key(t)
        assert lemma == "."


class TestAlignTokens:
    """Needleman-Wunsch global alignment."""

    def test_identical_sequences(self) -> None:
        a = [_tk("καί"), _tk("λέγω"), _tk("αὐτός")]
        b = [_tk("καί"), _tk("λέγω"), _tk("αὐτός")]
        pairs = align_tokens(a, b)
        assert len(pairs) >= 2
        # All should be 1:1 aligned
        assert all(pa is not None and pb is not None for pa, pb in pairs)

    def test_single_token(self) -> None:
        a = [_tk("καί")]
        b = [_tk("καί")]
        pairs = align_tokens(a, b)
        assert len(pairs) >= 1
        assert pairs[0] == (0, 0)

    def test_deletion_in_b(self) -> None:
        a = [_tk("καί"), _tk("λέγω"), _tk("αὐτός")]
        b = [_tk("καί"), _tk("αὐτός")]
        pairs = align_tokens(a, b)
        has_gap_a = any(pa is not None and pb is None for pa, pb in pairs)
        has_gap_b = any(pa is None and pb is not None for pa, pb in pairs)
        assert has_gap_a or has_gap_b, f"No gaps found in alignment: {pairs}"

    def test_empty_input_raises(self) -> None:
        import pytest

        a: list = []
        b = [_tk("καί")]
        with pytest.raises(ValueError, match="non-empty"):
            align_tokens(a, b)


class TestAlignmentScore:
    """Alignment quality statistics."""

    def test_perfect_match(self) -> None:
        a = [_tk("καί"), _tk("λέγω")]
        b = [_tk("καί"), _tk("λέγω")]
        pairs = [(0, 0), (1, 1)]
        s = alignment_score(a, b, pairs)
        assert s["lemma_match_rate"] == 1.0
        assert s["n_aligned"] == 2

    def test_surface_match_counted(self) -> None:
        a = [_tk("καί", text="καί"), _tk("λέγω", text="λέγει")]
        b = [_tk("καί", text="καὶ"), _tk("λέγω", text="λέγουσιν")]
        # Same lemmas, different surface forms
        pairs = [(0, 0), (1, 1)]
        s = alignment_score(a, b, pairs)
        assert s["lemma_match_rate"] == 1.0
        assert s["surface_match_rate"] < 1.0
