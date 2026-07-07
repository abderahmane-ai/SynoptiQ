"""Tests for the directional feature primitives."""

from __future__ import annotations

from synoptiq.direction.features import (
    centrality_asym,
    connective_vote,
    intro_lateness,
    pair_features,
    shared_count,
)


def _w(lemma: str, pos: str = "N-") -> dict:
    return {"lemma": lemma, "normalized": lemma, "pos": pos, "is_punctuation": False}


def _conn(word: str) -> dict:
    return {"lemma": word, "normalized": word, "pos": "C-", "is_punctuation": False}


def test_shared_count_counts_content_matches() -> None:
    a = [_w("alpha"), _w("beta"), _w("gamma")]
    b = [_w("alpha"), _w("beta"), _w("zeta")]
    assert shared_count(a, b) == 2


def test_centrality_is_antisymmetric() -> None:
    a, b, c = [_w("alpha"), _w("beta")], [_w("beta")], [_w("alpha"), _w("beta")]
    assert centrality_asym(a, b, c) == -centrality_asym(b, a, c)


def test_centrality_favours_the_more_shared_text() -> None:
    # A shares two words with witness C; B shares none -> A is more central (positive).
    a = [_w("alpha"), _w("beta")]
    b = [_w("zeta"), _w("eta")]
    c = [_w("alpha"), _w("beta")]
    assert centrality_asym(a, b, c) > 0


def test_connective_vote_signs_kai_to_smooth() -> None:
    # X keeps rough καί where Y smooths to δέ -> X is primitive (positive).
    x = [_w("word1"), _conn("και"), _w("word2")]
    y = [_w("word1"), _conn("δε"), _w("word2")]
    assert connective_vote(x, y) > 0
    assert connective_vote(y, x) < 0            # antisymmetric


def test_intro_lateness_antisymmetric() -> None:
    x = [_w("jesus"), _w("filler"), _w("filler"), _w("peter")]
    y = [_w("filler"), _w("filler"), _w("jesus"), _w("peter")]
    assert intro_lateness(x, y) == -intro_lateness(y, x)


def test_pair_features_omits_centrality_without_witness() -> None:
    a, b = [_w("alpha")], [_w("beta")]
    assert "centrality" not in pair_features(a, b)
    assert "centrality" in pair_features(a, b, [_w("alpha")])
