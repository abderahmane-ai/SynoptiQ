"""Tests for the two-regime DirectionScorer."""

from __future__ import annotations

from synoptiq.direction.scorer import DirectionScorer


def _w(lemma: str) -> dict:
    return {"lemma": lemma, "normalized": lemma, "pos": "N-", "is_punctuation": False}


def test_triangulated_regime_when_witness_present() -> None:
    sc = DirectionScorer.with_priors()
    a, b, c = [_w("a"), _w("b")], [_w("z")], [_w("a"), _w("b")]
    res = sc.score_pair("001", "Mark", a, "Matthew", b, tokens_c=c)
    assert res["regime"] == "triangulated"
    assert res["predicted_direction"] == "A_to_B"      # A is more central => source


def test_pair_only_regime_without_witness() -> None:
    sc = DirectionScorer.with_priors()
    res = sc.score_pair("001", "Matthew", [_w("a")], "Luke", [_w("b")])
    assert res["regime"] == "pair_only"


def test_probabilities_sum_to_one() -> None:
    sc = DirectionScorer.with_priors()
    r = sc.score_pair("001", "Mark", [_w("a"), _w("b")], "Matthew", [_w("z")],
                      tokens_c=[_w("a"), _w("b")])
    assert abs(r["prob_a_to_b"] + r["prob_b_to_a"] + r["prob_independent"] - 1.0) < 1e-6
    assert abs(r["confidence"] - (1.0 - r["prob_independent"])) < 1e-6


def test_neutral_evidence_abstains() -> None:
    sc = DirectionScorer.with_priors()
    # Identical A and B, identical witness -> no directional signal -> independent.
    a = [_w("a"), _w("b")]
    r = sc.score_pair("001", "Mark", a, "Matthew", list(a), tokens_c=list(a))
    assert r["predicted_direction"] == "independent"


def test_fit_calibrates_pair_features() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    phi = [{"connective": float(rng.normal(1.0, 0.5)), "fatigue": float(rng.normal(0, 1))}
           for _ in range(100)]
    sc = DirectionScorer().fit(phi, [1] * 100)         # connective positive => X source
    assert sc.weights["connective"] > 0
    assert "centrality" in sc.weights                  # prior added
