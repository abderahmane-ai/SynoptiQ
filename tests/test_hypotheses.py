"""Tests for the Bayesian synoptic-rooting engine."""

from __future__ import annotations

from synoptiq.bayesian.rooting import (
    HYPOTHESES,
    RelationshipCount,
    bayes_factor,
    posterior_over_stemmata,
    relationship_log_ml,
)


def test_marginal_likelihood_prefers_correct_direction() -> None:
    # 18/20 vote the first book is source => "first is source" should dominate.
    rel = ("Matthew", "Mark")
    ll_first = relationship_log_ml(18, 20, "Matthew", rel)
    ll_second = relationship_log_ml(18, 20, "Mark", rel)
    ll_indep = relationship_log_ml(18, 20, None, rel)
    assert ll_first > ll_indep > ll_second


def test_balanced_data_prefers_independent() -> None:
    # 6/12 (near 0.5) => the independent (theta=0.5) model fits best.
    rel = ("Matthew", "Luke")
    ll_first = relationship_log_ml(6, 12, "Matthew", rel)
    ll_indep = relationship_log_ml(6, 12, None, rel)
    assert ll_indep > ll_first


def test_zero_evidence_is_neutral() -> None:
    rel = ("Mark", "Luke")
    assert relationship_log_ml(0, 0, "Mark", rel) == 0.0
    assert relationship_log_ml(0, 0, None, rel) == 0.0


def test_markan_priority_excludes_matthean_hypotheses() -> None:
    # Mark voted source on both Markan relationships => 2SH/Farrer >> Griesbach/Augustinian.
    counts = {
        ("Matthew", "Mark"): RelationshipCount(("Matthew", "Mark"), 5, 50),   # Mark source
        ("Mark", "Luke"): RelationshipCount(("Mark", "Luke"), 45, 50),        # Mark source
        ("Matthew", "Luke"): RelationshipCount(("Matthew", "Luke"), 6, 12),   # ambiguous
    }
    post = posterior_over_stemmata(counts)
    markan = post["2SH"]["posterior"] + post["Farrer"]["posterior"]
    matthean = post["Griesbach"]["posterior"] + post["Augustinian"]["posterior"]
    assert markan > 0.95
    assert matthean < 0.05


def test_farrer_beats_2sh_when_mt_lk_directional() -> None:
    # A consistent Mt->Lk on the double tradition favours Farrer over 2SH.
    counts = {("Matthew", "Luke"): RelationshipCount(("Matthew", "Luke"), 11, 12)}
    assert bayes_factor(counts, "Farrer", "2SH") > 1.0


def test_posterior_normalises() -> None:
    counts = {("Matthew", "Mark"): RelationshipCount(("Matthew", "Mark"), 10, 30)}
    post = posterior_over_stemmata(counts)
    total = sum(post[h]["posterior"] for h in HYPOTHESES)
    assert abs(total - 1.0) < 1e-9
