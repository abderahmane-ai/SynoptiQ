"""Tests for the three-way agreement-structure primitive."""

from __future__ import annotations

from synoptiq.evaluation.agreement import Column, agreement_spectrum, align_three


def _w(lemma: str) -> dict:
    return {"lemma": lemma, "normalized": lemma, "pos": "N-", "is_punctuation": False}


def test_align_three_recovers_mt_lk_against_mark() -> None:
    # Mark: alpha beta gamma delta ; Mt & Lk both replace gamma with xi (agree vs Mark).
    mk = [_w("alpha"), _w("beta"), _w("gamma"), _w("delta")]
    mt = [_w("alpha"), _w("beta"), _w("xi"), _w("delta")]
    lk = [_w("alpha"), _w("beta"), _w("xi"), _w("delta")]
    cols = align_three(mt, mk, lk)
    sp = agreement_spectrum(cols, mt, mk, lk)
    assert sp.triple == 3                    # alpha, beta, delta
    assert sp.mt_lk_against_mark == 1        # xi (Mt=Lk, absent from Mark)
    assert sp.mark_singular == 1             # gamma (Mark only)


def test_pairwise_with_mark_detected() -> None:
    # Matthew keeps Mark's gamma; Luke changes it -> a Mark-Matthew pairwise agreement.
    mk = [_w("alpha"), _w("gamma"), _w("beta")]
    mt = [_w("alpha"), _w("gamma"), _w("beta")]
    lk = [_w("alpha"), _w("zeta"), _w("beta")]
    sp = agreement_spectrum(align_three(mt, mk, lk), mt, mk, lk)
    assert sp.mt_mk == 1                      # gamma shared by Mt+Mk, not Lk
    assert sp.triple == 2                     # alpha, beta


def test_all_columns_indexable() -> None:
    mk = [_w("a"), _w("b")]
    mt = [_w("a"), _w("c")]
    lk = [_w("a"), _w("d")]
    cols = align_three(mt, mk, lk)
    assert all(isinstance(c, Column) for c in cols)
    for c in cols:  # every non-None index is in range
        assert c.mt is None or 0 <= c.mt < len(mt)
        assert c.mk is None or 0 <= c.mk < len(mk)
        assert c.lk is None or 0 <= c.lk < len(lk)


def test_empty_mark_still_aligns_mt_lk() -> None:
    # No Markan parallel (double-tradition case): Mt-Lk agreement still surfaces.
    mt = [_w("alpha"), _w("beta")]
    lk = [_w("alpha"), _w("beta")]
    sp = agreement_spectrum(align_three(mt, [], lk), mt, [], lk)
    assert sp.mt_lk_against_mark == 2
    assert sp.mark_singular == 0
