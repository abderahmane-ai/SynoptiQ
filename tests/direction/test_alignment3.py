"""Tests for the three-way multiple alignment and agreement spectrum."""

from __future__ import annotations

from synoptiq.direction.alignment3 import Column, align_three
from synoptiq.direction.features import agreement_spectrum


def _w(lemma: str) -> dict:
    return {"lemma": lemma, "normalized": lemma, "pos": "N-", "is_punctuation": False}


def test_align_three_recovers_mt_lk_against_mark() -> None:
    mk = [_w("alpha"), _w("beta"), _w("gamma"), _w("delta")]
    mt = [_w("alpha"), _w("beta"), _w("xi"), _w("delta")]
    lk = [_w("alpha"), _w("beta"), _w("xi"), _w("delta")]
    sp = agreement_spectrum(align_three(mt, mk, lk), mt, mk, lk)
    assert sp.triple == 3                    # alpha, beta, delta
    assert sp.mt_lk_against_mark == 1        # xi (Mt=Lk, absent from Mark)
    assert sp.mark_singular == 1             # gamma (Mark only)


def test_pairwise_with_mark_detected() -> None:
    mk = [_w("alpha"), _w("gamma"), _w("beta")]
    mt = [_w("alpha"), _w("gamma"), _w("beta")]
    lk = [_w("alpha"), _w("zeta"), _w("beta")]
    sp = agreement_spectrum(align_three(mt, mk, lk), mt, mk, lk)
    assert sp.mt_mk == 1                      # gamma shared by Mt+Mk, not Lk
    assert sp.triple == 2                     # alpha, beta


def test_all_columns_indexable() -> None:
    mk, mt, lk = [_w("a"), _w("b")], [_w("a"), _w("c")], [_w("a"), _w("d")]
    cols = align_three(mt, mk, lk)
    assert all(isinstance(col, Column) for col in cols)
    for col in cols:
        assert col.mt is None or 0 <= col.mt < len(mt)
        assert col.mk is None or 0 <= col.mk < len(mk)
        assert col.lk is None or 0 <= col.lk < len(lk)


def test_empty_mark_still_aligns_mt_lk() -> None:
    mt = [_w("alpha"), _w("beta")]
    lk = [_w("alpha"), _w("beta")]
    sp = agreement_spectrum(align_three(mt, [], lk), mt, [], lk)
    assert sp.mt_lk_against_mark == 2
    assert sp.mark_singular == 0
