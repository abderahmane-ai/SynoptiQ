"""Tests for RPM variant extraction and polarization featurization."""

from __future__ import annotations

from collections import Counter

from synoptiq.data.frequency import FrequencyTable
from synoptiq.evaluation.variants import (
    extract_variants,
    featurize_pair,
    variant_features,
)


def _toks(words: list[str]) -> list[dict]:
    return [{"normalized": w, "text": w} for w in words]


def _freq() -> FrequencyTable:
    # "α" common, "ζ" rare, others mid.
    return FrequencyTable(Counter({"α": 100, "γ": 50, "δ": 40, "ε": 10, "β": 8, "ζ": 1}))


def test_extract_variants_types() -> None:
    x = _toks(["α", "β", "γ", "δ"])
    y = _toks(["α", "ε", "γ", "δ", "ζ"])
    align = [(0, 0), (1, None), (None, 1), (2, 2), (3, 3), (None, 4)]
    variants = extract_variants(x, y, align)
    kinds = [v.kind for v in variants]
    assert kinds == ["substitution", "y_plus"]
    sub = variants[0]
    assert sub.reading_x == ["β"] and sub.reading_y == ["ε"]
    assert variants[1].reading_y == ["ζ"] and variants[1].reading_x == []


def test_x_plus_detected() -> None:
    x = _toks(["α", "β", "γ"])
    y = _toks(["α", "γ"])
    align = [(0, 0), (1, None), (2, 1)]
    variants = extract_variants(x, y, align)
    assert [v.kind for v in variants] == ["x_plus"]
    assert variants[0].reading_x == ["β"]


def test_features_are_antisymmetric_under_swap() -> None:
    freq = _freq()
    x = _toks(["α", "β", "γ", "δ"])
    y = _toks(["α", "ε", "γ", "δ", "ζ"])
    align = [(0, 0), (1, None), (None, 1), (2, 2), (3, 3), (None, 4)]
    swapped = [(j, i) for i, j in align]

    for v_xy, v_yx in zip(extract_variants(x, y, align), extract_variants(y, x, swapped)):
        f_xy = variant_features(v_xy, freq)
        f_yx = variant_features(v_yx, freq)
        for name in ("harder_reading", "local_brevior", "connective_smooth"):
            assert f_xy[name] == -f_yx[name]


def test_harder_reading_sign() -> None:
    # X reads the rare word ζ, Y the common word α -> X harder -> positive (X primitive).
    freq = _freq()
    x = _toks(["ζ"])
    y = _toks(["α"])
    align = [(0, None), (None, 0)]
    v = extract_variants(x, y, align)[0]
    assert variant_features(v, freq)["harder_reading"] > 0


def test_connective_smoothing_sign() -> None:
    # X has rough και, Y has smooth δε -> Y smoothed -> X primitive -> positive.
    freq = _freq()
    x = _toks(["και"])
    y = _toks(["δε"])
    align = [(0, None), (None, 0)]
    v = extract_variants(x, y, align)[0]
    assert variant_features(v, freq)["connective_smooth"] == 1.0


def test_featurize_pair_aggregates() -> None:
    freq = _freq()
    x = _toks(["α", "ζ", "γ"])
    y = _toks(["α", "β", "γ"])
    align = [(0, 0), (1, None), (None, 1), (2, 2)]
    pv = featurize_pair(x, y, align, freq)
    assert len(pv.variants) == 1
    # ζ (rare) vs β (rarer-ish) — just assert the aggregate API works.
    assert isinstance(pv.mean_feature("harder_reading"), float)
