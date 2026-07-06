"""Tests for the editorial-fatigue directional features."""

from __future__ import annotations

from synoptiq.evaluation.fatigue import compute_fatigue_features


def _toks(words: list[str]) -> list[dict]:
    return [{"normalized": w, "text": w, "is_punctuation": False} for w in words]


def _align(a: list[dict], b: list[dict]) -> list[tuple[int | None, int | None]]:
    """Trivial identity alignment by shared surface form (enough for feature tests)."""
    out: list[tuple[int | None, int | None]] = []
    used_b: set[int] = set()
    for i, ta in enumerate(a):
        match = next((j for j, tb in enumerate(b)
                      if j not in used_b and tb["normalized"] == ta["normalized"]), None)
        if match is not None:
            used_b.add(match)
            out.append((i, match))
        else:
            out.append((i, None))
    for j in range(len(b)):
        if j not in used_b:
            out.append((None, j))
    return out


def test_features_are_antisymmetric_under_swap() -> None:
    a = _toks("αβρααμ ισαακ ιακωβ ιωσηφ μωυσης δαυειδ σολομων ηλιας ελισαιε".split())
    b = _toks("ισαακ ιακωβ μωυσης δαυειδ σολομων ηλιας ελισαιε αβρααμ ιωσηφ".split())
    align = _align(a, b)
    f_ab = compute_fatigue_features(a, b, align)
    f_ba = compute_fatigue_features(b, a, [(j, i) for i, j in align])
    for attr in ("intro_lateness_asym", "orphan_asym", "coverage_asym"):
        assert getattr(f_ab, attr) == -getattr(f_ba, attr)


def test_intro_lateness_positive_when_copy_introduces_entities_later() -> None:
    # A (source) introduces the names early; B (copy) drops the early intro and only
    # mentions them in its second half -> B introduces them later -> positive.
    names = "αβρααμ ισαακ ιακωβ".split()
    filler = "και δε εν εις προς απο".split()        # function words: not content
    a = _toks(names + filler)                        # names first (early)
    b = _toks(filler + names)                        # names last (late)
    f = compute_fatigue_features(a, b, _align(a, b))
    assert f.intro_lateness_asym > 0
    assert f.n_shared_content == 3


def test_no_shared_content_gives_zero_intro() -> None:
    a = _toks("αβρααμ ισαακ ιακωβ".split())
    b = _toks("μωυσης δαυειδ σολομων".split())
    f = compute_fatigue_features(a, b, _align(a, b))
    assert f.intro_lateness_asym == 0.0
    assert f.n_shared_content == 0


def test_stopwords_and_short_words_excluded() -> None:
    # Only function/short words shared -> no content entities.
    a = _toks("και δε εν εις ο η το".split())
    b = _toks("και δε εν εις ο η το".split())
    f = compute_fatigue_features(a, b, _align(a, b))
    assert f.n_shared_content == 0
