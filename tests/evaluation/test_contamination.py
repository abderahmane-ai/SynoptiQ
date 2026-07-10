"""Tests for the contamination-audit statistics (no model / torch needed)."""

from __future__ import annotations

import math

import pytest

from synoptiq.evaluation.contamination import (
    ContaminationReport,
    GroupScore,
    ease_ratio,
    exact_match_rate,
    memorization_gap,
    score_group,
)


def test_score_group_token_weighted() -> None:
    # Two chunks: 2.0 nats over 10 tokens, 4.0 nats over 30 tokens.
    g = score_group("gospel", [2.0, 4.0], [10, 30])
    expected_mean = (2.0 * 10 + 4.0 * 30) / 40
    assert g.mean_token_nll == pytest.approx(expected_mean)
    assert g.perplexity == pytest.approx(math.exp(expected_mean))
    assert g.n_tokens == 40
    assert g.n_chunks == 2


def test_score_group_validates_lengths_and_emptiness() -> None:
    with pytest.raises(ValueError, match="differ"):
        score_group("g", [1.0], [1, 2])
    with pytest.raises(ValueError, match="zero tokens"):
        score_group("g", [1.0], [0])


def test_ease_ratio_gt_one_when_gospel_easier() -> None:
    gospel = GroupScore("gospel", 5.0, math.log(5.0), n_chunks=1, n_tokens=1)
    control = GroupScore("control", 20.0, math.log(20.0), n_chunks=1, n_tokens=1)
    assert ease_ratio(gospel, control) == pytest.approx(4.0)


def _grp(name: str, ppl: float) -> GroupScore:
    return GroupScore(name, perplexity=ppl, mean_token_nll=math.log(ppl), n_chunks=1, n_tokens=100)


def test_memorization_gap_positive_when_original_memorized() -> None:
    # Original: very low gospel ppl (memorized) but normal control.
    # NS: normal on both. Gap should be large and positive.
    orig_g, orig_c = _grp("g", 2.0), _grp("c", 20.0)
    ns_g, ns_c = _grp("g", 18.0), _grp("c", 20.0)
    gap = memorization_gap(orig_g, orig_c, ns_g, ns_c)
    assert gap == pytest.approx(math.log(18.0) - math.log(2.0), abs=1e-9)
    assert gap > 0.25


def test_memorization_gap_zero_when_no_contamination() -> None:
    # Both models equally good on gospels and control → gap ~ 0.
    orig_g, orig_c = _grp("g", 15.0), _grp("c", 20.0)
    ns_g, ns_c = _grp("g", 15.0), _grp("c", 20.0)
    assert memorization_gap(orig_g, orig_c, ns_g, ns_c) == pytest.approx(0.0, abs=1e-9)


def test_exact_match_rate_normalizes_whitespace_and_case() -> None:
    preds = ["ὁ  λόγος", "FOO", "miss"]
    golds = ["ὁ λόγος", "foo", "hit"]
    assert exact_match_rate(preds, golds) == pytest.approx(2 / 3)


def test_exact_match_rate_validates_and_handles_empty() -> None:
    assert exact_match_rate([], []) == 0.0
    with pytest.raises(ValueError, match="differ"):
        exact_match_rate(["a"], ["a", "b"])


def test_report_paired_flags_on_gap() -> None:
    orig = ContaminationReport(
        orig_gospel=_grp("g", 2.0), orig_control=_grp("c", 20.0),
        ns_gospel=_grp("g", 18.0), ns_control=_grp("c", 20.0),
    )
    assert orig.paired
    assert orig.flagged
    assert "MEMORIZATION DETECTED" in orig.to_markdown()
    assert orig.to_dict()["memorization_gap"] > 0.25


def test_report_paired_clean_not_flagged() -> None:
    rep = ContaminationReport(
        orig_gospel=_grp("g", 15.0), orig_control=_grp("c", 20.0),
        ns_gospel=_grp("g", 15.0), ns_control=_grp("c", 20.0),
    )
    assert not rep.flagged
    assert "no material memorization" in rep.to_markdown()


def test_report_single_model_uses_exact_match_then_ease() -> None:
    # No NS groups → falls back to exact-match probe if present.
    rep = ContaminationReport(
        orig_gospel=_grp("g", 5.0), orig_control=_grp("c", 6.0),
        exact_match_gospel=0.4, exact_match_control=0.0,
    )
    assert not rep.paired
    assert rep.gap is None
    assert rep.flagged  # 0.4 > 0.10 exact-match threshold
    assert "preliminary" in rep.to_markdown()


def test_report_renders_with_gospel_em_but_no_control_em() -> None:
    # Paired audit path sets exact_match_control=None; markdown must not crash.
    rep = ContaminationReport(
        orig_gospel=_grp("g", 11.0), orig_control=_grp("c", 15.0),
        ns_gospel=_grp("g", 16.0), ns_control=_grp("c", 15.0),
        exact_match_gospel=0.07, exact_match_control=None,
    )
    md = rep.to_markdown()
    assert "control: n/a" in md
    assert "gospel: 7.0%" in md


def test_report_single_model_ease_ratio_fallback() -> None:
    # No NS, no exact match → ease ratio decides.
    flagged = ContaminationReport(orig_gospel=_grp("g", 5.0), orig_control=_grp("c", 20.0))
    clean = ContaminationReport(orig_gospel=_grp("g", 18.0), orig_control=_grp("c", 20.0))
    assert flagged.flagged      # ratio 4.0 > 1.5
    assert not clean.flagged    # ratio ~1.1
