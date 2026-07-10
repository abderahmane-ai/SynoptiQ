"""Tests for the Track-B verdict statistics (pure, no model)."""

from __future__ import annotations

import numpy as np
import pytest

from synoptiq.evaluation.verdict import (
    channel_recovery_gate,
    did_contrast,
    excess_lift_values,
    minor_agreement_test,
    null_threshold,
)


def test_excess_lift_is_control_minus_added() -> None:
    base = np.array([5.0, 6.0])
    added = np.array([3.0, 3.5])
    control = np.array([4.0, 5.0])
    ex = excess_lift_values(base, added, control)
    # control - added
    assert np.allclose(ex, [1.0, 1.5])


def test_excess_lift_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        excess_lift_values(np.zeros(3), np.zeros(2), np.zeros(3))


def test_mai_test_detects_real_lift() -> None:
    rng = np.random.default_rng(0)
    n = 40
    base = rng.normal(6.0, 0.5, n)
    added = base - rng.normal(1.0, 0.1, n)     # real witness helps by ~1 nat
    control = base - rng.normal(0.2, 0.1, n)   # control helps only a little
    v = minor_agreement_test(base, added, control, n_resamples=2000, seed=1)
    assert v.significant                        # excess > 0
    assert v.excess.estimate == pytest.approx(0.8, abs=0.1)
    assert v.raw_lift_mean > v.control_lift_mean


def test_mai_test_null_when_witness_equals_control() -> None:
    rng = np.random.default_rng(2)
    n = 40
    base = rng.normal(6.0, 0.5, n)
    added = base - rng.normal(0.5, 0.2, n)
    control = base - rng.normal(0.5, 0.2, n)   # same help as the witness → excess ~ 0
    v = minor_agreement_test(base, added, control, n_resamples=2000, seed=3)
    assert not v.significant
    assert v.excess.ci_low < 0.0 < v.excess.ci_high


def test_did_contrast_positive_when_overlap_higher() -> None:
    rng = np.random.default_rng(4)
    overlap = rng.normal(0.6, 0.1, 6)   # strong excess on overlap
    rest = rng.normal(0.0, 0.1, 55)     # none on rest
    res = did_contrast(overlap, rest, n_resamples=3000, seed=5)
    assert res.ci_low > 0.0
    assert res.estimate == pytest.approx(0.6, abs=0.15)


def test_null_threshold_from_controls() -> None:
    rng = np.random.default_rng(6)
    null = rng.normal(0.0, 0.1, 200)
    gt = null_threshold(null, ci_level=0.95)
    # 95th percentile of N(0, 0.1) ≈ 0.164
    assert 0.10 < gt.threshold < 0.22
    assert gt.null_mean == pytest.approx(0.0, abs=0.03)
    assert gt.n_units == 200


def test_null_threshold_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        null_threshold(np.array([]))


def test_channel_recovery_gate_passes_when_correct_channel_wins() -> None:
    rng = np.random.default_rng(7)
    correct = rng.normal(2.0, 0.2, 30)   # low NLL — the true channel
    wrong = rng.normal(3.0, 0.2, 30)     # higher NLL — the wrong channel
    g = channel_recovery_gate("G1", correct, wrong, n_resamples=2000, seed=8)
    assert g.passed
    assert g.margin.estimate == pytest.approx(1.0, abs=0.15)
    assert g.to_dict()["passed"] is True


def test_channel_recovery_gate_fails_when_indistinguishable() -> None:
    rng = np.random.default_rng(9)
    a = rng.normal(2.5, 0.3, 30)
    b = rng.normal(2.5, 0.3, 30)
    g = channel_recovery_gate("G2", a, b, n_resamples=2000, seed=10)
    assert not g.passed


def test_channel_recovery_gate_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        channel_recovery_gate("G1", np.zeros(3), np.zeros(2))
