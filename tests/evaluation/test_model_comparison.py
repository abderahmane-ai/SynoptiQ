"""Tests for Phase-5 model-comparison statistics and the power analysis."""

from __future__ import annotations

import numpy as np
import pytest

from synoptiq.evaluation.model_comparison import (
    GateReport,
    mde_did,
    mde_lift,
    simulate_did_power,
    simulate_lift_power,
)

# Small, fast simulation settings — enough to check qualitative behaviour.
_FAST = {"n_sims": 300, "sim_resamples": 300}


def test_zero_effect_has_low_detection_and_calibrated_fpr() -> None:
    weights = np.full(65, 100.0)
    res = simulate_lift_power(weights, snr=0.0, between_sd=0.15, seed=0, **_FAST)
    # With no signal, detection ≈ two-sided false-positive rate ≈ 1 − ci_level.
    assert res.detection_rate < 0.15
    assert res.false_positive_rate < 0.15


def test_large_effect_is_detected() -> None:
    weights = np.full(65, 100.0)
    res = simulate_lift_power(weights, snr=0.6, between_sd=0.15, seed=1, **_FAST)
    assert res.detection_rate > 0.8


def test_detection_is_monotonic_in_effect() -> None:
    weights = np.full(40, 80.0)
    rates = [
        simulate_lift_power(weights, snr=s, between_sd=0.15, seed=2, **_FAST).detection_rate
        for s in (0.0, 0.2, 0.4, 0.8)
    ]
    # non-decreasing up to Monte-Carlo noise
    assert rates[0] <= rates[1] + 0.05 <= rates[2] + 0.1 <= rates[3] + 0.15


def test_more_units_gives_more_power() -> None:
    # Probe the unsaturated regime: with large pericopes, within-token noise is
    # negligible and power is set by between-pericope heterogeneity + unit count,
    # so we use a small effect and large τ to keep both below ceiling.
    small = simulate_lift_power(np.full(17, 30.0), snr=0.1, between_sd=0.35, seed=3, **_FAST)
    large = simulate_lift_power(np.full(65, 30.0), snr=0.1, between_sd=0.35, seed=3, **_FAST)
    assert large.detection_rate > small.detection_rate


def test_mde_lift_finds_a_threshold() -> None:
    weights = np.full(65, 100.0)
    mde = mde_lift(
        weights,
        between_sd=0.15,
        target_power=0.8,
        snr_grid=np.linspace(0.0, 1.0, 11),
        seed=4,
        **_FAST,
    )
    assert 0.0 < mde.mde_snr < 1.0
    assert mde.achieved_power >= 0.8


def test_mde_unreachable_returns_inf() -> None:
    # One tiny pericope, tiny grid ceiling → target power unreachable.
    mde = mde_lift(
        np.array([3.0]),
        target_power=0.99,
        snr_grid=np.linspace(0.0, 0.05, 3),
        seed=5,
        **_FAST,
    )
    assert np.isinf(mde.mde_snr)


def test_did_detects_overlap_only_signal() -> None:
    overlap = np.full(5, 120.0)
    rest = np.full(60, 120.0)
    res = simulate_did_power(
        overlap, rest, snr_overlap=1.0, snr_rest=0.0, between_sd=0.1, seed=6, **_FAST
    )
    assert res.false_positive_rate < 0.15
    # 5-unit overlap partition is the bottleneck; a strong signal is still detectable
    assert res.detection_rate > 0.4


def test_did_mde_larger_than_one_sample_mde() -> None:
    # The 5-unit DiD needs a bigger per-token effect than the 65-unit lift test.
    weights_all = np.full(65, 100.0)
    lift_mde = mde_lift(
        weights_all, target_power=0.8, snr_grid=np.linspace(0, 1.2, 13), seed=7, **_FAST
    )
    did = mde_did(
        np.full(5, 100.0),
        np.full(60, 100.0),
        target_power=0.8,
        snr_grid=np.linspace(0, 1.5, 16),
        seed=7,
        **_FAST,
    )
    if np.isfinite(did.mde_snr) and np.isfinite(lift_mde.mde_snr):
        assert did.mde_snr >= lift_mde.mde_snr


def test_rejects_nonpositive_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        simulate_lift_power(np.array([100.0, 0.0, 50.0]), snr=0.5, **_FAST)


def test_gate_report_serialises() -> None:
    g = GateReport(name="G1", statistic=0.4, ci_low=0.1, ci_high=0.7, passed=True, detail="ok")
    d = g.to_dict()
    assert d["name"] == "G1"
    assert d["passed"] is True
