"""Tests for the grouped/paired bootstrap evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from synoptiq.evaluation.bootstrap import (
    accuracy_ci,
    difference_in_differences,
    paired_accuracy_delta,
    statistic_ci,
)


def test_accuracy_point_estimate_is_exact() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 0])  # 5/6 correct
    res = accuracy_ci(y_true, y_pred, n_resamples=200, seed=0)
    assert res.accuracy == pytest.approx(5 / 6)
    assert res.n_samples == 6
    assert res.n_units == 6  # no groups → each sample is its own unit


def test_perfect_predictions_have_degenerate_ci() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    res = accuracy_ci(y_true, y_true, n_resamples=200, seed=0)
    assert res.accuracy == 1.0
    assert res.ci_low == 1.0
    assert res.ci_high == 1.0
    assert res.boot_std == 0.0


def test_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, size=90)
    y_pred = y_true.copy()
    flip = rng.choice(90, size=30, replace=False)
    y_pred[flip] = (y_pred[flip] + 1) % 3  # ~2/3 correct
    res = accuracy_ci(y_true, y_pred, n_resamples=1000, seed=2)
    assert res.ci_low <= res.accuracy <= res.ci_high
    assert 0.0 <= res.ci_low < res.ci_high <= 1.0


def test_grouped_bootstrap_has_wider_ci_than_ungrouped() -> None:
    # Construct data where each group is internally consistent (all-correct or
    # all-wrong). Grouped resampling then has fewer independent units and a wider
    # interval than pretending every sample is independent.
    n_groups = 12
    per_group = 6
    rng = np.random.default_rng(3)
    group_correct = rng.random(n_groups) < 0.5
    y_true, y_pred, groups = [], [], []
    for g in range(n_groups):
        for _ in range(per_group):
            y_true.append(1)
            y_pred.append(1 if group_correct[g] else 0)
            groups.append(g)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    groups = np.array(groups)

    grouped = accuracy_ci(y_true, y_pred, groups=groups, n_resamples=2000, seed=4)
    ungrouped = accuracy_ci(y_true, y_pred, n_resamples=2000, seed=4)

    assert grouped.n_units == n_groups
    assert ungrouped.n_units == n_groups * per_group
    grouped_width = grouped.ci_high - grouped.ci_low
    ungrouped_width = ungrouped.ci_high - ungrouped.ci_low
    assert grouped_width > ungrouped_width


def test_paired_delta_detects_clear_winner() -> None:
    # Model A is always right, model B always wrong → delta must be +1 everywhere.
    y_true = np.arange(60) % 3
    pred_a = y_true.copy()
    pred_b = (y_true + 1) % 3
    res = paired_accuracy_delta(y_true, pred_a, pred_b, n_resamples=500, seed=5)
    assert res.accuracy_a == 1.0
    assert res.accuracy_b == 0.0
    assert res.delta == pytest.approx(1.0)
    assert res.delta_ci_low == pytest.approx(1.0)
    assert res.prob_a_beats_b == 1.0


def test_paired_delta_symmetric_when_models_equal() -> None:
    y_true = np.arange(60) % 3
    pred = y_true.copy()
    pred[::4] = (pred[::4] + 1) % 3  # both models identical, same errors
    res = paired_accuracy_delta(y_true, pred, pred, n_resamples=500, seed=6)
    assert res.delta == pytest.approx(0.0)
    assert res.delta_ci_low == pytest.approx(0.0)
    assert res.delta_ci_high == pytest.approx(0.0)
    assert res.prob_a_beats_b == 0.0  # strict inequality, ties don't count


def test_grouped_paired_delta_groups_swapped_halves() -> None:
    # Two samples per group (an original + its swap). A gets the group right,
    # B gets it wrong; grouping keeps the pair together across resamples.
    groups = np.repeat(np.arange(10), 2)
    y_true = np.ones(20, dtype=int)
    pred_a = np.ones(20, dtype=int)
    pred_b = np.zeros(20, dtype=int)
    res = paired_accuracy_delta(
        y_true, pred_a, pred_b, groups=groups, n_resamples=300, seed=7,
    )
    assert res.n_units == 10
    assert res.delta == pytest.approx(1.0)


def test_statistic_ci_recovers_positive_mean() -> None:
    rng = np.random.default_rng(10)
    values = rng.normal(0.5, 0.1, size=60)  # clearly positive
    res = statistic_ci(values, n_resamples=2000, seed=11)
    assert res.estimate == pytest.approx(values.mean())
    assert res.ci_low > 0.0
    assert res.prob_positive > 0.99


def test_statistic_ci_null_straddles_zero() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(0.0, 1.0, size=80)
    res = statistic_ci(values, n_resamples=2000, seed=13)
    assert res.ci_low < 0.0 < res.ci_high
    assert 0.2 < res.prob_positive < 0.8


def test_statistic_ci_grouped_is_wider() -> None:
    # Each group internally consistent → fewer independent units → wider CI.
    n_groups, per = 10, 6
    rng = np.random.default_rng(14)
    group_vals = rng.normal(0.3, 0.5, size=n_groups)
    values, groups = [], []
    for g in range(n_groups):
        values.extend([group_vals[g]] * per)
        groups.extend([g] * per)
    values = np.array(values)
    groups = np.array(groups)
    grouped = statistic_ci(values, groups=groups, n_resamples=2000, seed=15)
    flat = statistic_ci(values, n_resamples=2000, seed=15)
    assert grouped.n_units == n_groups
    assert (grouped.ci_high - grouped.ci_low) > (flat.ci_high - flat.ci_low)


def test_did_positive_when_group_a_higher() -> None:
    rng = np.random.default_rng(16)
    a = rng.normal(0.5, 0.1, size=8)   # overlap partition, high lift
    b = rng.normal(0.0, 0.1, size=50)  # rest, no lift
    res = difference_in_differences(a, b, n_resamples=3000, seed=17)
    assert res.estimate == pytest.approx(a.mean() - b.mean())
    assert res.ci_low > 0.0
    assert res.prob_positive > 0.95


def test_did_null_straddles_zero() -> None:
    rng = np.random.default_rng(18)
    a = rng.normal(0.0, 0.2, size=6)
    b = rng.normal(0.0, 0.2, size=55)
    res = difference_in_differences(a, b, n_resamples=3000, seed=19)
    assert res.ci_low < 0.0 < res.ci_high


def test_statistic_ci_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        statistic_ci(np.array([], dtype=float))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        accuracy_ci(np.array([0, 1, 2]), np.array([0, 1]))


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        accuracy_ci(np.array([], dtype=int), np.array([], dtype=int))


def test_groups_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="groups length"):
        accuracy_ci(
            np.array([0, 1, 2]), np.array([0, 1, 2]), groups=np.array([0, 1]),
        )
