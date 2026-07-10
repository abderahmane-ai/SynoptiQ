"""Pericope-grouped bootstrap confidence intervals for corpus evaluation.

Synoptic evaluation has very few independent units: a triple-tradition pericope
produces three book pairs, so several samples share one pericope and are *not*
independent. Point accuracy on a few dozen pericopes therefore has a very wide
sampling distribution, and two models that differ by a few points may be
indistinguishable.

This module provides a *cluster* (grouped) bootstrap that resamples whole
pericopes with replacement, so the reported confidence intervals reflect the real
number of independent units. It also provides a *paired* variant that compares two
models on the same resamples, which is the correct way to ask "is model B actually
better than model A" rather than comparing two independent point estimates.

Nothing here depends on the encoder or torch — it operates on prediction arrays,
so it is cheap to test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and bootstrap confidence interval for one model's accuracy."""

    accuracy: float          # observed accuracy on the full sample
    ci_low: float            # lower bound of the (1 - alpha) interval
    ci_high: float           # upper bound
    boot_mean: float         # mean accuracy across resamples
    boot_std: float          # standard deviation across resamples
    n_units: int             # number of independent units (groups) resampled
    n_samples: int           # number of underlying samples
    ci_level: float          # e.g. 0.95


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired comparison of two models over identical resamples."""

    accuracy_a: float
    accuracy_b: float
    delta: float             # accuracy_a - accuracy_b on the full sample
    delta_ci_low: float
    delta_ci_high: float
    prob_a_beats_b: float    # fraction of resamples where A > B
    n_units: int
    ci_level: float


@dataclass(frozen=True)
class StatisticResult:
    """Clustered bootstrap of a real-valued per-unit statistic (e.g. an NLL delta).

    Unlike :class:`BootstrapResult` (accuracy of 0/1 correctness), this summarises
    the *mean of an arbitrary real statistic* — used for Phase-5 likelihood-lift
    tests, where each pericope contributes a signed nats-per-token difference and
    the sign of the mean is the verdict.
    """

    estimate: float          # observed mean of the statistic on the full sample
    ci_low: float
    ci_high: float
    boot_mean: float
    boot_std: float
    prob_positive: float     # fraction of resamples with mean > 0 (a one-sided p-proxy)
    n_units: int
    n_samples: int
    ci_level: float


def _as_correct(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Return a boolean correctness vector, validating shape agreement."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        msg = f"y_true and y_pred must match: {true.shape} vs {pred.shape}"
        raise ValueError(msg)
    if true.ndim != 1:
        msg = f"expected 1-D arrays, got shape {true.shape}"
        raise ValueError(msg)
    return (true == pred).astype(np.float64)


def _group_index(groups: ArrayLike | None, n: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Map samples to group ids and return per-group sample-index arrays.

    With ``groups=None`` each sample is its own group (ordinary bootstrap).
    """
    if groups is None:
        unit_ids = np.arange(n)
    else:
        unit_ids = np.asarray(groups)
        if unit_ids.shape[0] != n:
            msg = f"groups length {unit_ids.shape[0]} != n_samples {n}"
            raise ValueError(msg)
    unique = list(dict.fromkeys(unit_ids.tolist()))  # preserve first-seen order
    members = [np.flatnonzero(unit_ids == u) for u in unique]
    return np.asarray(unique, dtype=object), members


def accuracy_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Grouped bootstrap accuracy with a percentile confidence interval.

    Args:
        y_true: True labels, shape [N].
        y_pred: Predicted labels, shape [N].
        groups: Group id per sample (e.g. pericope id). ``None`` → per-sample bootstrap.
        n_resamples: Number of bootstrap resamples.
        ci_level: Central interval mass (0.95 → 2.5%/97.5% percentiles).
        seed: RNG seed.

    Returns:
        BootstrapResult with observed accuracy, CI, and resample statistics.
    """
    correct = _as_correct(y_true, y_pred)
    n = correct.shape[0]
    if n == 0:
        msg = "cannot bootstrap an empty sample"
        raise ValueError(msg)
    _, members = _group_index(groups, n)
    n_units = len(members)

    rng = np.random.default_rng(seed)
    accs = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        drawn = rng.integers(0, n_units, size=n_units)
        idx = np.concatenate([members[g] for g in drawn])
        accs[r] = correct[idx].mean()

    alpha = 1.0 - ci_level
    lo, hi = np.quantile(accs, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapResult(
        accuracy=float(correct.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        boot_mean=float(accs.mean()),
        boot_std=float(accs.std()),
        n_units=n_units,
        n_samples=n,
        ci_level=ci_level,
    )


def statistic_ci(
    values: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> StatisticResult:
    """Clustered bootstrap of the mean of a real-valued per-sample statistic.

    This is the workhorse for Phase-5 verdicts: pass one signed value per sample
    (typically one per pericope, e.g. the excess likelihood lift ``Δ_p − Δ̃_p``),
    optionally grouped, and get a percentile CI on the mean plus the fraction of
    resamples in which the mean is strictly positive.

    Args:
        values: Real-valued statistic per sample, shape [N].
        groups: Group id per sample (cluster unit). ``None`` → per-sample bootstrap.
        n_resamples: Number of bootstrap resamples.
        ci_level: Central interval mass.
        seed: RNG seed.

    Returns:
        StatisticResult with the observed mean, CI, and ``prob_positive``.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 1:
        msg = f"expected 1-D values, got shape {vals.shape}"
        raise ValueError(msg)
    n = vals.shape[0]
    if n == 0:
        msg = "cannot bootstrap an empty sample"
        raise ValueError(msg)
    _, members = _group_index(groups, n)
    n_units = len(members)

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        drawn = rng.integers(0, n_units, size=n_units)
        idx = np.concatenate([members[g] for g in drawn])
        means[r] = vals[idx].mean()

    alpha = 1.0 - ci_level
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return StatisticResult(
        estimate=float(vals.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        boot_mean=float(means.mean()),
        boot_std=float(means.std()),
        prob_positive=float((means > 0).mean()),
        n_units=n_units,
        n_samples=n,
        ci_level=ci_level,
    )


def difference_in_differences(
    values_a: ArrayLike,
    values_b: ArrayLike,
    *,
    groups_a: ArrayLike | None = None,
    groups_b: ArrayLike | None = None,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> StatisticResult:
    """Clustered bootstrap of ``mean(values_a) − mean(values_b)`` for two disjoint groups.

    The E2 difference-in-differences statistic: ``values_a`` is the per-pericope
    likelihood lift on Mark-Q overlap pericopes, ``values_b`` the lift on the
    rest. A positive delta with a CI excluding 0 means the lift concentrates in
    the overlap partition — the 2SH prediction. Because the two partitions are
    resampled independently each iteration, the interval reflects the (small)
    number of overlap units, which is the real power bottleneck.

    Returns:
        StatisticResult whose ``estimate`` is the difference of means and
        ``n_units`` the total number of resampled clusters across both groups.
    """
    va = np.asarray(values_a, dtype=np.float64)
    vb = np.asarray(values_b, dtype=np.float64)
    if va.ndim != 1 or vb.ndim != 1:
        msg = "expected 1-D value arrays"
        raise ValueError(msg)
    if va.shape[0] == 0 or vb.shape[0] == 0:
        msg = "cannot bootstrap an empty group"
        raise ValueError(msg)
    _, members_a = _group_index(groups_a, va.shape[0])
    _, members_b = _group_index(groups_b, vb.shape[0])
    na, nb = len(members_a), len(members_b)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        ia = np.concatenate([members_a[g] for g in rng.integers(0, na, size=na)])
        ib = np.concatenate([members_b[g] for g in rng.integers(0, nb, size=nb)])
        deltas[r] = va[ia].mean() - vb[ib].mean()

    alpha = 1.0 - ci_level
    lo, hi = np.quantile(deltas, [alpha / 2, 1.0 - alpha / 2])
    return StatisticResult(
        estimate=float(va.mean() - vb.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        boot_mean=float(deltas.mean()),
        boot_std=float(deltas.std()),
        prob_positive=float((deltas > 0).mean()),
        n_units=na + nb,
        n_samples=va.shape[0] + vb.shape[0],
        ci_level=ci_level,
    )


def paired_accuracy_delta(
    y_true: ArrayLike,
    y_pred_a: ArrayLike,
    y_pred_b: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Paired grouped bootstrap of the accuracy difference (A minus B).

    Both models are scored on the *same* resampled units each iteration, so the
    delta interval accounts for their correlated errors. This is the correct test
    for "does A beat B" on a shared evaluation set.

    Args:
        y_true: True labels, shape [N].
        y_pred_a: Model A predictions, shape [N].
        y_pred_b: Model B predictions, shape [N].
        groups: Group id per sample. ``None`` → per-sample bootstrap.
        n_resamples: Number of bootstrap resamples.
        ci_level: Central interval mass for the delta.
        seed: RNG seed.

    Returns:
        PairedBootstrapResult with per-model accuracy, delta, delta CI, and the
        fraction of resamples in which A strictly beats B.
    """
    correct_a = _as_correct(y_true, y_pred_a)
    correct_b = _as_correct(y_true, y_pred_b)
    n = correct_a.shape[0]
    if n == 0:
        msg = "cannot bootstrap an empty sample"
        raise ValueError(msg)
    if correct_b.shape[0] != n:
        msg = "y_pred_a and y_pred_b must have equal length"
        raise ValueError(msg)
    _, members = _group_index(groups, n)
    n_units = len(members)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        drawn = rng.integers(0, n_units, size=n_units)
        idx = np.concatenate([members[g] for g in drawn])
        deltas[r] = correct_a[idx].mean() - correct_b[idx].mean()

    alpha = 1.0 - ci_level
    lo, hi = np.quantile(deltas, [alpha / 2, 1.0 - alpha / 2])
    return PairedBootstrapResult(
        accuracy_a=float(correct_a.mean()),
        accuracy_b=float(correct_b.mean()),
        delta=float(correct_a.mean() - correct_b.mean()),
        delta_ci_low=float(lo),
        delta_ci_high=float(hi),
        prob_a_beats_b=float((deltas > 0).mean()),
        n_units=n_units,
        ci_level=ci_level,
    )
