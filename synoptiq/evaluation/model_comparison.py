"""Phase-5 model-comparison statistics: lift, difference-in-differences, power.

This module holds the verdict statistics for Track B (source identification) and,
crucially, the **preregistered power analysis** that decides — *before* any
double-tradition text is scored — whether a test can discriminate at the corpus's
sample size. Phase 3 died in part because no one asked that question quantitatively
first; here it is a gate (kill criterion K2 in ``docs/SOURCE_CRITICISM_STUDY.md``).

The power model is deliberately conservative. Each pericope *p* contributes one
signed per-token statistic (e.g. the excess likelihood lift from adding Matthew to
Mark when predicting Luke). Its estimate has two independent sources of spread:

* **within-pericope sampling** — averaging over ``w_p`` tokens shrinks the SD by
  ``1/sqrt(w_p)``, so long pericopes are more precise (heteroscedastic);
* **between-pericope heterogeneity** ``τ`` — the true effect genuinely varies by
  pericope. This is what a *cluster* bootstrap exists to capture; setting τ = 0
  would make power collapse onto raw token count and lie about the real N.

All effects are expressed as a **per-token signal-to-noise ratio** (``snr`` = mean
per-token effect / per-token noise SD), which is dimensionless and needs no
assumption about the absolute nats scale — that scale is fixed empirically by the
gates (G3 noise floor) before these curves are read off.

Pure numpy; no torch, no corpus import — cheap to test and to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ── Results ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PowerResult:
    """Detection rate of a Phase-5 test at a given effect size and sample."""

    snr: float               # per-token signal-to-noise ratio simulated
    between_sd: float        # between-pericope heterogeneity τ (σ units)
    detection_rate: float    # P(test declares the true-signed effect) over sims
    false_positive_rate: float  # detection rate at snr = 0 (should ≈ 1 − ci_level)
    n_units: int
    n_sims: int


@dataclass(frozen=True)
class MDEResult:
    """Minimum detectable effect: smallest snr reaching the target detection rate."""

    mde_snr: float           # np.inf if the target is unreachable on the grid
    target_power: float
    between_sd: float
    n_units: int
    achieved_power: float    # detection rate at mde_snr (>= target if finite)


# ── One-sample lift test (E2 core; also the E1 channel Λ) ─────────────────────


def _draw_unit_stats(
    rng: np.random.Generator,
    weights: NDArray[np.float64],
    snr: float,
    between_sd: float,
    n_sims: int,
) -> NDArray[np.float64]:
    """Simulate the per-pericope statistic for ``n_sims`` experiments.

    Returns an array of shape (n_sims, n_units): each row is one experiment's
    per-pericope statistics, distributed Normal(snr, τ² + 1/w_p) in σ units.
    """
    n_units = weights.shape[0]
    within_var = 1.0 / weights                      # (n_units,)
    total_sd = np.sqrt(between_sd**2 + within_var)  # (n_units,)
    noise = rng.standard_normal((n_sims, n_units)) * total_sd
    return snr + noise


def _bootstrap_ci_low(
    rng: np.random.Generator,
    stats: NDArray[np.float64],
    n_resamples: int,
    ci_level: float,
) -> NDArray[np.float64]:
    """Lower CI bound of the clustered mean for each experiment (vectorised).

    ``stats`` is (n_sims, n_units). For each sim we draw ``n_units`` clusters
    with replacement ``n_resamples`` times and take the requested lower quantile
    of the resample means. Mirrors ``bootstrap.statistic_ci`` but batched.
    """
    n_sims, n_units = stats.shape
    alpha = 1.0 - ci_level
    ci_low = np.empty(n_sims, dtype=np.float64)
    for s in range(n_sims):
        idx = rng.integers(0, n_units, size=(n_resamples, n_units))
        means = stats[s][idx].mean(axis=1)
        ci_low[s] = np.quantile(means, alpha / 2)
    return ci_low


def simulate_lift_power(
    weights: ArrayLike,
    *,
    snr: float,
    between_sd: float = 0.15,
    n_sims: int = 2000,
    sim_resamples: int = 800,
    ci_level: float = 0.95,
    seed: int = 42,
) -> PowerResult:
    """Detection rate of the one-sample lift test at effect ``snr``.

    A test "detects" when the two-sided bootstrap CI of the mean per-pericope
    statistic excludes 0 on the correct side. The false-positive rate is measured
    in the same run at snr = 0 and should sit near ``1 − ci_level``; if it does
    not, the CI is miscalibrated for this cluster structure and the test is unfit.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] == 0:
        msg = "weights must be a non-empty 1-D array"
        raise ValueError(msg)
    if np.any(w <= 0):
        msg = "weights (token counts) must be positive"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    signal = _draw_unit_stats(rng, w, snr, between_sd, n_sims)
    ci_low = _bootstrap_ci_low(rng, signal, sim_resamples, ci_level)
    detection = float(np.mean(ci_low > 0.0))

    null = _draw_unit_stats(rng, w, 0.0, between_sd, n_sims)
    null_low = _bootstrap_ci_low(rng, null, sim_resamples, ci_level)
    null_high_rng = np.random.default_rng(seed + 1)
    # false positive on either side: mean CI excludes 0
    null_means_hi = np.empty(n_sims)
    alpha = 1.0 - ci_level
    for s in range(n_sims):
        idx = null_high_rng.integers(0, w.shape[0], size=(sim_resamples, w.shape[0]))
        means = null[s][idx].mean(axis=1)
        null_means_hi[s] = np.quantile(means, 1.0 - alpha / 2)
    fpr = float(np.mean((null_low > 0.0) | (null_means_hi < 0.0)))

    return PowerResult(
        snr=snr,
        between_sd=between_sd,
        detection_rate=detection,
        false_positive_rate=fpr,
        n_units=w.shape[0],
        n_sims=n_sims,
    )


def mde_lift(
    weights: ArrayLike,
    *,
    between_sd: float = 0.15,
    target_power: float = 0.80,
    snr_grid: ArrayLike | None = None,
    n_sims: int = 2000,
    sim_resamples: int = 800,
    ci_level: float = 0.95,
    seed: int = 42,
) -> MDEResult:
    """Smallest per-token snr whose detection rate reaches ``target_power``."""
    grid = (
        np.asarray(snr_grid, dtype=np.float64)
        if snr_grid is not None
        else np.linspace(0.0, 1.0, 21)
    )
    w = np.asarray(weights, dtype=np.float64)
    best_snr = float("inf")
    best_power = 0.0
    for snr in grid:
        res = simulate_lift_power(
            w,
            snr=float(snr),
            between_sd=between_sd,
            n_sims=n_sims,
            sim_resamples=sim_resamples,
            ci_level=ci_level,
            seed=seed,
        )
        if res.detection_rate >= target_power:
            best_snr = float(snr)
            best_power = res.detection_rate
            break
        best_power = res.detection_rate
    return MDEResult(
        mde_snr=best_snr,
        target_power=target_power,
        between_sd=between_sd,
        n_units=w.shape[0],
        achieved_power=best_power,
    )


# ── Difference-in-differences (E2 overlap-vs-rest) ────────────────────────────


def simulate_did_power(
    overlap_weights: ArrayLike,
    rest_weights: ArrayLike,
    *,
    snr_overlap: float,
    snr_rest: float = 0.0,
    between_sd: float = 0.15,
    n_sims: int = 2000,
    sim_resamples: int = 800,
    ci_level: float = 0.95,
    seed: int = 42,
) -> PowerResult:
    """Detection rate of the E2 difference-in-differences contrast.

    Under 2SH the excess Matthew→Luke lift lives only in the Mark-Q overlap
    partition (``snr_overlap`` > 0, ``snr_rest`` ≈ 0); under Farrer it is uniform
    (both equal). The contrast detects when the CI of
    ``mean(overlap) − mean(rest)`` excludes 0. Because the overlap partition is
    tiny (~5 pericopes) this is the design's true power bottleneck.
    """
    wo = np.asarray(overlap_weights, dtype=np.float64)
    wr = np.asarray(rest_weights, dtype=np.float64)
    for w in (wo, wr):
        if w.ndim != 1 or w.shape[0] == 0 or np.any(w <= 0):
            msg = "weights must be non-empty positive 1-D arrays"
            raise ValueError(msg)

    no, nr = wo.shape[0], wr.shape[0]
    alpha = 1.0 - ci_level

    def _did_excludes_zero(so: float, sr: float, key: int) -> float:
        g = np.random.default_rng(key)
        ov = _draw_unit_stats(g, wo, so, between_sd, n_sims)
        re = _draw_unit_stats(g, wr, sr, between_sd, n_sims)
        excl = np.empty(n_sims, dtype=bool)
        for s in range(n_sims):
            io = g.integers(0, no, size=(sim_resamples, no))
            ir = g.integers(0, nr, size=(sim_resamples, nr))
            deltas = ov[s][io].mean(axis=1) - re[s][ir].mean(axis=1)
            lo, hi = np.quantile(deltas, [alpha / 2, 1.0 - alpha / 2])
            excl[s] = (lo > 0.0) or (hi < 0.0)
        return float(np.mean(excl))

    detection = _did_excludes_zero(snr_overlap, snr_rest, seed)
    fpr = _did_excludes_zero(0.0, 0.0, seed + 7)
    return PowerResult(
        snr=snr_overlap - snr_rest,
        between_sd=between_sd,
        detection_rate=detection,
        false_positive_rate=fpr,
        n_units=no + nr,
        n_sims=n_sims,
    )


def mde_did(
    overlap_weights: ArrayLike,
    rest_weights: ArrayLike,
    *,
    between_sd: float = 0.15,
    target_power: float = 0.80,
    snr_grid: ArrayLike | None = None,
    n_sims: int = 2000,
    sim_resamples: int = 800,
    ci_level: float = 0.95,
    seed: int = 42,
) -> MDEResult:
    """Smallest overlap snr (with rest = 0) whose DiD detection reaches target."""
    grid = (
        np.asarray(snr_grid, dtype=np.float64)
        if snr_grid is not None
        else np.linspace(0.0, 1.5, 31)
    )
    best_snr = float("inf")
    best_power = 0.0
    for snr in grid:
        res = simulate_did_power(
            overlap_weights,
            rest_weights,
            snr_overlap=float(snr),
            snr_rest=0.0,
            between_sd=between_sd,
            n_sims=n_sims,
            sim_resamples=sim_resamples,
            ci_level=ci_level,
            seed=seed,
        )
        if res.detection_rate >= target_power:
            best_snr = float(snr)
            best_power = res.detection_rate
            break
        best_power = res.detection_rate
    return MDEResult(
        mde_snr=best_snr,
        target_power=target_power,
        between_sd=between_sd,
        n_units=int(np.asarray(overlap_weights).shape[0] + np.asarray(rest_weights).shape[0]),
        achieved_power=best_power,
    )


# ── Gate reporting (calibration on known-answer data) ─────────────────────────


@dataclass(frozen=True)
class GateReport:
    """Outcome of one calibration gate (G1–G4 in the design).

    A gate compares the machinery's preferred channel to the ground-truth channel
    on data where both live hypotheses agree on the answer. ``passed`` gates the
    whole phase: no verdict is computed until every mandatory gate passes.
    """

    name: str
    statistic: float         # signed margin; > 0 means the correct channel won
    ci_low: float
    ci_high: float
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "passed": self.passed,
            "detail": self.detail,
        }
