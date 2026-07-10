"""Verdict statistics: excess-lift, difference-in-differences, and gate thresholds.

This is the decision core for Track B (source identification, M3–M5). Every input is an
array of teacher-forced NLLs produced by the validated scoring path (`redactor.score` /
`fid.score`); nothing here touches a model or a GPU, so it is fully unit-tested.

The E2 primitive is the **excess lift**: does adding witness X to the context reduce the
NLL of the target *beyond what a matched control context achieves*?

    lift_p          = NLL(y | base)          − NLL(y | base, X)          # X helps
    control_lift_p  = NLL(y | base)          − NLL(y | base, Ctrl)       # a control helps
    excess_p        = lift_p − control_lift_p = NLL(y | base, Ctrl) − NLL(y | base, X)

``excess_p > 0`` means the *real* added witness beats a control context of matched length
— genuine information beyond the base, not a length/style artifact (the control cancels
those). The verdict is the pericope-clustered bootstrap of ``excess_p``.

The **difference-in-differences** contrasts excess lift on the Mark-Q overlap partition
vs the rest — the single most confound-resistant statistic (genre/style/model pathologies
hit both partitions equally). The **gate threshold** (G3) is the null noise floor: run the
excess-lift machinery where no real signal should exist and read off its 95th percentile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from synoptiq.evaluation.bootstrap import (
    StatisticResult,
    difference_in_differences,
    statistic_ci,
)


def excess_lift_values(
    base_nll: ArrayLike,
    added_nll: ArrayLike,
    control_nll: ArrayLike,
) -> np.ndarray:
    """Per-pericope excess lift ``NLL(y|base,Ctrl) − NLL(y|base,X)``.

    Equivalently ``lift − control_lift``. All three inputs are per-pericope NLLs
    (nats/token) of the *same* target, so length and target style cancel.
    """
    base = np.asarray(base_nll, dtype=np.float64)
    added = np.asarray(added_nll, dtype=np.float64)
    control = np.asarray(control_nll, dtype=np.float64)
    if not (base.shape == added.shape == control.shape):
        msg = f"shape mismatch: base {base.shape}, added {added.shape}, control {control.shape}"
        raise ValueError(msg)
    return control - added  # == (base - added) - (base - control)


@dataclass(frozen=True)
class LiftVerdict:
    """E2 excess-lift verdict with its clustered bootstrap interval."""

    excess: StatisticResult   # clustered bootstrap of mean excess lift (nats/token)
    raw_lift_mean: float      # mean (base − added): how much the witness helps, unadjusted
    control_lift_mean: float  # mean (base − control): how much a control helps

    @property
    def significant(self) -> bool:
        """CI excludes 0 on the positive side (witness beats control)."""
        return self.excess.ci_low > 0.0


def minor_agreement_test(
    base_nll: ArrayLike,
    added_nll: ArrayLike,
    control_nll: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> LiftVerdict:
    """E2: is the excess lift from the real added witness > 0 (clustered over pericopes)?"""
    base = np.asarray(base_nll, dtype=np.float64)
    added = np.asarray(added_nll, dtype=np.float64)
    control = np.asarray(control_nll, dtype=np.float64)
    excess = excess_lift_values(base, added, control)
    res = statistic_ci(excess, groups=groups, n_resamples=n_resamples, ci_level=ci_level, seed=seed)
    return LiftVerdict(
        excess=res,
        raw_lift_mean=float(np.mean(base - added)),
        control_lift_mean=float(np.mean(base - control)),
    )


def did_contrast(
    overlap_excess: ArrayLike,
    rest_excess: ArrayLike,
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> StatisticResult:
    """E2 difference-in-differences: excess lift on Mark-Q overlap minus on the rest.

    Positive with a CI excluding 0 ⇒ the information concentrates in the overlap
    partition (the Two-Source prediction). Uniform (delta ≈ 0) ⇒ Farrer/MPH-like.
    """
    return difference_in_differences(
        overlap_excess, rest_excess,
        n_resamples=n_resamples, ci_level=ci_level, seed=seed,
    )


@dataclass(frozen=True)
class GateThreshold:
    """G3 null noise floor: the excess-lift value a real signal must clear."""

    threshold: float          # 1-sided (ci_level) quantile of the null excess distribution
    null_mean: float
    null_std: float
    n_units: int
    ci_level: float


def null_threshold(
    null_excess: ArrayLike,
    *,
    ci_level: float = 0.95,
) -> GateThreshold:
    """Derive the G3 claim threshold from excess lifts computed on negative controls.

    ``null_excess`` should be per-pericope excess lifts where *no* real signal exists
    (e.g. two independent control contexts). A real E2 effect counts only if it exceeds
    this floor's ``ci_level`` quantile — this is the pre-registered, data-derived threshold.
    """
    vals = np.asarray(null_excess, dtype=np.float64)
    if vals.ndim != 1 or vals.shape[0] == 0:
        msg = "null_excess must be a non-empty 1-D array"
        raise ValueError(msg)
    return GateThreshold(
        threshold=float(np.quantile(vals, ci_level)),
        null_mean=float(vals.mean()),
        null_std=float(vals.std()),
        n_units=vals.shape[0],
        ci_level=ci_level,
    )


@dataclass(frozen=True)
class GateOutcome:
    """Result of a calibration gate (G1/G2): did the machinery recover the known answer?"""

    name: str
    margin: StatisticResult   # signed margin (correct channel − wrong channel), clustered
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "margin": self.margin.estimate,
            "ci": [self.margin.ci_low, self.margin.ci_high],
            "passed": self.passed,
            "detail": self.detail,
        }


def channel_recovery_gate(
    name: str,
    correct_channel_nll: ArrayLike,
    wrong_channel_nll: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    n_resamples: int = 10_000,
    seed: int = 42,
    detail: str = "",
) -> GateOutcome:
    """G1/G2: the correct channel must assign lower NLL than the wrong one on known cases.

    ``margin_p = wrong − correct`` (nats/token); a positive clustered mean whose CI
    excludes 0 means the machinery prefers the channel both live hypotheses agree is true.
    """
    correct = np.asarray(correct_channel_nll, dtype=np.float64)
    wrong = np.asarray(wrong_channel_nll, dtype=np.float64)
    if correct.shape != wrong.shape:
        msg = f"shape mismatch: correct {correct.shape}, wrong {wrong.shape}"
        raise ValueError(msg)
    margin = statistic_ci(wrong - correct, groups=groups, n_resamples=n_resamples, seed=seed)
    return GateOutcome(name=name, margin=margin, passed=margin.ci_low > 0.0, detail=detail)
