"""Tests for the Redactional Polarization aggregation model."""

from __future__ import annotations

import numpy as np

from synoptiq.models.polarization import PolarizationScorer


def _fit_scorer() -> PolarizationScorer:
    # Two features; feature 1 cleanly indicates X-source (positive), feature 0 is noise.
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(1.0, 0.5, size=n)          # positive when X is source
    noise = rng.normal(0.0, 1.0, size=n)
    phi = np.column_stack([noise, signal])
    x_is_source = np.ones(n, dtype=int)
    return PolarizationScorer(("noise", "signal")).fit(phi, x_is_source)


def test_score_is_antisymmetric() -> None:
    scorer = _fit_scorer()
    phi = np.array([[0.3, 1.2], [-0.5, 0.8]])
    np.testing.assert_allclose(scorer.score(phi), -scorer.score(-phi), atol=1e-9)


def test_learns_positive_weight_on_signal() -> None:
    scorer = _fit_scorer()
    w = scorer.weight_dict()
    assert w["signal"] > 0
    assert abs(w["signal"]) > abs(w["noise"])  # signal dominates noise


def test_predicts_x_source_for_positive_signal() -> None:
    scorer = _fit_scorer()
    phi = np.array([[0.0, 2.0]])          # strong X-source evidence
    pred, abstain = scorer.predict_with_abstention(phi, tau=0.0)
    assert pred[0] == 0                    # 0 => X is source
    assert not abstain[0]


def test_abstention_masks_low_confidence() -> None:
    scorer = _fit_scorer()
    phi = np.array([[0.0, 2.0], [0.0, 0.001]])
    _pred, abstain = scorer.predict_with_abstention(phi, tau=0.05)
    assert not abstain[0]                  # confident
    assert abstain[1]                      # near-zero score -> abstain


def test_score_before_fit_raises() -> None:
    import pytest
    with pytest.raises(RuntimeError, match="fit"):
        PolarizationScorer(("a", "b")).score(np.zeros((1, 2)))
