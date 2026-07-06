"""Redactional Polarization Model — aggregate per-variant polarizations with abstention.

Per-variant directional signal is sparse and weak (H1: only connective-smoothing polarizes).
The RPM claim is that *aggregating* many weak, independent polarization votes over a pair's
variants yields a confident per-pericope verdict, and that a confidence threshold lets the
scorer ABSTAIN on directionally-neutral pericopes rather than guess.

The model is a linear score over the summed per-variant feature vector:
    S(X, Y) = w . sum_v phi(v)          (positive => X is the source)
Since every phi(v) negates under an A<->B swap, sum_v phi(v) negates too, so S is
antisymmetric by construction. Weights are learned by logistic regression on swap-augmented
real known-direction pairs (no intercept, so the swap symmetry is preserved), and stored so
that a positive score means "X is the source". Confidence is |S|; abstain when |S| < tau.

Deliberately small and interpretable: the learned weights map directly to textual-criticism
canons, so a scholar can read why RPM ruled a given way (and see the length-confounded
lectio-brevior weight collapse toward zero on mixed-polarity training data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PolarizationScorer:
    """Linear, swap-antisymmetric aggregator over summed variant features."""

    feature_names: tuple[str, ...]
    weights: np.ndarray | None = None  # positive-weighted so score>0 => X is source

    def fit(self, phi: np.ndarray, x_is_source: np.ndarray, c: float = 1.0) -> PolarizationScorer:
        """Fit on summed feature vectors.

        Args:
            phi: [N, F] summed per-variant features, one row per (ordered) pair.
            x_is_source: [N] bool/int, 1 if X is the source in that ordering.
            c: inverse L2 regularization strength.
        """
        # Swap-augment so the model cannot exploit a constant: (phi, X-source) and its
        # negation (-phi, X-copy). Label 1 = "X is the copy" for a standard logistic fit.
        x_aug = np.vstack([phi, -phi])
        y_aug = np.concatenate([1 - x_is_source, x_is_source])  # 1 => X is copy
        clf = LogisticRegression(fit_intercept=False, C=c, max_iter=2000)
        clf.fit(x_aug, y_aug)
        # decision_function > 0 => "X is copy"; negate so score>0 => "X is source".
        self.weights = -clf.coef_.ravel()
        return self

    def score(self, phi: np.ndarray) -> np.ndarray:
        """Signed direction score per pair (positive => X is the source)."""
        if self.weights is None:
            msg = "PolarizationScorer must be fit before scoring"
            raise RuntimeError(msg)
        return phi @ self.weights

    def predict_with_abstention(
        self, phi: np.ndarray, tau: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (prediction, abstain_mask).

        prediction: 0 => X is source, 1 => Y is source. abstain_mask True where |score|<=tau.
        """
        s = self.score(phi)
        pred = np.where(s > 0, 0, 1)
        return pred, np.abs(s) <= tau

    def weight_dict(self) -> dict[str, float]:
        """Learned weight per feature (interpretability)."""
        if self.weights is None:
            return {}
        return {n: round(float(w), 4) for n, w in zip(self.feature_names, self.weights)}
