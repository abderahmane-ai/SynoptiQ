"""The DirectionScorer — the Phase-3 sensor.

Given a parallel pericope it emits, per ordered pair, calibrated direction probabilities
``[A→B, B→A, independent]`` with a confidence, abstaining when the evidence is neutral. These
per-pericope probabilities are the observations the Phase-6 Bayesian comparison consumes.

Two regimes (the robust signal needs a third witness):

* **triangulated** — a third gospel is present. The dominant feature is the theory-neutral
  agreement-structure ``centrality`` signal; its sign is fixed a priori (the more central text
  is the more primitive one) and validated, unsupervised, on the synoptics (never fit to
  synoptic labels — required for Phase-6 admissibility).
* **pair_only** — a bare two-text pair (double tradition, external corpora). Falls back to the
  weak connective + fatigue features and abstains heavily.

The score is a swap-antisymmetric linear combination of the signed features (positive => A is
the source); it is squashed into probabilities so that ``|score|`` small => independent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import exp, tanh
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from synoptiq.direction.features import PAIR_ONLY_FEATURES, pair_features
from synoptiq.utils.types_ import Direction, DirectionScores

Token = Mapping[str, Any]

# Prior weight for the triangulation feature: its sign is fixed by the agreement-structure
# argument (more central => more primitive) and is not fit to synoptic labels.
_CENTRALITY_PRIOR = 1.0


@dataclass
class DirectionScorer:
    """Swap-antisymmetric, calibrated, two-regime direction scorer."""

    weights: dict[str, float] = field(default_factory=dict)   # per-feature (pair-only fit)
    scales: dict[str, float] = field(default_factory=dict)    # per-feature std, for scaling
    k: float = 1.0        # logit slope of the direction probability
    tau: float = 0.25     # confidence scale: smaller => more decisive, less abstention

    # ── calibration ────────────────────────────────────────────────────────────────────
    def fit(self, phi: list[dict[str, float]], x_is_source: Sequence[int]) -> DirectionScorer:
        """Calibrate the pair-only feature weights on external known-direction pairs.

        ``phi`` are per-pair feature dicts (X = the first passage); ``x_is_source`` is 1 when
        X is the source. Fit swap-augmented (no intercept) so the scorer stays antisymmetric.
        The triangulation ``centrality`` weight is not fit here — it is a fixed prior.
        """
        names = PAIR_ONLY_FEATURES
        mat = np.array([[f.get(n, 0.0) for n in names] for f in phi], dtype=float)
        self.scales = {n: float(mat[:, i].std() + 1e-9) for i, n in enumerate(names)}
        scaled = mat / np.array([self.scales[n] for n in names])
        y = np.asarray(x_is_source, dtype=int)
        x_aug = np.vstack([scaled, -scaled])
        y_aug = np.concatenate([1 - y, y])         # 1 => X is the copy (standard logistic)
        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
        clf.fit(x_aug, y_aug)
        # decision_function > 0 => "X is copy"; negate so weight>0 => "X is source".
        self.weights = {n: -float(w) for n, w in zip(names, clf.coef_.ravel())}
        self.weights.setdefault("centrality", _CENTRALITY_PRIOR)
        self.scales.setdefault("centrality", 1.0)
        return self

    @classmethod
    def with_priors(cls) -> DirectionScorer:
        """A scorer with sensible prior weights (no fitting) — centrality-led."""
        return cls(
            weights={"centrality": _CENTRALITY_PRIOR, "connective": 0.5, "fatigue": 0.0},
            scales={"centrality": 1.0, "connective": 1.0, "fatigue": 1.0},
        )

    # ── scoring ────────────────────────────────────────────────────────────────────────
    def _raw_score(self, feats: dict[str, float]) -> float:
        """Signed score (positive => A is the source) over whatever features are present."""
        s = 0.0
        for name, val in feats.items():
            w = self.weights.get(name, 0.0)
            sc = self.scales.get(name, 1.0)
            s += w * (val / sc)
        return s

    def _to_scores(self, s: float) -> tuple[float, float, float]:
        """Map a signed score to (prob_a_to_b, prob_b_to_a, prob_independent)."""
        confidence = 1.0 - exp(-abs(s) / self.tau)     # 0 at s=0, -> 1 as |s| grows
        # split the confident mass by the sign of s via a logistic on the score
        p_dir = 0.5 * (1.0 + tanh(self.k * s))         # -> 1 for s>>0 (A_to_B), 0 for s<<0
        return (confidence * p_dir, confidence * (1.0 - p_dir), 1.0 - confidence)

    def score_features(
        self, feats: dict[str, float], regime: str,
    ) -> tuple[float, float, float, float, Direction, str]:
        """Core: features -> (p_a_to_b, p_b_to_a, p_indep, confidence, direction, regime)."""
        s = self._raw_score(feats)
        p_ab, p_ba, p_ind = self._to_scores(s)
        if p_ind >= max(p_ab, p_ba):
            direction: Direction = "independent"
        else:
            direction = "A_to_B" if s > 0 else "B_to_A"
        return p_ab, p_ba, p_ind, 1.0 - p_ind, direction, regime

    def score_pair(
        self, pericope_id: str, book_a: str, tokens_a: Sequence[Token],
        book_b: str, tokens_b: Sequence[Token],
        tokens_c: Sequence[Token] | None = None, freq: Any = None,  # noqa: ANN401
    ) -> DirectionScores:
        """Score one ordered pair; pass the third gospel's tokens for the triangulated regime."""
        feats = pair_features(tokens_a, tokens_b, tokens_c, freq)
        regime = "triangulated" if ("centrality" in feats) else "pair_only"
        p_ab, p_ba, p_ind, conf, direction, regime = self.score_features(feats, regime)
        return DirectionScores(
            pericope_id=pericope_id, book_a=book_a, book_b=book_b,  # type: ignore[typeddict-item]
            prob_a_to_b=round(p_ab, 4), prob_b_to_a=round(p_ba, 4),
            prob_independent=round(p_ind, 4), predicted_direction=direction,
            confidence=round(conf, 4), regime=regime,
        )
