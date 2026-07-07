"""Phase 6 — pool the DirectionScorer's per-pericope direction probabilities into a
posterior over the four synoptic stemmata.

The four classical hypotheses are four *rootings* of the one three-taxon (Matthew, Mark,
Luke) relationship. Each predicts a definite copying direction for the three pairwise
relationships (or, for 2SH on Matthew-Luke, *independence* — no consistent direction):

    relationship  | 2SH      Farrer    Griesbach  Augustinian
    --------------|-----------------------------------------------
    Matthew-Mark  | Mk->Mt   Mk->Mt    Mt->Mk     Mt->Mk
    Mark-Luke     | Mk->Lk   Mk->Lk    Lk->Mk     Mk->Lk
    Matthew-Luke  | (indep)  Mt->Lk    Mt->Lk     Mt->Lk

This module is the **consumer** of the Phase-3 DirectionScorer: :func:`relationship_counts`
turns a list of per-pericope ``DirectionScores`` into (k, n) per relationship — among the
``n`` non-abstaining pericopes, ``k`` predict the *first-named* book is the source. Each
hypothesis's prediction is then scored by a Beta-Bernoulli **marginal likelihood**:

    * "first book is source" : theta ~ Uniform(0.5, 1)   (a consistent majority > 1/2)
    * "second book is source": theta ~ Uniform(0, 0.5)
    * "independent" (2SH, Mt-Lk): theta = 0.5            (no consistent direction)

These three models share the same (k, n) data, so their marginal likelihoods are directly
comparable — a proper Bayes factor. The stemma posterior is the product of per-relationship
marginal likelihoods times a uniform prior over the four hypotheses.

Non-circularity: the scorer is unsupervised on the synoptics (its triangulation sign is a
fixed prior, its pair-only weights are fit only on external known-direction corpora), so
feeding its output here is not circular.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import betainc, betaln

from synoptiq.utils.types_ import DirectionScores

# Canonical ordering of the three pairwise relationships (first, second).
RELATIONSHIPS: tuple[tuple[str, str], ...] = (
    ("Matthew", "Mark"),
    ("Mark", "Luke"),
    ("Matthew", "Luke"),
)

# Predicted source book per hypothesis per relationship; ``None`` => independent.
HYPOTHESES: dict[str, dict[tuple[str, str], str | None]] = {
    "2SH": {
        ("Matthew", "Mark"): "Mark",
        ("Mark", "Luke"): "Mark",
        ("Matthew", "Luke"): None,
    },
    "Farrer": {
        ("Matthew", "Mark"): "Mark",
        ("Mark", "Luke"): "Mark",
        ("Matthew", "Luke"): "Matthew",
    },
    "Griesbach": {
        ("Matthew", "Mark"): "Matthew",
        ("Mark", "Luke"): "Luke",
        ("Matthew", "Luke"): "Matthew",
    },
    "Augustinian": {
        ("Matthew", "Mark"): "Matthew",
        ("Mark", "Luke"): "Mark",
        ("Matthew", "Luke"): "Matthew",
    },
}


@dataclass(frozen=True)
class RelationshipCount:
    """Pooled directional votes for one pairwise relationship.

    ``k`` = pericopes voting the *first-named* book is the source, among ``n`` non-silent
    pericopes (silent/abstaining pericopes are excluded — they carry no evidence).
    """

    relationship: tuple[str, str]
    k: int
    n: int

    @property
    def frac_first(self) -> float:
        """Fraction of non-silent pericopes voting the first book is the source."""
        return self.k / self.n if self.n else float("nan")


def relationship_log_ml(k: int, n: int, predicted_source: str | None,
                        relationship: tuple[str, str]) -> float:
    """Log marginal likelihood of (k, n) under a hypothesis's predicted direction.

    Args:
        k: votes that the first-named book is the source.
        n: non-silent pericopes.
        predicted_source: the book the hypothesis says is the source, or None (independent).
        relationship: (first, second) book names, to interpret ``predicted_source``.
    """
    if n == 0:
        return 0.0
    if predicted_source is None:                       # independent: theta = 0.5
        return n * np.log(0.5)
    first, _second = relationship
    # Regularized incomplete beta I_0.5(k+1, n-k+1); the half-interval density is 2.
    inc = float(betainc(k + 1, n - k + 1, 0.5))
    log_area_below = np.log(np.clip(inc, 1e-300, 1.0))          # theta in (0, 0.5)
    log_area_above = np.log(np.clip(1.0 - inc, 1e-300, 1.0))    # theta in (0.5, 1)
    base = np.log(2.0) + betaln(k + 1, n - k + 1)
    if predicted_source == first:
        return float(base + log_area_above)
    return float(base + log_area_below)


def posterior_over_stemmata(
    counts: dict[tuple[str, str], RelationshipCount],
    *,
    prior: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Posterior over the four stemmata given pooled per-relationship votes.

    Args:
        counts: relationship -> RelationshipCount (only the relationships present are used;
            a hypothesis is scored on whatever relationships are supplied).
        prior: optional prior over hypothesis names (defaults to uniform).

    Returns:
        {hypothesis: {"log_evidence", "posterior"}} plus a "_relationships" summary.
    """
    names = list(HYPOTHESES)
    if prior is None:
        prior = {h: 1.0 / len(names) for h in names}
    log_ev: dict[str, float] = {}
    for h in names:
        total = 0.0
        for rel, pred in HYPOTHESES[h].items():
            rc = counts.get(rel)
            if rc is None:
                continue
            total += relationship_log_ml(rc.k, rc.n, pred, rel)
        log_ev[h] = total + np.log(prior[h])
    # Normalise in log-space.
    m = max(log_ev.values())
    unnorm = {h: np.exp(log_ev[h] - m) for h in names}
    z = sum(unnorm.values())
    out: dict[str, dict] = {
        h: {"log_evidence": round(log_ev[h], 4), "posterior": round(unnorm[h] / z, 4)}
        for h in names
    }
    out["_relationships"] = {
        f"{r[0]}-{r[1]}": {"k": c.k, "n": c.n, "frac_first": round(c.frac_first, 3)}
        for r, c in counts.items()
    }
    return out


def bayes_factor(counts: dict[tuple[str, str], RelationshipCount],
                 h_num: str, h_den: str) -> float:
    """Bayes factor between two hypotheses on the supplied relationships (num/den)."""
    log_num = sum(relationship_log_ml(c.k, c.n, HYPOTHESES[h_num][r], r)
                  for r, c in counts.items())
    log_den = sum(relationship_log_ml(c.k, c.n, HYPOTHESES[h_den][r], r)
                  for r, c in counts.items())
    return float(np.exp(log_num - log_den))


def _canonical(a: str, b: str) -> tuple[str, str] | None:
    """Map an unordered book pair to its canonical (first, second) relationship."""
    for rel in RELATIONSHIPS:
        if {a, b} == set(rel):
            return rel
    return None


def relationship_counts(
    scores: Sequence[DirectionScores],
) -> dict[tuple[str, str], RelationshipCount]:
    """Turn per-pericope DirectionScores into (k, n) per pairwise relationship.

    ``n`` counts the non-abstaining (``predicted_direction != "independent"``) pericopes for a
    relationship; ``k`` counts those predicting the canonical *first-named* book is the source.
    """
    tally: dict[tuple[str, str], list[int]] = {rel: [0, 0] for rel in RELATIONSHIPS}
    for ds in scores:
        rel = _canonical(ds["book_a"], ds["book_b"])
        if rel is None or ds["predicted_direction"] == "independent":
            continue
        source = ds["book_a"] if ds["predicted_direction"] == "A_to_B" else ds["book_b"]
        tally[rel][1] += 1
        if source == rel[0]:
            tally[rel][0] += 1
    return {rel: RelationshipCount(rel, k, n) for rel, (k, n) in tally.items()}
