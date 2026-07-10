"""Reconstruction-quality metrics for Track A (rebuild Mark from Matthew+Luke).

The headline number is how well the Fusion-in-Decoder reproduces held-out Mark, scored
against the gold text with a bag-of-tokens F1 (order-insensitive, robust to the word-order
freedom of Greek) plus exact-match. Normalisation lets the same code report surface-level
and lemma-level (accent-stripped) agreement. Pericope-grouped means feed the cluster
bootstrap in ``synoptiq.evaluation.bootstrap`` for confidence intervals.

Pure functions over strings; no torch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass


def _tokens(text: str, normalize: Callable[[str], str] | None) -> list[str]:
    toks = text.split()
    if normalize is not None:
        toks = [normalize(t) for t in toks]
    return [t for t in toks if t]


def token_f1(pred: str, gold: str, *, normalize: Callable[[str], str] | None = None) -> float:
    """Bag-of-tokens F1 between a prediction and gold (multiset overlap).

    Order-insensitive: counts how many tokens overlap (with multiplicity), then
    combines precision (overlap / pred length) and recall (overlap / gold length).
    Empty gold and empty pred → 1.0; one empty and the other not → 0.0.
    """
    p = Counter(_tokens(pred, normalize))
    g = Counter(_tokens(gold, normalize))
    n_pred, n_gold = sum(p.values()), sum(g.values())
    if n_pred == 0 and n_gold == 0:
        return 1.0
    if n_pred == 0 or n_gold == 0:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision = overlap / n_pred
    recall = overlap / n_gold
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str, *, normalize: Callable[[str], str] | None = None) -> float:
    """1.0 if the normalised token sequences are identical, else 0.0."""
    return float(_tokens(pred, normalize) == _tokens(gold, normalize))


@dataclass(frozen=True)
class ReconstructionResult:
    """Aggregate reconstruction quality over a set of pericopes."""

    mean_f1: float
    mean_exact_match: float
    n: int
    per_example_f1: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_f1": self.mean_f1,
            "mean_exact_match": self.mean_exact_match,
            "n": self.n,
        }


def evaluate_reconstruction(
    predictions: Sequence[str],
    golds: Sequence[str],
    *,
    normalize: Callable[[str], str] | None = None,
) -> ReconstructionResult:
    """Mean token-F1 and exact-match across aligned (prediction, gold) pairs.

    ``per_example_f1`` is returned so callers can cluster-bootstrap it by pericope.
    """
    if len(predictions) != len(golds):
        msg = f"predictions ({len(predictions)}) and golds ({len(golds)}) differ"
        raise ValueError(msg)
    if not predictions:
        return ReconstructionResult(0.0, 0.0, 0, [])
    f1s = [token_f1(p, g, normalize=normalize) for p, g in zip(predictions, golds, strict=True)]
    ems = [exact_match(p, g, normalize=normalize) for p, g in zip(predictions, golds, strict=True)]
    return ReconstructionResult(
        mean_f1=sum(f1s) / len(f1s),
        mean_exact_match=sum(ems) / len(ems),
        n=len(f1s),
        per_example_f1=f1s,
    )


def nearest_witness_baseline(
    witness_texts: Sequence[str], gold: str, *, normalize: Callable[[str], str] | None = None
) -> float:
    """Best token-F1 achievable by copying a single witness verbatim.

    The reconstruction must beat this to justify the fusion machinery — copying
    Matthew (or Luke) straight through is the trivial baseline.
    """
    return max((token_f1(w, gold, normalize=normalize) for w in witness_texts), default=0.0)
