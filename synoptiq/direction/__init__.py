"""Phase-3 Direction Scorer — per-pericope copying-direction probabilities.

The sensor of the SynoptiQ pipeline: given a parallel pericope it emits, per ordered pair,
calibrated ``[A→B, B→A, independent]`` probabilities with abstention, using the robust
agreement-structure signal when a third witness is present (triangulated regime) and weak
connective/fatigue features otherwise (pair-only regime). Its output feeds the Phase-6
Bayesian model comparison (``synoptiq.bayesian.rooting``).
"""

from synoptiq.direction.alignment3 import Column, align_three
from synoptiq.direction.features import (
    AgreementSpectrum,
    agreement_spectrum,
    centrality_asym,
    connective_vote,
    intro_lateness,
    pair_features,
    shared_count,
)
from synoptiq.direction.scorer import DirectionScorer

__all__ = [
    "AgreementSpectrum",
    "Column",
    "DirectionScorer",
    "agreement_spectrum",
    "align_three",
    "centrality_asym",
    "connective_vote",
    "intro_lateness",
    "pair_features",
    "shared_count",
]
