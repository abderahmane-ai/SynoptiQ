"""Archived Phase-3 dead-end investigation (kept for reproducibility, not on the live path).

These modules implement the *global passage-score* approaches to copying-direction detection
that were rigorously tested and found to be confounded with passage length and Markan style
(see ``docs/DIRECTION_SCORER_FINDINGS.md``, "The negative results that forced this design").
They are retained so the negative results in Paper B remain reproducible, but nothing in the
live pipeline imports them. The working approach is the Redactional Polarization Model
(``synoptiq/evaluation/variants.py``, ``synoptiq/models/polarization.py``,
``synoptiq/bayesian/rooting.py``).

Contents:
    direction.py            10-feature swap-equivariant DirectionScorer + MDLDirectionHead
    nll_direction.py        conditional-NLL / MDL codelength direction scoring
    direction_baseline.py   cosine / pooled-embedding logistic-regression baselines
    redaction.py            synthetic same-author, length-decorrelated redaction generator
    direction_training.py   DirectionDataset + DirectionTrainer (was training/direction.py)
"""
