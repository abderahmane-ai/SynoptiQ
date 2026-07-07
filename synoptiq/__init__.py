"""
SynoptiQ — A Multi-Task Neural Source Criticism Framework for the Synoptic Problem.

This package provides tools for:
- Corpus loading and alignment of the Synoptic Gospels
- Domain-adaptive language model pre-training on Koine Greek (KoineFormer)
- Proto-Q reconstruction via Fusion-in-Decoder (QReconstructor)
- Model interpretability via SHAP and Hawkins 1899 comparison

Public API (stable across phases):
    Corpus          — The central data access object
    Book            — Type alias for gospel names
    Tradition       — Type alias for tradition types
    TokenRecord     — TypedDict for a single annotated token
    PericopeAlignment — TypedDict for aligned parallel passages
"""

from __future__ import annotations

from synoptiq._about import __author__, __description__, __license__, __version__

# Corpus is imported lazily to avoid pulling in heavy data deps at package import time.
# Use: from synoptiq.data.corpus import Corpus
# or:  from synoptiq import Corpus  (triggers data module import)
from synoptiq.data.corpus import Corpus
from synoptiq.utils.types_ import (
    Book,
    ConlluToken,
    Genre,
    MorphRecord,
    PericopeAlignment,
    SplitResult,
    TokenRecord,
    Tradition,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    # Data
    "Corpus",
    # Types
    "Book",
    "Tradition",
    "Genre",
    "TokenRecord",
    "MorphRecord",
    "ConlluToken",
    "PericopeAlignment",
    "SplitResult",
]
