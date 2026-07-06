"""Variant extraction and polarization featurization — the core RPM primitive.

Direction is a *rooting* problem (stemmatology): it is decided by polarizing individual
textual variants (which of two readings is primitive), not by averaging a whole passage
into one number (which only ever recovers length or authorial style). This module turns an
aligned pair into a sequence of typed variants and computes, per variant, SIGNED local
features grounded in the directional canons of textual criticism.

Sign convention: **positive => X's reading is the primitive one (X is the source)**. Every
feature negates when X and Y are swapped, so the downstream polarization is antisymmetric.

The token aligner (synoptiq.data.alignment.align_tokens) forces mismatches to gaps, so a
lexical substitution appears as an adjacent run of X-only and Y-only columns; we reconstruct
those into a single ``substitution`` variant, and isolated runs into ``x_plus`` / ``y_plus``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from synoptiq.data.frequency import FrequencyTable
from synoptiq.utils.greek import normalize_greek

# Koine connectives, de-accented. Redactors tend to replace paratactic καί with δέ/οὖν/...
_ROUGH_CONNECTIVES = frozenset({"και"})
_SMOOTH_CONNECTIVES = frozenset({"δε", "ουν", "τοτε", "γαρ", "διο"})

FEATURE_NAMES: tuple[str, ...] = (
    "harder_reading",     # lectio difficilior: harder (rarer) reading is primitive
    "local_brevior",      # lectio brevior (LOCAL): the shorter span is primitive
    "connective_smooth",  # rough καί replaced by a smooth connective => the rough side is primitive
)


@dataclass(frozen=True)
class Variant:
    """One aligned difference between passages X and Y."""

    kind: str                    # "substitution" | "x_plus" | "y_plus"
    reading_x: list[str]         # normalized X-side tokens (empty for y_plus)
    reading_y: list[str]         # normalized Y-side tokens (empty for x_plus)
    position: float              # normalized position [0,1] of the variant in the pair


def _norm(tok: Mapping[str, Any]) -> str:
    """Normalized surface form of a token dict / TokenRecord."""
    return tok.get("normalized") or normalize_greek(tok.get("text", ""))


def extract_variants(
    tokens_x: Sequence[Mapping[str, Any]],
    tokens_y: Sequence[Mapping[str, Any]],
    alignment: Sequence[tuple[int | None, int | None]],
) -> list[Variant]:
    """Reconstruct typed variants from an alignment (match columns end an edit block)."""
    variants: list[Variant] = []
    buf_x: list[str] = []
    buf_y: list[str] = []
    n = max(len(alignment) - 1, 1)
    block_start: float | None = None

    def _flush(pos: float) -> None:
        nonlocal block_start
        if buf_x and buf_y:
            kind = "substitution"
        elif buf_x:
            kind = "x_plus"
        elif buf_y:
            kind = "y_plus"
        else:
            block_start = None
            return
        variants.append(Variant(kind, list(buf_x), list(buf_y), block_start or pos))
        buf_x.clear()
        buf_y.clear()
        block_start = None

    for c, (i, j) in enumerate(alignment):
        pos = c / n
        if i is not None and j is not None:
            _flush(pos)
        else:
            if block_start is None:
                block_start = pos
            if i is not None and i < len(tokens_x):
                buf_x.append(_norm(tokens_x[i]))
            if j is not None and j < len(tokens_y):
                buf_y.append(_norm(tokens_y[j]))
    _flush(1.0)
    return variants


def _mean_markedness(words: list[str], freq: FrequencyTable) -> float:
    return sum(freq.markedness(w) for w in words) / len(words) if words else 0.0


def _connective_signal(v: Variant) -> float:
    """+1 if X has a rough connective where Y has a smooth one (Y smoothed => X primitive)."""
    if v.kind != "substitution":
        return 0.0
    x_rough = any(w in _ROUGH_CONNECTIVES for w in v.reading_x)
    x_smooth = any(w in _SMOOTH_CONNECTIVES for w in v.reading_x)
    y_rough = any(w in _ROUGH_CONNECTIVES for w in v.reading_y)
    y_smooth = any(w in _SMOOTH_CONNECTIVES for w in v.reading_y)
    if x_rough and y_smooth:
        return 1.0
    if y_rough and x_smooth:
        return -1.0
    return 0.0


def variant_features(v: Variant, freq: FrequencyTable) -> dict[str, float]:
    """Signed polarization features for one variant (positive => X is primitive/source).

    - ``harder_reading``: markedness(X) - markedness(Y) for substitutions (lectio difficilior).
    - ``local_brevior``: len(Y) - len(X) at this variant (lectio brevior applied LOCALLY;
      the shorter reading is primitive). Note this is a per-variant span length, not passage
      length — H1 tests whether it is nonetheless just a length proxy.
    - ``connective_smooth``: rough και replaced by a smooth connective.
    """
    if v.kind == "substitution":
        harder = _mean_markedness(v.reading_x, freq) - _mean_markedness(v.reading_y, freq)
    else:
        harder = 0.0
    return {
        "harder_reading": harder,
        "local_brevior": float(len(v.reading_y) - len(v.reading_x)),
        "connective_smooth": _connective_signal(v),
    }


@dataclass
class PairVariants:
    """All variants of a pair plus convenience aggregates for analysis."""

    variants: list[Variant]
    features: list[dict[str, float]] = field(default_factory=list)

    def mean_feature(self, name: str) -> float:
        """Mean of a signed feature over variants (0 if none)."""
        vals = [f[name] for f in self.features]
        return sum(vals) / len(vals) if vals else 0.0

    def sum_feature(self, name: str) -> float:
        """Sum of a signed feature over variants."""
        return sum(f[name] for f in self.features)


def featurize_pair(
    tokens_x: Sequence[Mapping[str, Any]],
    tokens_y: Sequence[Mapping[str, Any]],
    alignment: Sequence[tuple[int | None, int | None]],
    freq: FrequencyTable,
) -> PairVariants:
    """Extract variants and compute their signed features for one aligned pair."""
    variants = extract_variants(tokens_x, tokens_y, alignment)
    feats = [variant_features(v, freq) for v in variants]
    return PairVariants(variants=variants, features=feats)
