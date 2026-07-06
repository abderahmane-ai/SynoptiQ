"""Editorial-fatigue directional features from an aligned passage pair.

Motivation. Every global score I tried (similarity, NLL, MDL) collapsed to *length*
or *Markan style*, because averaging a whole passage into one number drowns the few
tell-tale edits that actually carry direction. Editorial fatigue (Goodacre 1998) is a
different kind of signal: a copyist makes characteristic changes early, then tires and
reverts to the source — leaving an *internal inconsistency in the copy* that only makes
sense one way. Crucially it is a WITHIN-pair, position-normalized signal, so global
length and global style are constants within a pair and cannot confound it.

This module computes SIGNED directional features from a token alignment. Sign convention:
**positive => A is the source (A_to_B).** Each feature negates when A and B are swapped.
The features are deliberately position-normalized ([0,1] within each text) so absolute
length carries no information — that is the whole point.

Inputs are generic: `tokens_*` are lists of dicts/TokenRecords exposing at least
``normalized`` (and optionally ``lemma``/``is_punctuation``); ``alignment`` is the list
of (i|None, j|None) index pairs produced by synoptiq.data.alignment.align_tokens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Longer Greek function words (de-accented, lowercase) to exclude from "content".
# Short function words (και, δε, εν, εις, ο, η ...) are already dropped by the length
# filter; this set catches the longer pronoun/preposition forms.
_STOPWORDS: frozenset[str] = frozenset({
    "αυτος", "αυτου", "αυτω", "αυτον", "αυτοι", "αυτων", "αυτοις", "αυτους",
    "αυτη", "αυτης", "αυται", "αυτας", "εαυτου", "εαυτον", "εαυτων",
    "ουτος", "ουτου", "τουτο", "τουτου", "τουτον", "ταυτα", "τουτων", "τουτοις",
    "εκεινος", "εκεινου", "εκεινων", "οστις", "ητις", "οτι", "ινα", "εαν",
    "ουτως", "καθως", "ωστε", "αλλα", "δια", "κατα", "μετα", "περι", "υπο",
    "επι", "παρα", "συν", "προς", "απο", "εκ", "εις", "εν", "ανα", "προ",
    "ουκ", "ουχ", "μη", "μεν", "τε", "γαρ", "ουν", "και", "δε",
})


def _content_key(tok: Mapping[str, Any]) -> str | None:
    """Return the content-word key for a token, or None if it is a function word.

    Uses ``lemma`` when present (synoptic tokens), else ``normalized`` (external
    tokens). Filters punctuation, very short tokens, and the stopword set.
    """
    if tok.get("is_punctuation"):
        return None
    key = (tok.get("lemma") or tok.get("normalized") or "").lower()
    if len(key) < 4 or key in _STOPWORDS:
        return None
    return key


def _first_positions(tokens: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Map each content key to the normalized position [0,1] of its FIRST occurrence."""
    n = max(len(tokens) - 1, 1)
    first: dict[str, float] = {}
    for i, tok in enumerate(tokens):
        key = _content_key(tok)
        if key is not None and key not in first:
            first[key] = i / n
    return first


def _content_positions(tokens: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Map each content key to all its normalized positions [0,1]."""
    n = max(len(tokens) - 1, 1)
    out: dict[str, list[float]] = {}
    for i, tok in enumerate(tokens):
        key = _content_key(tok)
        if key is not None:
            out.setdefault(key, []).append(i / n)
    return out


@dataclass(frozen=True)
class FatigueFeatures:
    """Signed directional fatigue features (positive => A is the source)."""

    intro_lateness_asym: float   # shared entities introduced later in the copy
    orphan_asym: float           # dangling late references whose antecedent the copy dropped
    coverage_asym: float         # asymmetry of content retention
    n_shared_content: int        # support: number of shared content entities
    edit_frontloading: float     # symmetric descriptor: <0 => edits are front-loaded (fatigue-like)

    def as_dict(self) -> dict[str, float]:
        """Feature dict for tabulation."""
        return {
            "intro_lateness_asym": self.intro_lateness_asym,
            "orphan_asym": self.orphan_asym,
            "coverage_asym": self.coverage_asym,
            "edit_frontloading": self.edit_frontloading,
            "n_shared_content": float(self.n_shared_content),
        }


def compute_fatigue_features(
    tokens_a: Sequence[Mapping[str, Any]],
    tokens_b: Sequence[Mapping[str, Any]],
    alignment: Sequence[tuple[int | None, int | None]],
) -> FatigueFeatures:
    """Compute signed directional fatigue features for one aligned pair.

    Sign convention: positive => A is the source. Every signed feature negates under an
    A<->B swap, and all are normalized by within-text position so length is not used.
    """
    first_a = _first_positions(tokens_a)
    first_b = _first_positions(tokens_b)
    pos_a = _content_positions(tokens_a)
    pos_b = _content_positions(tokens_b)
    shared = set(first_a) & set(first_b)

    # ── intro_lateness_asym ────────────────────────────────────────────────
    # Shared entities that the copy introduces LATER than the source: an abbreviator
    # drops early (introductory) mentions but keeps later ones. mean(first_B - first_A);
    # positive => B introduces shared entities later => B is the copy => A is source.
    if shared:
        intro = sum(first_b[k] - first_a[k] for k in shared) / len(shared)
    else:
        intro = 0.0

    # ── orphan_asym (Goodacre dangling reference) ──────────────────────────
    # A "late orphan" in text X = a content entity appearing only in X's SECOND half
    # while the OTHER text introduced it in its FIRST half: X dropped the antecedent
    # but kept the later reference. The copy accrues more orphans. orphan_A - orphan_B;
    # positive => A has more dangling refs => A is the copy => sign is checked empirically.
    def _orphans(pos_x: dict[str, list[float]], first_other: dict[str, float]) -> int:
        n = 0
        for key, positions in pos_x.items():
            if min(positions) > 0.5 and first_other.get(key, 1.0) <= 0.5:
                n += 1
        return n

    orphan_a = _orphans(pos_a, first_b)
    orphan_b = _orphans(pos_b, first_a)
    orphan = float(orphan_a - orphan_b)

    # ── coverage_asym ──────────────────────────────────────────────────────
    # Fraction of each text's content entities that the other reproduces. A faithful
    # copy reproduces most of its source's content; a source is not fully reproduced by
    # a selective copy. cov(B in A) - cov(A in B); sign checked empirically.
    ca = len(set(first_a))
    cb = len(set(first_b))
    cov_b_in_a = len(shared) / cb if cb else 0.0
    cov_a_in_b = len(shared) / ca if ca else 0.0
    coverage = cov_b_in_a - cov_a_in_b

    # ── edit_frontloading (symmetric descriptor) ───────────────────────────
    # Is non-verbatim editing concentrated early? Correlate a per-column "is-edit"
    # indicator with its normalized position across the alignment. Negative => edits
    # front-loaded => consistent with a fatigued copy relationship (not directional).
    edit_frontloading = _edit_frontloading(alignment)

    return FatigueFeatures(
        intro_lateness_asym=intro,
        orphan_asym=orphan,
        coverage_asym=coverage,
        n_shared_content=len(shared),
        edit_frontloading=edit_frontloading,
    )


def _edit_frontloading(alignment: Sequence[tuple[int | None, int | None]]) -> float:
    """Correlation between column position and being an edit (gap) column.

    Returns 0 if degenerate. Negative means edits cluster early (fatigue-like).
    """
    if len(alignment) < 3:
        return 0.0
    positions = []
    is_edit = []
    n = len(alignment) - 1
    for c, (i, j) in enumerate(alignment):
        positions.append(c / n)
        is_edit.append(1.0 if (i is None or j is None) else 0.0)
    # Pearson correlation without numpy dependency.
    m_p = sum(positions) / len(positions)
    m_e = sum(is_edit) / len(is_edit)
    cov = sum((p - m_p) * (e - m_e) for p, e in zip(positions, is_edit))
    var_p = sum((p - m_p) ** 2 for p in positions)
    var_e = sum((e - m_e) ** 2 for e in is_edit)
    if var_p <= 0 or var_e <= 0:
        return 0.0
    return cov / (var_p * var_e) ** 0.5
