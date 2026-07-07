"""Directional features for the DirectionScorer — signed, positive => A is the source.

Consolidates every validated direction signal in one place:

* **Triangulation** (needs a third witness): the *centrality asymmetry* — the text that
  agrees more with the third gospel is the more central / primitive one. This is the robust,
  theory-neutral agreement-structure signal (Abakuks/Goodacre) that recovered Markan priority.
* **Connective smoothing** (pair-only): rough ``καί`` replaced by a smooth connective marks
  the rough side primitive. Kept as one *demoted* feature — it is partly Markan-style.
* **Editorial fatigue** (pair-only): shared entities introduced later in the copy. Genre-
  limited (chance on the synoptics), retained as a weak feature.

Also exposes the three-way **agreement spectrum** (triple / pairwise / singular /
Mt-Lk-against-Mark) used by the report and the Phase-6 topology check.

Dropped after validation: ``local_brevior`` (pure length) and ``harder_reading`` (chance).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from synoptiq.data.alignment import align_tokens
from synoptiq.data.frequency import FrequencyTable
from synoptiq.direction.alignment3 import Column
from synoptiq.utils.greek import normalize_greek

Token = Mapping[str, Any]

# Koine connectives, de-accented. Redactors replace paratactic καί with δέ/οὖν/τότε/...
_ROUGH_CONNECTIVES = frozenset({"και"})
_SMOOTH_CONNECTIVES = frozenset({"δε", "ουν", "τοτε", "γαρ", "διο"})


def _lemma(tok: Token) -> str:
    return (tok.get("lemma") or tok.get("normalized") or "").lower()


def _is_content(tok: Token) -> bool:
    return not tok.get("is_punctuation") and bool(_lemma(tok))


def _norm(tok: Token) -> str:
    return tok.get("normalized") or normalize_greek(tok.get("text", ""))


# ── Shared-word counts (basis of the triangulation feature) ─────────────────────────────

def shared_count(a: Sequence[Token], b: Sequence[Token]) -> int:
    """Number of content words A and B share, via the (lemma, POS) token alignment."""
    if len(a) < 1 or len(b) < 1:
        return 0
    try:
        al = align_tokens(list(a), list(b))
    except ValueError:
        return 0
    n = 0
    for i, j in al:
        if i is None or j is None:
            continue
        if i < len(a) and j < len(b) and _is_content(a[i]) and _lemma(a[i]) == _lemma(b[j]):
            n += 1
    return n


def centrality_asym(a: Sequence[Token], b: Sequence[Token], c: Sequence[Token]) -> float:
    """Signed centrality of A vs B relative to the third witness C (positive => A more central).

    The more primitive text of a pair agrees more with an independent third witness. Returns
    a length-fair ratio in [-1, 1]: (sh(A,C) - sh(B,C)) / (sh(A,C) + sh(B,C) + 1).
    """
    sac, sbc = shared_count(a, c), shared_count(b, c)
    return (sac - sbc) / (sac + sbc + 1)


# ── Connective-smoothing canon (pair-only, demoted) ─────────────────────────────────────

@dataclass(frozen=True)
class Variant:
    """One aligned difference between passages X and Y."""

    kind: str                    # "substitution" | "x_plus" | "y_plus"
    reading_x: list[str]
    reading_y: list[str]
    position: float


def extract_variants(
    tokens_x: Sequence[Token], tokens_y: Sequence[Token],
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

    for col, (i, j) in enumerate(alignment):
        pos = col / n
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


def connective_vote(
    tokens_x: Sequence[Token], tokens_y: Sequence[Token], freq: FrequencyTable | None = None,
) -> float:
    """Summed connective-smoothing vote for a pair (positive => X primitive). freq unused
    (kept for signature symmetry with the dropped markedness canon)."""
    _ = freq
    if len(tokens_x) < 1 or len(tokens_y) < 1:
        return 0.0
    try:
        al = align_tokens(list(tokens_x), list(tokens_y))
    except ValueError:
        return 0.0
    return sum(_connective_signal(v) for v in extract_variants(tokens_x, tokens_y, al))


# ── Editorial fatigue (pair-only, genre-limited) ────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "αυτος", "αυτου", "αυτω", "αυτον", "αυτοι", "αυτων", "αυτοις", "αυτους",
    "αυτη", "αυτης", "αυται", "αυτας", "εαυτου", "εαυτον", "εαυτων",
    "ουτος", "ουτου", "τουτο", "τουτου", "τουτον", "ταυτα", "τουτων", "τουτοις",
    "εκεινος", "εκεινου", "εκεινων", "οστις", "ητις", "οτι", "ινα", "εαν",
    "ουτως", "καθως", "ωστε", "αλλα", "δια", "κατα", "μετα", "περι", "υπο",
    "επι", "παρα", "συν", "προς", "απο", "εκ", "εις", "εν", "ανα", "προ",
    "ουκ", "ουχ", "μη", "μεν", "τε", "γαρ", "ουν", "και", "δε",
})


def _fatigue_key(tok: Token) -> str | None:
    if tok.get("is_punctuation"):
        return None
    key = (tok.get("lemma") or tok.get("normalized") or "").lower()
    if len(key) < 4 or key in _STOPWORDS:
        return None
    return key


def _first_positions(tokens: Sequence[Token]) -> dict[str, float]:
    n = max(len(tokens) - 1, 1)
    first: dict[str, float] = {}
    for i, tok in enumerate(tokens):
        key = _fatigue_key(tok)
        if key is not None and key not in first:
            first[key] = i / n
    return first


def intro_lateness(tokens_x: Sequence[Token], tokens_y: Sequence[Token]) -> float:
    """Signed fatigue feature (positive => X is the source): shared entities introduced later
    in Y than in X (an abbreviating copy drops early introductions)."""
    fx, fy = _first_positions(tokens_x), _first_positions(tokens_y)
    shared = set(fx) & set(fy)
    if not shared:
        return 0.0
    return sum(fy[k] - fx[k] for k in shared) / len(shared)


# ── Three-way agreement spectrum (report + Phase-6 topology) ────────────────────────────

@dataclass(frozen=True)
class AgreementSpectrum:
    """Positional agreement counts over a three-way alignment (content words)."""

    triple: int
    mt_mk: int
    mk_lk: int
    mt_lk_against_mark: int
    mark_singular: int
    matthew_singular: int
    luke_singular: int
    n_content_columns: int

    def as_dict(self) -> dict[str, int]:
        """Counts as a plain dict."""
        return {
            "triple": self.triple, "mt_mk": self.mt_mk, "mk_lk": self.mk_lk,
            "mt_lk_against_mark": self.mt_lk_against_mark,
            "mark_singular": self.mark_singular, "matthew_singular": self.matthew_singular,
            "luke_singular": self.luke_singular, "n_content_columns": self.n_content_columns,
        }


def agreement_spectrum(
    cols: Sequence[Column], mt: Sequence[Token], mk: Sequence[Token], lk: Sequence[Token],
) -> AgreementSpectrum:
    """Count the Goodacre-style agreement spectrum over a three-way alignment."""
    c: Counter[str] = Counter()

    def lem(book: Sequence[Token], idx: int | None) -> str | None:
        if idx is None:
            return None
        tok = book[idx]
        return _lemma(tok) if _is_content(tok) else None

    for col in cols:
        m, k, ll = lem(mt, col.mt), lem(mk, col.mk), lem(lk, col.lk)
        if m is None and k is None and ll is None:
            continue
        c["n"] += 1
        if m and k and ll and m == k == ll:
            c["triple"] += 1
        elif m and k and m == k and ll != m:
            c["mt_mk"] += 1
        elif k and ll and k == ll and m != k:
            c["mk_lk"] += 1
        elif m and ll and m == ll and k != m:
            c["mt_lk_against_mark"] += 1
        if k and m != k and ll != k:
            c["mark_sing"] += 1
        if m and k != m and ll != m:
            c["mt_sing"] += 1
        if ll and k != ll and m != ll:
            c["lk_sing"] += 1

    return AgreementSpectrum(
        triple=c["triple"], mt_mk=c["mt_mk"], mk_lk=c["mk_lk"],
        mt_lk_against_mark=c["mt_lk_against_mark"], mark_singular=c["mark_sing"],
        matthew_singular=c["mt_sing"], luke_singular=c["lk_sing"], n_content_columns=c["n"],
    )


# ── Unified per-pair feature vector (what the scorer consumes) ───────────────────────────

# Feature order is fixed; every feature is signed so that positive => A (the first passage)
# is the source, and negates under an A<->B swap (keeps the scorer antisymmetric).
TRIANGULATED_FEATURES: tuple[str, ...] = ("centrality", "connective", "fatigue")
PAIR_ONLY_FEATURES: tuple[str, ...] = ("connective", "fatigue")


def pair_features(
    tokens_a: Sequence[Token], tokens_b: Sequence[Token],
    tokens_c: Sequence[Token] | None = None, freq: FrequencyTable | None = None,
) -> dict[str, float]:
    """Signed directional features for ordered pair (A, B); positive => A is the source.

    Includes the triangulation ``centrality`` feature only when a third witness C is given.
    """
    feats = {
        "connective": connective_vote(tokens_a, tokens_b, freq),
        "fatigue": intro_lateness(tokens_a, tokens_b),
    }
    if tokens_c is not None and len(tokens_c) > 0:
        feats["centrality"] = centrality_asym(tokens_a, tokens_b, tokens_c)
    return feats


@dataclass
class PairVariants:
    """All variants of a pair plus convenience aggregates (used by the findings report)."""

    variants: list[Variant]
    features: list[dict[str, float]] = field(default_factory=list)

    def sum_feature(self, name: str) -> float:
        """Sum of a signed feature over variants."""
        return sum(f[name] for f in self.features)
