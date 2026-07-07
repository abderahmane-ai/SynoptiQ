"""Three-way agreement structure — the robust, theory-neutral direction primitive.

Copying direction among Matthew, Mark, and Luke is identifiable from the *structure of
their word-level agreements*, without any assumption about which readings are "better"
(the fragile part of the canon-based RPM). The four hypotheses are four graphical
structures with distinct, measurable fingerprints in the agreement data:

    2SH        Mark = common source (fork)   -> Mt _|_ Lk | Mark  (Mt-Lk-vs-Mark rare)
    Farrer     chain Mk -> Mt -> Luke         -> Mt-Lk agreement against Mark is systematic
    Griesbach  Mark conflates Mt + Lk (sink)  -> Mark almost never has a reading absent
                                                  from BOTH others (Markan-singular ~ 0)
    Augustinian chain Mt -> Mk -> Luke         -> same CI as 2SH; separated by orientation

To read those fingerprints we need a *positional* three-way alignment (which word lines up
with which across all three), not just pairwise. This module builds one by anchoring on
Mark (the validated agreement hub) and merging the Mark-Matthew and Mark-Luke alignments,
then directly aligning the leftover, non-Markan material so that Matthew-Luke agreements
*against* Mark land in the same column. The agreement spectrum over the columns is the
Goodacre-style count (triple / pairwise / singular / Mt-Lk-against-Mark), computed
automatically over every aligned position.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from synoptiq.data.alignment import align_tokens

Token = Mapping[str, Any]


@dataclass(frozen=True)
class Column:
    """One aligned position across the three gospels; index into each book, or None."""

    mt: int | None
    mk: int | None
    lk: int | None


def _safe_align(a: Sequence[Token], b: Sequence[Token]) -> list[tuple[int | None, int | None]]:
    """align_tokens, but degrade gracefully to all-gaps on empty/unalignable input."""
    if not a and not b:
        return []
    if not a:
        return [(None, j) for j in range(len(b))]
    if not b:
        return [(i, None) for i in range(len(a))]
    try:
        return align_tokens(list(a), list(b))
    except ValueError:
        return [(i, None) for i in range(len(a))] + [(None, j) for j in range(len(b))]


def _index_matches(
    alignment: Sequence[tuple[int | None, int | None]],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """From a Mark-vs-other alignment, return (mark_idx -> other_idx) matches and
    (after-mark_idx -> [other_idx]) inserts (other-only tokens with no Markan partner).

    ``-1`` keys the inserts that precede Mark's first token.
    """
    match: dict[int, int] = {}
    inserts: dict[int, list[int]] = {}
    cur = -1
    for i, j in alignment:          # i = Mark index, j = other-gospel index
        if i is not None:
            cur = i
            if j is not None:
                match[i] = j
        elif j is not None:
            inserts.setdefault(cur, []).append(j)
    return match, inserts


def _emit_inserts(
    cols: list[Column], mt_idx: list[int], lk_idx: list[int],
    mt: Sequence[Token], lk: Sequence[Token],
) -> None:
    """Append columns for non-Markan material, directly aligning Mt vs Lk so that
    Matthew-Luke agreements *against* Mark share a column."""
    if not mt_idx and not lk_idx:
        return
    if mt_idx and lk_idx:
        sub = _safe_align([mt[j] for j in mt_idx], [lk[j] for j in lk_idx])
        for a, b in sub:
            cols.append(Column(mt=mt_idx[a] if a is not None else None, mk=None,
                               lk=lk_idx[b] if b is not None else None))
    else:
        cols.extend(Column(mt=j, mk=None, lk=None) for j in mt_idx)
        cols.extend(Column(mt=None, mk=None, lk=j) for j in lk_idx)


def align_three(
    mt: Sequence[Token], mk: Sequence[Token], lk: Sequence[Token],
) -> list[Column]:
    """Mark-anchored three-way multiple alignment of the parallel passages.

    Returns an ordered list of columns; each column indexes into mt/mk/lk (None = gap).
    """
    mk2mt, mt_ins = _index_matches(_safe_align(mk, mt))
    mk2lk, lk_ins = _index_matches(_safe_align(mk, lk))
    cols: list[Column] = []
    _emit_inserts(cols, mt_ins.get(-1, []), lk_ins.get(-1, []), mt, lk)
    for i in range(len(mk)):
        cols.append(Column(mt=mk2mt.get(i), mk=i, lk=mk2lk.get(i)))
        _emit_inserts(cols, mt_ins.get(i, []), lk_ins.get(i, []), mt, lk)
    return cols


def _lemma(tok: Token) -> str:
    return (tok.get("lemma") or tok.get("normalized") or "").lower()


def _is_content(tok: Token) -> bool:
    return not tok.get("is_punctuation") and bool(_lemma(tok))


@dataclass(frozen=True)
class AgreementSpectrum:
    """Positional agreement counts over a three-way alignment (content words).

    Column classes (mutually exclusive): triple / mt_mk / mk_lk / mt_lk_against_mark / other.
    Singular flags (independent, per book): the book has a reading shared by neither other.
    """

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
        # Exclusive column class.
        if m and k and ll and m == k == ll:
            c["triple"] += 1
        elif m and k and m == k and ll != m:
            c["mt_mk"] += 1
        elif k and ll and k == ll and m != k:
            c["mk_lk"] += 1
        elif m and ll and m == ll and k != m:
            c["mt_lk_against_mark"] += 1
        # Independent singular flags (a reading shared by neither other book).
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
