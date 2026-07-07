"""Three-way multiple alignment of the parallel synoptic texts (Matthew, Mark, Luke).

The robust direction signal is the *structure of word-level agreements* across three
witnesses (Abakuks; Goodacre), which requires a positional alignment of all three at once,
not just pairwise. We build one by anchoring on Mark (the validated agreement hub): merge the
Mark-Matthew and Mark-Luke alignments, then directly align the leftover non-Markan material
so that Matthew-Luke agreements *against* Mark land in the same column.

Reuses the pairwise token aligner (:func:`synoptiq.data.alignment.align_tokens`), which
matches on the (normalized-lemma, POS) key and forces mismatches to gaps.
"""

from __future__ import annotations

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
