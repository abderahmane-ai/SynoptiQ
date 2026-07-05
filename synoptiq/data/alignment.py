"""Token-level alignment of parallel Gospel passages.

Implements Needleman-Wunsch global alignment using Bio.Align.PairwiseAligner
(the current non-deprecated BioPython API, confirmed as of Biopython ≥1.80).

Comparison key: (normalized lemma, POS) — not surface form. This handles
morphological variation: Matthew might write ἀπεκρίθη (aorist passive) where
Mark writes ἀποκριθείς (aorist passive participle) — same lemma, same POS, so
they should align even though the surface strings differ.

Scoring is deliberately binary. Each distinct (lemma, POS) pair is encoded as
a single Private-Use-Area codepoint, so the aligner reduces to character
identity:

  identical (lemma, POS)  → +2.5  (``match_score``)
  anything else           → -100  (``mismatch_score`` — effectively forbids
                                    aligning non-identical tokens, forcing a gap)
  gap opening             → -5.0
  gap extension           → -0.5

A richer graded matrix (lemma-only, POS-only, surface bonus) was prototyped but
abandoned: a custom PairwiseAligner substitution matrix proved fragile for
non-trivial alphabets. Surface-form agreement is still reported by
``alignment_score`` as a quality statistic; it just does not steer the path.

Usage:
    from synoptiq.data.alignment import align_tokens
    pairs = align_tokens(matthew_tokens, mark_tokens)
    # pairs: list of (idx_in_matthew, idx_in_mark) — None = gap
"""

from __future__ import annotations

from typing import Final

from Bio import Align  # type: ignore[import-untyped]

from synoptiq.utils.greek import normalize_greek
from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import TokenRecord

_LOG = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_GAP_OPEN: Final = -5.0
DEFAULT_GAP_EXTEND: Final = -0.5
DEFAULT_MATCH: Final = 2.5  # score for an identical (lemma, POS) pair
DEFAULT_MISMATCH: Final = -100.0  # effectively forbids aligning non-identical tokens

# Maximum sequence length before we warn (NW is O(n*m))
NW_WARN_LENGTH: Final = 300

# ── Token key helpers ─────────────────────────────────────────────────────────


def _make_token_key(record: TokenRecord) -> tuple[str, str, str]:
    """Compute a (normalized_lemma, pos, normalized_surface) comparison key.

    Args:
        record: A TokenRecord dict.

    Returns:
        3-tuple of (lemma_norm, pos, surface_norm) for scoring.
    """
    lemma_norm = normalize_greek(record.get("lemma") or record.get("normalized", ""))
    pos = record.get("pos", "")
    surface_norm = record.get("normalized", "")
    return lemma_norm, pos, surface_norm


# ── Alignment via PairwiseAligner ─────────────────────────────────────────────


def _indices_from_alignment(
    alignment: Any,  # Bio.Align.Alignment  # noqa: F821
    len_a: int,
    len_b: int,
) -> list[tuple[int | None, int | None]]:
    """Convert a Bio.Align.Alignment object to gap-explicit index pairs.

    Args:
        alignment: A Bio.Align.Alignment result.
        len_a: Length of sequence A.
        len_b: Length of sequence B.

    Returns:
        List of (idx_a, idx_b) pairs where None indicates a gap.
    """
    pairs: list[tuple[int | None, int | None]] = []

    # alignment.aligned returns an array of shape (2, n_aligned_blocks, 2)
    # Each block: ((a_start, a_end), (b_start, b_end))
    try:
        aligned = alignment.aligned
    except AttributeError:
        # Fallback: trivial pairing
        for i in range(min(len_a, len_b)):
            pairs.append((i, i))
        return pairs

    # Walk through aligned blocks to reconstruct full alignment with gaps
    a_pos = 0
    b_pos = 0

    for (a_start, a_end), (b_start, b_end) in zip(aligned[0], aligned[1]):
        # Gap in A (insertions in B)
        while b_pos < b_start:
            pairs.append((None, b_pos))
            b_pos += 1

        # Gap in B (insertions in A)
        while a_pos < a_start:
            pairs.append((a_pos, None))
            a_pos += 1

        # Aligned block
        for a_i, b_i in zip(range(a_start, a_end), range(b_start, b_end)):
            pairs.append((a_i, b_i))
            a_pos = a_i + 1
            b_pos = b_i + 1

    # Trailing gaps
    while a_pos < len_a:
        pairs.append((a_pos, None))
        a_pos += 1
    while b_pos < len_b:
        pairs.append((None, b_pos))
        b_pos += 1

    return pairs


# ── Public API ────────────────────────────────────────────────────────────────


def align_tokens(
    tokens_a: list[TokenRecord],
    tokens_b: list[TokenRecord],
    *,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    match: float = DEFAULT_MATCH,
    mismatch: float = DEFAULT_MISMATCH,
) -> list[tuple[int | None, int | None]]:
    """Align two lists of TokenRecords using Needleman-Wunsch global alignment.

    Uses Bio.Align.PairwiseAligner (non-deprecated API, Biopython ≥1.80). Each
    token is reduced to its (normalized lemma, POS) key and encoded as a single
    Private-Use-Area character, so alignment is exact-match on that key: two
    tokens either share a (lemma, POS) pair (``match``) or they do not
    (``mismatch``). Surface form does not affect the path.

    Args:
        tokens_a: Tokens from the first gospel passage.
        tokens_b: Tokens from the second gospel passage.
        gap_open: Gap opening penalty (default -5.0).
        gap_extend: Gap extension penalty (default -0.5).
        match: Score for an identical (lemma, POS) pair (default 2.5).
        mismatch: Score for non-identical tokens (default -100.0, which
            effectively forbids aligning them and forces a gap instead).

    Returns:
        List of (idx_a, idx_b) pairs. None indicates a gap in that sequence.
        Length of output ≥ max(len(tokens_a), len(tokens_b)).

    Raises:
        ValueError: If either token list is empty.
    """
    if not tokens_a or not tokens_b:
        msg = "Both token sequences must be non-empty"
        raise ValueError(msg)

    len_a = len(tokens_a)
    len_b = len(tokens_b)

    # Warn on very long sequences (quadratic time complexity)
    if len_a > NW_WARN_LENGTH or len_b > NW_WARN_LENGTH:
        _LOG.warning(
            "Long sequences for Needleman-Wunsch — may be slow",
            extra={"len_a": len_a, "len_b": len_b},
        )

    keys_a = [_make_token_key(t) for t in tokens_a]
    keys_b = [_make_token_key(t) for t in tokens_b]

    # Encode each unique (lemma, pos) pair as a single character.
    # We use a simple character encoding rather than a substitution matrix
    # because Bio.Align.PairwiseAligner with a custom matrix is fragile
    # for non-trivial alphabets (mismatched dimensions, shape errors).
    char_map: dict[tuple[str, str], str] = {}
    codepoint = 0xE000  # Private Use Area — doesn't conflict with real text
    max_codepoint = 0xF8FF

    def _encode_key(lemma: str, pos: str) -> str:
        """Map a (lemma, pos) pair to a stable Private-Use-Area character."""
        k = (lemma, pos)
        if k not in char_map:
            if codepoint + len(char_map) > max_codepoint:
                # Fallback: reuse characters for exotic POS combinations
                char_map[k] = chr(codepoint + (len(char_map) % (max_codepoint - codepoint)))
            else:
                char_map[k] = chr(codepoint + len(char_map))
        return char_map[k]

    seq_a = "".join(_encode_key(k[0], k[1]) for k in keys_a)
    seq_b = "".join(_encode_key(k[0], k[1]) for k in keys_b)

    # Configure PairwiseAligner with simple match/mismatch scoring.
    # Each unique (lemma, pos) pair maps to a single character.
    # Same character = exact match → match_score.
    # Different character = no match → mismatch_score.
    # No substitution matrix — just character identity.
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    # Run alignment (returns multiple equally-scored alignments; take best)
    alignments = aligner.align(seq_a, seq_b)

    try:
        best = next(iter(alignments))
    except StopIteration:
        # Should never happen with NW global alignment, but handle gracefully
        _LOG.warning("PairwiseAligner returned no alignments — using trivial pairing")
        return [
            (i if i < len_a else None, i if i < len_b else None) for i in range(max(len_a, len_b))
        ]

    return _indices_from_alignment(best, len_a, len_b)


def alignment_score(
    tokens_a: list[TokenRecord],
    tokens_b: list[TokenRecord],
    pairs: list[tuple[int | None, int | None]],
) -> dict[str, float]:
    """Compute alignment quality statistics for a list of (idx_a, idx_b) pairs.

    Args:
        tokens_a: Tokens from the first sequence.
        tokens_b: Tokens from the second sequence.
        pairs: Output of align_tokens().

    Returns:
        Dict with keys: n_aligned, n_gaps_a, n_gaps_b, lemma_match_rate,
        surface_match_rate, pos_match_rate.
    """
    aligned = [(a, b) for a, b in pairs if a is not None and b is not None]
    gaps_a = sum(1 for a, _ in pairs if a is None)
    gaps_b = sum(1 for _, b in pairs if b is None)

    lemma_matches = 0
    surface_matches = 0
    pos_matches = 0

    for idx_a, idx_b in aligned:
        ta = tokens_a[idx_a]
        tb = tokens_b[idx_b]

        la = normalize_greek(ta.get("lemma", ""))
        lb = normalize_greek(tb.get("lemma", ""))
        if la and lb and la == lb:
            lemma_matches += 1

        sa = ta.get("normalized", "")
        sb = tb.get("normalized", "")
        if sa and sb and sa == sb:
            surface_matches += 1

        pa = ta.get("pos", "")
        pb = tb.get("pos", "")
        if pa and pb and pa == pb:
            pos_matches += 1

    n = max(len(aligned), 1)
    return {
        "n_aligned": len(aligned),
        "n_gaps_a": gaps_a,
        "n_gaps_b": gaps_b,
        "lemma_match_rate": lemma_matches / n,
        "surface_match_rate": surface_matches / n,
        "pos_match_rate": pos_matches / n,
    }
