"""Data augmentation for SynoptiQ training data.

Three augmentation strategies for the direction scorer training data:

1. Bootstrap resampling (BaggingDataset):
   Sample pericopes with replacement to create N training variants.
   Increases effective dataset size without fabricating new examples.

2. Moving window sub-sampling:
   Create sub-pericope examples by sliding a window over the token sequence.
   A copyist's behavior is consistent within a pericope; sub-sequences
   should still show direction asymmetry. Increases training density.

3. Scribal noise injection (synthetic examples):
   Apply controlled text corruption to known-direction pairs to create
   synthetic training examples. Noise types:
   - Haplography: delete one of two adjacent identical (or near-identical) tokens
   - Dittography: duplicate a token (or adjacent token pair)
   - Transposition: swap two adjacent tokens

   Error rate is set low (1%) to mimic attested scribal error frequencies,
   not to randomly corrupt text. We generate ~500 synthetic pairs total.
   These are labeled with the same direction as their uncorrupted source.
"""

from __future__ import annotations

from copy import deepcopy
import random

from synoptiq.utils.types_ import TokenRecord

# ── Bootstrap resampling ──────────────────────────────────────────────────────


def bootstrap_pericopes(
    pericope_ids: list[str],
    *,
    n_samples: int = 100,
    seed: int = 42,
) -> list[list[str]]:
    """Generate bootstrap samples (with replacement) of pericope IDs.

    Args:
        pericope_ids: List of pericope IDs to sample from.
        n_samples: Number of bootstrap samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of n_samples lists, each of the same length as pericope_ids,
        drawn with replacement.

    Example:
        >>> samples = bootstrap_pericopes(["001", "002", "003"], n_samples=3)
        >>> len(samples)
        3
        >>> len(samples[0])
        3
    """
    rng = random.Random(seed)
    n = len(pericope_ids)
    return [[rng.choice(pericope_ids) for _ in range(n)] for _ in range(n_samples)]


# ── Moving window sub-sampling ────────────────────────────────────────────────


def sliding_windows(
    tokens: list[TokenRecord],
    *,
    window_size: int = 5,
    stride: int = 2,
    min_window_size: int = 3,
) -> list[list[TokenRecord]]:
    """Slide a window over a token sequence to generate sub-pericope examples.

    Args:
        tokens: List of TokenRecords from one book's version of a pericope.
        window_size: Number of tokens per window (default 5 verses ~ 50 tokens).
        stride: Step between window starts (default 2).
        min_window_size: Minimum tokens for a valid window.

    Returns:
        List of token sublists. May be empty if sequence is too short.

    Example:
        >>> windows = sliding_windows(tokens, window_size=3, stride=1)
    """
    if len(tokens) < min_window_size:
        return [tokens] if tokens else []

    windows: list[list[TokenRecord]] = []
    start = 0
    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        window = tokens[start:end]
        if len(window) >= min_window_size:
            windows.append(window)
        start += stride

    return windows


# ── Scribal noise injection ───────────────────────────────────────────────────


def add_scribal_noise(
    tokens: list[TokenRecord],
    *,
    error_rate: float = 0.01,
    seed: int | None = None,
) -> list[TokenRecord]:
    """Apply simulated scribal errors to a token sequence.

    Error types and their probabilities (conditional on an error occurring):
      - Haplography (0.40): Delete one of two adjacent near-identical tokens.
        Condition: normalized forms have edit distance ≤ 2.
      - Dittography (0.35): Duplicate a randomly selected token.
      - Transposition (0.25): Swap two adjacent tokens.

    The error_rate controls how many tokens may be affected total.
    At 1%, a 100-token pericope has ~1 error on average.

    IMPORTANT: This function returns a COPY of the token list.
    The original is never modified.

    Args:
        tokens: List of TokenRecords to corrupt.
        error_rate: Expected fraction of tokens to corrupt (default 0.01).
        seed: Optional random seed.

    Returns:
        New list with scribal errors applied. Length may differ from input
        due to insertion (dittography) or deletion (haplography).
    """
    if not tokens:
        return []

    rng = random.Random(seed)
    result = [deepcopy(t) for t in tokens]
    n_errors = max(1, round(len(result) * error_rate))

    for _ in range(n_errors):
        if len(result) < 2:
            break

        error_type = rng.random()
        idx = rng.randrange(len(result) - 1)

        if error_type < 0.40:
            # Haplography: delete current token if it's near-identical to neighbor
            t_curr = result[idx]["normalized"]
            t_next = result[idx + 1]["normalized"]
            if t_curr and t_next and (t_curr == t_next or _edit_dist_1(t_curr, t_next)):
                result.pop(idx)

        elif error_type < 0.75:
            # Dittography: duplicate token at idx
            dup = deepcopy(result[idx])
            result.insert(idx + 1, dup)

        else:
            # Transposition: swap tokens at idx and idx+1
            result[idx], result[idx + 1] = result[idx + 1], result[idx]

    return result


def _edit_dist_1(a: str, b: str) -> bool:
    """Return True if strings a and b have edit distance ≤ 1.

    Used to detect haplography candidates (near-identical adjacent tokens).
    Fast O(n) check without computing the full Levenshtein matrix.

    Args:
        a: First string.
        b: Second string.

    Returns:
        True if strings differ by at most 1 character operation.
    """
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True

    # Try deletion of one character from longer string
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    for i in range(len(longer)):
        if longer[:i] + longer[i + 1 :] == shorter:
            return True
    return False
