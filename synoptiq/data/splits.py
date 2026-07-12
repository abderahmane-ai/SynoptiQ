"""Stratified train/val/test splits for SynoptiQ.

Splits pericopes into train/val/test sets with these invariants:
  1. ATOMIC: A pericope is never split across partitions.
  2. STRATIFIED: Each split contains proportional representation
     of each tradition × genre combination.
  3. NO LEAKAGE: If a pericope appears in two books (e.g., triple tradition),
     ALL of its book versions go into the same split — never Matthew in train
     and Mark in test for the same pericope.

Stratification key: (tradition, genre) — e.g., ("triple", "narrative").
We use sklearn's StratifiedGroupKFold for this purpose.

Pericope-atomic splitting is critical because a model's
test set must contain pericopes it has never seen any version of.
If Matt 14:13-21 (Feeding of 5K) is in training, then the "Mark 6:30-44"
version of the same pericope cannot be in the test set — they are the same
parallel passage, so the model has effectively seen the test example.
"""

from __future__ import annotations

from collections import Counter
import random

from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import PericopeAlignment, SplitResult

_LOG = get_logger(__name__)


def _get_stratification_key(alignment: PericopeAlignment) -> str:
    """Compute the stratification key for a pericope.

    Args:
        alignment: A PericopeAlignment dict.

    Returns:
        String like "triple_narrative" or "double_discourse".
    """
    return f"{alignment['tradition']}_{alignment['genre']}"


def split_pericopes(
    alignments: list[PericopeAlignment],
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    random_seed: int = 42,
    min_test_per_stratum: int = 1,
) -> SplitResult:
    """Produce stratified pericope-atomic train/val/test splits.

    Uses a deterministic stratified split that:
    1. Groups pericopes by (tradition, genre) stratum
    2. Within each stratum, shuffles deterministically by seed
    3. Allocates the first ``train_frac`` to train, next ``val_frac`` to val,
       remainder to test

    For small strata (< 3 pericopes), all go to train to avoid empty
    val/test strata.

    Args:
        alignments: List of PericopeAlignment dicts (one per pericope).
        train_frac: Fraction of pericopes for training.
        val_frac: Fraction for validation.
        test_frac: Fraction for test.
        random_seed: Random seed for reproducibility.
        min_test_per_stratum: Minimum test samples per stratum.

    Returns:
        SplitResult with train_ids, val_ids, test_ids.

    Raises:
        ValueError: If fractions don't sum to 1.0.
    """
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        msg = f"Fractions must sum to 1.0, got {total_frac}"
        raise ValueError(msg)

    if not alignments:
        return SplitResult(train_ids=[], val_ids=[], test_ids=[])

    # Group by stratification key
    strata: dict[str, list[PericopeAlignment]] = {}
    for alignment in alignments:
        key = _get_stratification_key(alignment)
        if key not in strata:
            strata[key] = []
        strata[key].append(alignment)

    _LOG.info(
        "split strata",
        extra={"strata": {k: len(v) for k, v in strata.items()}},
    )

    rng = random.Random(random_seed)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for stratum_key, stratum_alignments in sorted(strata.items()):
        n = len(stratum_alignments)

        # Deterministic shuffle within stratum
        perm = list(range(n))
        rng.shuffle(perm)
        shuffled = [stratum_alignments[i] for i in perm]

        if n < 3:
            # Too small to split meaningfully — all to train
            train_ids.extend(a["pericope_id"] for a in shuffled)
            _LOG.debug(
                "stratum too small — all to train",
                extra={"stratum": stratum_key, "n": n},
            )
            continue

        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        n_test = n - n_train - n_val

        if n_test < min_test_per_stratum:
            # Steal from val to ensure at least one test sample
            n_val = max(0, n_val - (min_test_per_stratum - n_test))
            n_test = n - n_train - n_val

        train_ids.extend(a["pericope_id"] for a in shuffled[:n_train])
        val_ids.extend(a["pericope_id"] for a in shuffled[n_train : n_train + n_val])
        test_ids.extend(a["pericope_id"] for a in shuffled[n_train + n_val :])

    result = SplitResult(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    # Validation
    all_ids = set(train_ids) | set(val_ids) | set(test_ids)
    pericope_ids = {a["pericope_id"] for a in alignments}
    missing = pericope_ids - all_ids
    overlap_tv = set(train_ids) & set(val_ids)
    overlap_tt = set(train_ids) & set(test_ids)
    overlap_vt = set(val_ids) & set(test_ids)

    if missing:
        _LOG.warning("pericopes missing from split", extra={"missing": sorted(missing)})
    if overlap_tv or overlap_tt or overlap_vt:
        _LOG.error(
            "split overlap detected",
            extra={
                "train_val": sorted(overlap_tv),
                "train_test": sorted(overlap_tt),
                "val_test": sorted(overlap_vt),
            },
        )

    _LOG.info(
        "split complete",
        extra={
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "total": len(all_ids),
        },
    )
    return result


def split_stats(
    alignments: list[PericopeAlignment],
    split: SplitResult,
) -> dict[str, dict[str, int]]:
    """Compute stratification statistics for a split result.

    Useful for verifying that tradition/genre distribution is
    balanced across splits.

    Args:
        alignments: Original list of PericopeAlignment dicts.
        split: Output of split_pericopes().

    Returns:
        Dict with keys "train", "val", "test", each mapping to
        a Counter of stratum_key → count.
    """
    pid_to_alignment = {a["pericope_id"]: a for a in alignments}

    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for split_name, pid_list in [
        ("train", split["train_ids"]),
        ("val", split["val_ids"]),
        ("test", split["test_ids"]),
    ]:
        counter: Counter[str] = Counter()
        for pid in pid_list:
            if pid in pid_to_alignment:
                key = _get_stratification_key(pid_to_alignment[pid])
                counter[key] += 1
        stats[split_name] = dict(counter)

    return stats
