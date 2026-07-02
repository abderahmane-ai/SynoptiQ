"""Tests for synoptiq/data/splits.py — stratified pericope-level splits."""

from __future__ import annotations

from typing import Any

from synoptiq.data.splits import split_pericopes


def _make_alignment(pericope_id: str, tradition: str, books: list[str]) -> dict[str, Any]:
    return {
        "pericope_id": pericope_id,
        "tradition": tradition,
        "genre": "narrative",
        "books": books,
        "tokens": {},
        "alignment": {},
    }


class TestSplitPericopes:
    def test_split_sum_to_total(self) -> None:
        """Train + val + test should cover all pericopes."""
        alignments = [
            _make_alignment(f"p{i:03d}", "triple", ["Matthew", "Mark", "Luke"]) for i in range(50)
        ]
        result = split_pericopes(alignments, train_frac=0.60, val_frac=0.20, test_frac=0.20)
        all_split = set(result["train_ids"]) | set(result["val_ids"]) | set(result["test_ids"])
        assert len(all_split) == 50

    def test_split_no_overlap(self) -> None:
        """No pericope should appear in more than one split."""
        alignments = [
            _make_alignment(f"p{i:03d}", "triple", ["Matthew", "Mark", "Luke"]) for i in range(50)
        ]
        result = split_pericopes(alignments, train_frac=0.60, val_frac=0.20, test_frac=0.20)
        train_s = set(result["train_ids"])
        val_s = set(result["val_ids"])
        test_s = set(result["test_ids"])
        assert train_s.isdisjoint(val_s)
        assert train_s.isdisjoint(test_s)
        assert val_s.isdisjoint(test_s)

    def test_each_stratum_present(self) -> None:
        """Mixed traditions should all appear in each split."""
        alignments = []
        for i in range(20):
            alignments.append(_make_alignment(f"t{i:03d}", "triple", ["Matthew", "Mark", "Luke"]))
        for i in range(20):
            alignments.append(_make_alignment(f"d{i:03d}", "double", ["Matthew", "Luke"]))
        for i in range(10):
            alignments.append(_make_alignment(f"m{i:03d}", "mark_unique", ["Mark"]))
        result = split_pericopes(alignments, train_frac=0.60, val_frac=0.20, test_frac=0.20)
        # Each split should have at least one pericope from each tradition
        for split_ids in [result["train_ids"], result["val_ids"], result["test_ids"]]:
            traditions = {a["tradition"] for a in alignments if a["pericope_id"] in split_ids}
            assert "triple" in traditions
            assert "double" in traditions
            assert "mark_unique" in traditions

    def test_small_dataset(self) -> None:
        """Small datasets: all may go to train if insufficient per stratum."""
        alignments = [
            _make_alignment("p001", "triple", ["Matthew", "Mark", "Luke"]),
            _make_alignment("p002", "double", ["Matthew", "Luke"]),
            _make_alignment("p003", "mark_unique", ["Mark"]),
        ]
        result = split_pericopes(alignments)
        # With 3 pericopes x 1 per stratum, val/test may be empty
        assert len(result["train_ids"]) >= 1
        # Don't require val/test to be non-empty for tiny datasets

    def test_deterministic_with_fixed_seed(self) -> None:
        """Same random_seed → same split."""
        alignments = [
            _make_alignment(f"p{i:03d}", "triple", ["Matthew", "Mark", "Luke"]) for i in range(30)
        ]
        r1 = split_pericopes(alignments, random_seed=42)
        r2 = split_pericopes(alignments, random_seed=42)
        assert r1["train_ids"] == r2["train_ids"]
        assert r1["val_ids"] == r2["val_ids"]
        assert r1["test_ids"] == r2["test_ids"]

    def test_different_seed_different_split(self) -> None:
        """Different seeds produce different assignments."""
        alignments = [
            _make_alignment(f"p{i:03d}", "triple", ["Matthew", "Mark", "Luke"]) for i in range(30)
        ]
        r1 = split_pericopes(alignments, random_seed=42)
        r2 = split_pericopes(alignments, random_seed=123)
        assert r1["train_ids"] != r2["train_ids"] or r1["val_ids"] != r2["val_ids"]
