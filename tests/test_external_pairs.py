"""Tests for the external known-direction pair loader."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from synoptiq.data.external_pairs import load_external_pairs


class _StubTokenizer:
    """Minimal stand-in for the GreTa tokenizer (no network, deterministic).

    Maps each whitespace token to a stable id via hashing, pads/truncates to
    max_length, and returns the HF-style dict of batched tensors.
    """

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        truncation: bool,      # noqa: ARG002
        padding: str,          # noqa: ARG002
        return_tensors: str,   # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        ids = [ (abs(hash(w)) % 1000) + 1 for w in text.split() ][:max_length]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            mask.append(0)
        return {
            "input_ids": torch.tensor(ids).unsqueeze(0),
            "attention_mask": torch.tensor(mask).unsqueeze(0),
        }


def _write_pairs(tmp_path: Path) -> Path:
    data = {
        "description": "test",
        "pairs": [
            {
                "id": "p1", "group": "g1", "direction": "A_to_B",
                "book_a": "Jude", "book_b": "2Peter",
                "text_a": "αλφα βητα γαμμα", "text_b": "δελτα epsilon",
            },
            {
                "id": "p2", "group": "g2", "direction": "independent",
                "book_a": "X", "book_b": "Y",
                "text_a": "one two", "text_b": "three four five",
            },
        ],
    }
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_pairs_with_swap_augmentation(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path)
    samples = load_external_pairs(path, _StubTokenizer(), max_length=8)
    # 2 pairs x 2 (original + swap) = 4 samples
    assert len(samples) == 4
    for s in samples:
        assert s["input_ids_a"].shape == (8,)
        assert s["attention_mask_a"].shape == (8,)
        assert s["direction_label"].dtype == torch.long


def test_swap_flips_label_and_swaps_text(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path)
    samples = load_external_pairs(path, _StubTokenizer(), max_length=8)
    original = next(s for s in samples if s["id"] == "p1")
    swap = next(s for s in samples if s["id"] == "p1_swap")

    assert original["direction_label"].item() == 0  # A_to_B
    assert swap["direction_label"].item() == 1       # B_to_A
    # swapped a/b are the original b/a
    assert torch.equal(swap["input_ids_a"], original["input_ids_b"])
    assert torch.equal(swap["input_ids_b"], original["input_ids_a"])
    # swap keeps the source group for bootstrap
    assert swap["group"] == original["group"] == "g1"


def test_independent_label_is_invariant_under_swap(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path)
    samples = load_external_pairs(path, _StubTokenizer(), max_length=8)
    ind = [s for s in samples if s["id"].startswith("p2")]
    assert all(s["direction_label"].item() == 2 for s in ind)


def test_augment_swap_disabled(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path)
    samples = load_external_pairs(path, _StubTokenizer(), max_length=8, augment_swap=False)
    assert len(samples) == 2
    assert all(not s["id"].endswith("_swap") for s in samples)


def test_real_external_file_is_loadable_if_present() -> None:
    # If the committed dataset exists, it must parse and have balanced swap pairs.
    real = Path("data/external/known_direction_pairs.json")
    if not real.exists():
        return
    samples = load_external_pairs(real, _StubTokenizer(), max_length=128)
    assert len(samples) > 0
    assert len(samples) % 2 == 0  # swap augmentation balances the set
    labels = {s["direction_label"].item() for s in samples}
    assert labels <= {0, 1, 2}
