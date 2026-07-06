"""Loader for the external known-direction evaluation pairs.

These pairs (built by scripts/build_external_pairs.py) have a scholarly-consensus
copying direction and contain no synoptic author, so they test whether a direction
probe measures *direction* rather than *authorship*. Unlike DirectionDataset, this
loader takes raw Greek text pairs — it does not need a Corpus, alignments, or the
Aland pericope table — so it can serve any known-direction corpus (Jude/2 Peter now,
LXX Samuel-Kings/Chronicles later) behind the same tensor interface.

The output dict matches DirectionScorer.forward / the NLL probe: input_ids_a,
attention_mask_a, input_ids_b, attention_mask_b, direction_label, plus a `group`
key for pericope/block-level bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import AutoTokenizer

_DIRECTION_TO_IDX = {"A_to_B": 0, "B_to_A": 1, "independent": 2}


def load_external_pairs(
    path: str | Path,
    tokenizer: AutoTokenizer,
    *,
    max_length: int = 512,
    augment_swap: bool = True,
) -> list[dict[str, Any]]:
    """Load known-direction pairs and tokenize them.

    Args:
        path: Path to a known-direction JSON (see build_external_pairs.py).
        tokenizer: GreTa tokenizer (with pad token added).
        max_length: Truncation/padding length.
        augment_swap: If True, also emit the A<->B swapped copy with the flipped
            label, so both orderings are present and slot position carries no signal.
            The swapped copy keeps the source `group` for bootstrap grouping.

    Returns:
        List of sample dicts with tokenized tensors, direction_label, and metadata.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []

    def _encode(text: str) -> dict[str, torch.Tensor]:
        enc = tokenizer(
            text, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return enc

    for pair in data["pairs"]:
        direction = pair["direction"]
        enc_a = _encode(pair["text_a"])
        enc_b = _encode(pair["text_b"])
        samples.append({
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "direction_label": torch.tensor(_DIRECTION_TO_IDX[direction], dtype=torch.long),
            "group": pair.get("group", pair["id"]),
            "id": pair["id"],
        })

        if augment_swap:
            if direction == "A_to_B":
                swapped_dir = "B_to_A"
            elif direction == "B_to_A":
                swapped_dir = "A_to_B"
            else:
                swapped_dir = "independent"
            samples.append({
                "input_ids_a": enc_b["input_ids"].squeeze(0),
                "attention_mask_a": enc_b["attention_mask"].squeeze(0),
                "input_ids_b": enc_a["input_ids"].squeeze(0),
                "attention_mask_b": enc_a["attention_mask"].squeeze(0),
                "direction_label": torch.tensor(
                    _DIRECTION_TO_IDX[swapped_dir], dtype=torch.long,
                ),
                "group": pair.get("group", pair["id"]),
                "id": pair["id"] + "_swap",
            })

    return samples
