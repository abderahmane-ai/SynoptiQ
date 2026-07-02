"""Evaluation benchmarks for KoineFormer.

Compares KoineFormer against baseline Greek NLP models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results for a single model on a single task."""
    model_name: str
    task: str
    metric_name: str
    value: float
    n_samples: int


def evaluate_pos_tagging(
    model: Any,
    corpus: Any,
    tokenizer: Any,
    *,
    split: str = "test",
    device: str = "cpu",
) -> BenchmarkResult:
    """Evaluate POS tagging accuracy on the held-out test split."""
    from synoptiq.models.encoder import MultiTaskEncoder

    encoder = MultiTaskEncoder(model.model.base_model.encoder)
    encoder.to(device)
    encoder.eval()

    pos_to_idx: dict[str, int] = {}
    for token in corpus.get_tokens(split="train"):
        p = token.get("pos", "")
        if p and p not in pos_to_idx:
            pos_to_idx[p] = len(pos_to_idx)

    idx_to_pos = {v: k for k, v in pos_to_idx.items()}
    correct = 0
    total = 0

    for book in ("Matthew", "Mark", "Luke"):
        tokens = corpus.get_tokens(book=book, split=split)
        verses: dict[tuple[int, int], list[Any]] = {}
        for t in tokens:
            key = (int(t["chapter"]), int(t["verse"]))
            verses.setdefault(key, []).append(t)

        for _vref, verse_tokens in verses.items():
            text = " ".join(str(t["text"]) for t in verse_tokens)
            encoded = tokenizer(
                text, max_length=128, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = encoder(input_ids, attention_mask, task="pos")
            preds = outputs["pos_logits"].argmax(dim=-1)[0]

            for i, t in enumerate(verse_tokens):
                if i >= 128:
                    break
                gold_pos = t.get("pos", "")
                if gold_pos and gold_pos in pos_to_idx:
                    total += 1
                    if idx_to_pos.get(preds[i].item()) == gold_pos:
                        correct += 1

    accuracy = correct / max(total, 1)
    return BenchmarkResult(
        model_name=model.model_id,
        task="POS tagging",
        metric_name="accuracy",
        value=accuracy,
        n_samples=total,
    )
