"""Multi-task LoRA fine-tuning for KoineFormer.

Trains task-specific heads (POS, dependency parsing, lemmatisation,
pericope classification) on the frozen KoineFormer encoder with
task-specific LoRA adapters.

Usage:
    from synoptiq.training.multitask import MultiTaskTrainer
    trainer = MultiTaskTrainer(model, corpus, config)
    metrics = trainer.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# ── Dataset ─────────────────────────────────────────────────────────────────


class POSDataset(Dataset):
    """Token-level POS tagging dataset backed by the SynoptiQ Corpus.

    Each sample is a (token_sequence, POS_labels) pair for one verse.
    """

    def __init__(
        self,
        corpus: Any,
        tokenizer: Any,
        *,
        split: str = "train",
        max_length: int = 128,
    ) -> None:
        self._samples: list[dict[str, torch.Tensor]] = []
        self._build(corpus, tokenizer, split, max_length)

    def _build(
        self, corpus: Any, tokenizer: Any, split: str, max_length: int
    ) -> None:
        """Pre-tokenize all verses into (input_ids, pos_labels) pairs."""
        pos_to_idx: dict[str, int] = {}
        for token in corpus.get_tokens(split=split):
            p = token.get("pos", "")
            if p and p not in pos_to_idx:
                pos_to_idx[p] = len(pos_to_idx)

        _LOG.info("POS dataset built", extra={"n_pos_tags": len(pos_to_idx)})

        # Group by verse and tokenize
        for book in ("Matthew", "Mark", "Luke"):
            tokens = corpus.get_tokens(book=book, split=split)
            # Group into verses
            verses: dict[tuple[int, int], list[Any]] = {}
            for t in tokens:
                key = (int(t["chapter"]), int(t["verse"]))  # type: ignore[arg-type]
                verses.setdefault(key, []).append(t)

            for _vref, verse_tokens in verses.items():
                text = " ".join(str(t["text"]) for t in verse_tokens)
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                # Build POS label tensor
                pos_labels = torch.full((max_length,), -100, dtype=torch.long)
                for i, t in enumerate(verse_tokens):
                    p = t.get("pos", "")
                    if p and p in pos_to_idx and i < max_length:
                        pos_labels[i] = pos_to_idx[p]

                self._samples.append({
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                    "pos_labels": pos_labels,
                })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._samples[idx]


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class MultiTaskTrainingConfig:
    """Hyperparameters for multi-task LoRA fine-tuning.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        max_length: Maximum token sequence length.
        output_dir: Directory for multi-task head checkpoints.
    """

    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_length: int = 128
    output_dir: Path = field(default_factory=lambda: Path("outputs/multitask"))


# ── Trainer ─────────────────────────────────────────────────────────────────


class MultiTaskTrainer:
    """Trains task-specific heads on the frozen KoineFormer encoder.

    Args:
        model: A KoineFormer instance with frozen encoder.
        corpus: SynoptiQ Corpus for training tokens.
        tokenizer: GreTa tokenizer.
        config: MultiTaskTrainingConfig.
        device: Torch device.
    """

    def __init__(
        self,
        model: Any,
        corpus: Any,
        tokenizer: Any,
        config: MultiTaskTrainingConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._corpus = corpus
        self._tokenizer = tokenizer
        self._config = config or MultiTaskTrainingConfig()
        self._device = device

    def run(self) -> dict[str, dict[str, float]]:
        """Run multi-task training (POS tagging only initially).

        Returns:
            Dict of task_name → {"loss": final_loss, "accuracy": final_accuracy}.
        """
        from synoptiq.models.encoder import MultiTaskConfig, MultiTaskEncoder

        config = self._config

        # Build POS dataset
        train_ds = POSDataset(self._corpus, self._tokenizer, split="train",
                              max_length=config.max_length)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        # Build multi-task encoder
        mt_config = MultiTaskConfig()
        encoder = MultiTaskEncoder(self._model.model.base_model.encoder, mt_config)
        encoder.to(self._device)
        encoder.train()

        optimizer = AdamW(encoder.parameters(), lr=config.learning_rate)
        pos_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        _LOG.info("multi-task training starting", extra={"epochs": config.epochs})

        history: dict[str, list[float]] = {"loss": [], "pos_accuracy": []}

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                pos_labels = batch["pos_labels"].to(self._device)

                optimizer.zero_grad()
                outputs = encoder(input_ids, attention_mask, task="pos")
                pos_logits = outputs["pos_logits"]  # [B, S, n_classes]

                loss = pos_criterion(pos_logits.permute(0, 2, 1), pos_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = pos_logits.argmax(dim=-1)
                mask = pos_labels != -100
                correct += (preds[mask] == pos_labels[mask]).sum().item()
                total += mask.sum().item()

            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / max(total, 1)
            history["loss"].append(avg_loss)
            history["pos_accuracy"].append(accuracy)

            _LOG.info(
                f"epoch {epoch + 1}/{config.epochs}",
                extra={"loss": f"{avg_loss:.4f}", "pos_acc": f"{accuracy:.2%}"},
            )

        # Save heads
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), output_dir / "multitask_heads.pt")
        _LOG.info("multi-task training complete", extra={"output": str(output_dir)})

        return {
            "pos": {
                "final_loss": history["loss"][-1],
                "final_accuracy": history["pos_accuracy"][-1],
            }
        }
