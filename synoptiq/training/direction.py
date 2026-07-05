"""Direction scorer training — dataset, data loader, and training loop.

Provides a DirectionDataset that wraps Corpus.direction_pairs() into
batched tensors, and a DirectionTrainer with AMP, checkpointing, and
GRL annealing.
"""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import signal
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from synoptiq.data.augmentation import add_scribal_noise
from synoptiq.data.corpus import Corpus
from synoptiq.models.direction import DirectionScorer
from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import Book

if TYPE_CHECKING:
    from transformers import AutoTokenizer

_LOG = get_logger(__name__)

# Label mapping
_DIRECTION_TO_IDX = {"A_to_B": 0, "B_to_A": 1, "independent": 2}
_BOOK_TO_AUTHOR_IDX = {"Matthew": 0, "Mark": 1, "Luke": 2}


# ── Training config ───────────────────────────────────────────────────────────


@dataclass
class DirectionTrainingConfig:
    """Mutable training config for the direction scorer."""

    batch_size: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 5_000
    warmup_steps: int = 500
    val_steps: int = 250
    save_steps: int = 1_000
    max_length: int = 512
    use_amp: bool = True
    grl_warmup_steps: int = 1_000
    grl_lambda_max: float = 1.0
    output_dir: Path = field(default_factory=lambda: Path("outputs/direction"))
    min_aligned_tokens: int = 5
    use_sliding_windows: bool = False  # TODO: fix for aligned-pair windows
    window_size: int = 32
    window_stride: int = 16
    use_scribal_noise: bool = True


# ── Dataset ───────────────────────────────────────────────────────────────────


class DirectionDataset(Dataset):
    """Map-style dataset wrapping Corpus.direction_pairs().

    Each sample is a pair of aligned passages with a direction label.
    Uses the existing splits.json for train/val/test separation.

    Labels for triple tradition (known direction):
      - (Mark, Matthew)  → A_to_B
      - (Mark, Luke)     → A_to_B
      - (Matthew, Luke)  → independent (under 2SH)

    Labels for double tradition:
      - (Matthew, Luke)  → independent
    """

    def __init__(
        self,
        corpus: Corpus,
        tokenizer: AutoTokenizer,
        *,
        split: str = "train",
        max_length: int = 512,
        min_aligned_tokens: int = 5,
        use_sliding_windows: bool = False,
        window_size: int = 32,
        window_stride: int = 16,
        use_scribal_noise: bool = True,
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_aligned_tokens = min_aligned_tokens
        self.use_sliding_windows = use_sliding_windows
        self.window_size = window_size
        self.window_stride = window_stride
        self.use_scribal_noise = use_scribal_noise

        self.samples: list[dict] = []
        self._build_samples(split)

        _LOG.info(
            "direction dataset built",
            extra={"split": split, "n_samples": len(self.samples)},
        )

    def _build_samples(self, split: str) -> None:
        """Build samples from direction_pairs, applying augmentation."""
        for book_a, tokens_a, book_b, tokens_b, alignment in self.corpus.direction_pairs(
            split=split,
        ):
            # Filter to aligned token pairs (both non-None)
            aligned = [(i, j) for i, j in alignment if i is not None and j is not None]
            if len(aligned) < self.min_aligned_tokens:
                continue

            # Determine direction label
            direction = self._get_direction_label(book_a, book_b)

            # TODO: sliding-window augmentation for aligned pairs
            if self.use_sliding_windows and len(aligned) > self.window_size:
                pass  # NYI — existing sliding_windows works on single passages
            else:
                self.samples.append({
                    "tokens_a": tokens_a,
                    "tokens_b": tokens_b,
                    "book_a": book_a,
                    "book_b": book_b,
                    "direction": direction,
                })

        # Augment: swap A↔B for inverse labels
        swapped = []
        for s in self.samples:
            if s["direction"] == "A_to_B":
                new_dir = "B_to_A"
            elif s["direction"] == "B_to_A":
                new_dir = "A_to_B"
            else:
                new_dir = "independent"
            swapped.append({
                "tokens_a": s["tokens_b"],
                "tokens_b": s["tokens_a"],
                "book_a": s["book_b"],
                "book_b": s["book_a"],
                "direction": new_dir,
            })
        self.samples.extend(swapped)

        _LOG.info(f"built {len(self.samples)} samples (with swap augmentation)")

    @staticmethod
    def _get_direction_label(book_a: Book, book_b: Book) -> str:
        """Get known direction label for a book pair.

        Under Markan priority (2SH):
          - Mark → Matthew, Mark → Luke (Mark is the source)
          - Matthew ↔ Luke are independent (both used Mark + Q)
        """
        pair = (book_a, book_b)
        if pair == ("Mark", "Matthew") or pair == ("Mark", "Luke"):
            return "A_to_B"  # source=A=Mark, target=B
        if pair == ("Matthew", "Mark") or pair == ("Luke", "Mark"):
            return "B_to_A"
        # Matthew ↔ Luke or any other pair
        return "independent"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        tokens_a = sample["tokens_a"]
        tokens_b = sample["tokens_b"]

        # Apply scribal noise if enabled
        if self.use_scribal_noise and torch.rand(1).item() < 0.3:
            tokens_a = add_scribal_noise(tokens_a, error_rate=0.01)

        # Build text from token records and pad
        text_a = " ".join(t["text"] for t in tokens_a)
        text_b = " ".join(t["text"] for t in tokens_b)

        encoded_a = self.tokenizer(
            text_a, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        encoded_b = self.tokenizer(
            text_b, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        label = _DIRECTION_TO_IDX[sample["direction"]]
        author_a = _BOOK_TO_AUTHOR_IDX.get(sample["book_a"], 0)
        author_b = _BOOK_TO_AUTHOR_IDX.get(sample["book_b"], 0)

        return {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "direction_label": torch.tensor(label, dtype=torch.long),
            "author_label_a": torch.tensor(author_a, dtype=torch.long),
            "author_label_b": torch.tensor(author_b, dtype=torch.long),
        }


# ── Trainer ───────────────────────────────────────────────────────────────────


class DirectionTrainer:
    """Training loop for the DirectionScorer.

    Features: AMP (FP16), GRL annealing, crash-safe checkpointing with
    SIGTERM handler, cosine LR schedule, periodic validation.
    """

    def __init__(
        self,
        scorer: DirectionScorer,
        train_dataset: DirectionDataset,
        val_dataset: DirectionDataset | None,
        config: DirectionTrainingConfig,
        device: str = "cuda",
    ):
        self.scorer = scorer
        self.config = config
        self.device = device

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
        ) if val_dataset else None

        self.optimizer = AdamW(
            scorer.cross_attn.parameters(),
            lr=config.learning_rate,
        )
        # Also optimize classifier and discriminator
        self.optimizer.add_param_group({
            "params": scorer.classifier.parameters(),
            "lr": config.learning_rate,
        })
        self.optimizer.add_param_group({
            "params": scorer.author_disc.parameters(),
            "lr": config.learning_rate,
        })

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.max_steps)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if config.use_amp and device.startswith("cuda")
            else None
        )

        self.direction_loss_fn = nn.CrossEntropyLoss()
        self.author_loss_fn = nn.CrossEntropyLoss()

        self.global_step = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_accuracy": [],
        }

        self._setup_signal_handler()

    def _setup_signal_handler(self) -> None:
        """Handle SIGTERM for graceful shutdown on spot preemption."""
        self._interrupted = False

        def _handler(signum: int, frame: object) -> None:
            _LOG.warning("received SIGTERM — saving checkpoint")
            self._interrupted = True

        signal.signal(signal.SIGTERM, _handler)

    def _grl_lambda(self) -> float:
        """Linear warmup schedule for GRL gradient scale."""
        if self.global_step >= self.config.grl_warmup_steps:
            return self.config.grl_lambda_max
        return self.config.grl_lambda_max * (self.global_step / self.config.grl_warmup_steps)

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop."""
        scorer = self.scorer.to(self.device)
        scorer.train()
        config = self.config

        train_iter = iter(self.train_loader)
        best_val_acc = 0.0

        _LOG.info(f"starting direction scorer training: {config.max_steps} steps")

        while self.global_step < config.max_steps:
            if self._interrupted:
                self._save_checkpoint()
                _LOG.info("checkpoint saved, exiting")
                return self.history

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            with torch.amp.autocast("cuda", enabled=config.use_amp and self.device.startswith("cuda")):
                output = scorer(
                    input_ids_a=batch["input_ids_a"],
                    attention_mask_a=batch["attention_mask_a"],
                    input_ids_b=batch["input_ids_b"],
                    attention_mask_b=batch["attention_mask_b"],
                )

                # Direction loss
                dir_loss = self.direction_loss_fn(
                    output["direction_logits"], batch["direction_label"],
                )

                # Adversarial author loss
                author_loss_a = self.author_loss_fn(
                    output["author_logits_a"], batch["author_label_a"],
                )
                author_loss_b = self.author_loss_fn(
                    output["author_logits_b"], batch["author_label_b"],
                )
                author_loss = (author_loss_a + author_loss_b) / 2

                # Total loss
                loss = dir_loss + 0.1 * author_loss

            # Backward with AMP
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            # Update GRL lambda
            scorer.set_grl_lambda(self._grl_lambda())

            self.history["train_loss"].append(loss.item())

            # Logging
            if self.global_step % 50 == 0:
                _LOG.info(
                    f"step {self.global_step}/{config.max_steps}",
                    extra={
                        "loss": round(loss.item(), 4),
                        "dir_loss": round(dir_loss.item(), 4),
                        "author_loss": round(author_loss.item(), 4),
                        "grl_lambda": round(scorer._grl_lambda, 3),
                        "lr": round(self.scheduler.get_last_lr()[0], 6),
                    },
                )

            # Validation
            if self.global_step % config.val_steps == 0 and self.val_loader is not None:
                val_metrics = self._validate()
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                _LOG.info(f"validation @ step {self.global_step}", extra=val_metrics)

                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    self._save_checkpoint(suffix="best")

            # Checkpoint
            if self.global_step % config.save_steps == 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint(suffix="final")
        _LOG.info("training complete", extra={"best_val_acc": round(best_val_acc, 4)})
        return self.history

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Evaluate on validation set."""
        self.scorer.eval()
        total_loss = 0.0
        correct = 0
        n_samples = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.scorer(
                input_ids_a=batch["input_ids_a"],
                attention_mask_a=batch["attention_mask_a"],
                input_ids_b=batch["input_ids_b"],
                attention_mask_b=batch["attention_mask_b"],
            )

            dir_loss = self.direction_loss_fn(
                output["direction_logits"], batch["direction_label"],
            )
            total_loss += dir_loss.item() * len(batch["direction_label"])

            preds = output["direction_logits"].argmax(dim=1)
            correct += (preds == batch["direction_label"]).sum().item()
            n_samples += len(batch["direction_label"])

        self.scorer.train()
        return {
            "loss": round(total_loss / max(n_samples, 1), 4),
            "accuracy": round(correct / max(n_samples, 1), 6),
        }

    def _save_checkpoint(self, suffix: str | None = None) -> None:
        """Save model + optimizer + scheduler state."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        step_tag = suffix or f"step-{self.global_step}"
        ckpt_dir = output_dir / step_tag
        ckpt_dir.mkdir(exist_ok=True)

        # Mark incomplete until fully written
        incomplete_marker = ckpt_dir / ".incomplete"
        incomplete_marker.touch()

        # Save model weights
        torch.save(self.scorer.state_dict(), ckpt_dir / "model.pt")

        # Save training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "history": self.history,
        }, ckpt_dir / "training_state.pt")

        # Remove marker — checkpoint is complete
        incomplete_marker.unlink()
        _LOG.info(f"checkpoint saved: {ckpt_dir}")


# ── Collate function ──────────────────────────────────────────────────────────


def collate_direction_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a list of samples into a batch."""
    return {
        "input_ids_a": torch.stack([s["input_ids_a"] for s in batch]),
        "attention_mask_a": torch.stack([s["attention_mask_a"] for s in batch]),
        "input_ids_b": torch.stack([s["input_ids_b"] for s in batch]),
        "attention_mask_b": torch.stack([s["attention_mask_b"] for s in batch]),
        "direction_label": torch.stack([s["direction_label"] for s in batch]),
        "author_label_a": torch.stack([s["author_label_a"] for s in batch]),
        "author_label_b": torch.stack([s["author_label_b"] for s in batch]),
    }
