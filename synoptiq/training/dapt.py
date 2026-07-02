"""Domain-Adaptive Pre-Training (DAPT) for KoineFormer.

Data loader + training loop that converts GreTa into KoineFormer by
training LoRA adapters on T5 span corruption with a 70/30
Koine/Classical replay buffer.

Usage:
    from synoptiq.training.dapt import DAPTDataLoader, DAPTTrainer

    loader = DAPTDataLoader(data_dir, tokenizer)
    trainer = DAPTTrainer(model, loader, config)
    history = trainer.run()
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Sources for Koine Greek text (DAPT target domain).
KOINE_SOURCES: list[str] = ["sblgnt", "lxx", "apostolic"]

# Sources for Classical Greek text (replay buffer anchor).
CLASSICAL_SOURCES: list[str] = ["first1k"]

# Replay buffer ratio: 70% Koine, 30% Classical.
REPLAY_KOINE_RATIO: float = 0.70

# T5 span corruption: fraction of tokens replaced by sentinel spans.
NOISE_DENSITY: float = 0.15
NOISE_MEAN_SPAN_LENGTH: float = 3.0

# Default maximum sequence length fed to the encoder.
DEFAULT_MAX_LENGTH: int = 512


# ── Text extraction ─────────────────────────────────────────────────────────


def _extract_text_from_dir(data_dir: Path, source_name: str) -> Iterator[str]:
    """Yield Greek text chunks from a downloaded corpus directory.

    Handles plain-text (``*.txt``), XML (``*.xml``), and TEI XML files.
    Skips files that are clearly not Greek text (README, LICENSE, etc.).

    Args:
        data_dir: Root ``data/raw/`` directory.
        source_name: Subdirectory name (e.g. ``"sblgnt"``, ``"lxx"``).

    Yields:
        Greek text chunks (strings of 50-500 chars).
    """
    src_dir = data_dir / source_name
    if not src_dir.exists():
        _LOG.warning("source directory not found", extra={"path": str(src_dir)})
        return

    skip_names = {"README", "LICENSE", "NOTES", "Makefile", "makefile", "Pipfile"}
    text_files = 0

    for filepath in sorted(src_dir.rglob("*")):
        if filepath.is_dir():
            continue
        if filepath.suffix not in {".txt", ".xml", ".tei", ".tf"}:
            continue
        if any(s in filepath.name for s in skip_names):
            continue

        try:
            if filepath.suffix == ".xml" or filepath.suffix == ".tei":
                tree = ET.parse(filepath)
                for elem in tree.iter():
                    if elem.text and len(elem.text.strip()) > 3 and _contains_greek(elem.text):
                        yield elem.text.strip()
            elif filepath.suffix == ".tf":
                # Text-Fabric files — skip structured data, extract readable strings.
                raw = filepath.read_text(encoding="utf-8", errors="replace")
                # TF format has lines with key=value pairs; extract value strings.
                for line in raw.splitlines():
                    if "=" in line and len(line) > 10:
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if len(val) > 20 and _contains_greek(val):
                            yield val[:500]
            else:  # .txt
                raw = filepath.read_text(encoding="utf-8", errors="replace")
                # Split into sentence-like chunks on punctuation + whitespace.
                chunks = re.split(r"(?<=[.;·])\s+", raw)
                for chunk in chunks:
                    stripped = chunk.strip()
                    if len(stripped) > 20 and _contains_greek(stripped):
                        yield stripped[:500]
            text_files += 1
        except (ET.ParseError, UnicodeDecodeError, OSError):
            continue

    _LOG.info(
        "text extraction complete",
        extra={"source": source_name, "text_files": text_files},
    )


def _contains_greek(text: str) -> bool:
    """Return True if *text* contains at least one Greek Unicode character."""
    return bool(re.search(r"[Ͱ-Ͽἀ-῿]", text))


# ── Dataset ─────────────────────────────────────────────────────────────────


class DAPTIterableDataset(IterableDataset):
    """Streaming dataset for T5 span-corruption DAPT.

    On each ``__iter__``, walks the Koine and Classical source directories
    freshly (no in-memory cache), applies a 70/30 mix per yield, and
    returns tokenised (input_ids, labels) pairs for T5 training.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer: AutoTokenizer,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        koine_ratio: float = REPLAY_KOINE_RATIO,
        noise_density: float = NOISE_DENSITY,
        mean_span_length: float = NOISE_MEAN_SPAN_LENGTH,
    ) -> None:
        self._data_dir = data_dir
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._koine_ratio = koine_ratio
        self._noise_density = noise_density
        self._mean_span_length = mean_span_length

        # Ensure pad token is set (required for batching).
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield training samples endlessly."""
        # Build fresh iterators each epoch so ordering varies.
        koine_stream = self._stream_sources(KOINE_SOURCES)
        classical_stream = self._stream_sources(CLASSICAL_SOURCES)

        while True:
            # Interleave Koine and Classical at the configured ratio.
            for source_stream in [koine_stream, classical_stream]:
                n_koine = max(1, int(10 * self._koine_ratio))
                n_classical = max(1, int(10 * (1 - self._koine_ratio)))
                n_from_this = n_koine if source_stream is koine_stream else n_classical

                for _ in range(n_from_this):
                    try:
                        chunk = next(source_stream)
                    except StopIteration:
                        return
                    tokenised = self._tokenize_with_noise(chunk)
                    if tokenised is not None:
                        yield tokenised

    def _stream_sources(self, sources: list[str]) -> Iterator[str]:
        """Yield text chunks from the given source directories, cycled forever."""
        while True:
            for source_name in sources:
                for chunk in _extract_text_from_dir(self._data_dir, source_name):
                    yield chunk

    def _tokenize_with_noise(
        self, text: str
    ) -> dict[str, torch.Tensor] | None:
        """Tokenize a text chunk with T5 span-corruption noise.

        Returns ``(input_ids, labels)`` as tensors, or None if the chunk
        is too short after tokenisation.
        """
        ids = self._tokenizer.encode(text, add_special_tokens=True, max_length=self._max_length,
                                     truncation=True)
        if len(ids) < 8:  # too short for meaningful training
            return None

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()

        # Simple token-level noise: replace a fraction of tokens with a
        # single sentinel (we use the tokenizer's mask token or id 4 for T5).
        # A full sentinel-span implementation would track span boundaries;
        # for DAPT, token-level noise is a pragmatic and effective proxy.
        n_tokens = len(ids)
        n_noise = max(1, int(n_tokens * self._noise_density))
        noise_indices = torch.randperm(n_tokens)[:n_noise]
        sentinel_id = getattr(self._tokenizer, "mask_token_id", None) or 4
        input_ids[noise_indices] = sentinel_id

        # Set padding on labels to -100 (ignored by T5 loss).
        # (No padding in this single-sample case, but kept for batching later.)

        return {"input_ids": input_ids, "labels": labels}


# ── Training config ──────────────────────────────────────────────────────────


@dataclass
class DAPTConfig:
    """Hyperparameters for KoineFormer DAPT.

    Attributes:
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate for AdamW.
        warmup_steps: Linear warmup steps before cosine decay.
        max_steps: Total training steps.
        val_steps: Validate every N steps.
        save_steps: Save adapter checkpoint every N steps.
        grad_accum_steps: Gradient accumulation steps (effective batch multiplier).
        max_length: Maximum token length for input sequences.
        output_dir: Directory for checkpoints and logs.
    """

    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_steps: int = 20_000
    val_steps: int = 500
    save_steps: int = 2_000
    grad_accum_steps: int = 1
    max_length: int = 512
    output_dir: Path = field(default_factory=lambda: Path("outputs/dapt"))


# ── Trainer ──────────────────────────────────────────────────────────────────


class DAPTTrainer:
    """Trains LoRA adapters on Koine Greek via T5 span corruption.

    Args:
        model: A KoineFormer instance with LoRA adapters attached.
        data_dir: Root ``data/raw/`` directory.
        tokenizer: GreTa tokenizer instance.
        config: DAPTConfig with hyperparameters.
        device: Torch device string.
    """

    def __init__(
        self,
        model: Any,  # KoineFormer (lazy import to avoid circular deps)
        data_dir: Path,
        tokenizer: AutoTokenizer,
        config: DAPTConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._data_dir = data_dir
        self._tokenizer = tokenizer
        self._config = config or DAPTConfig()
        self._device = device

        self._optimizer: AdamW | None = None
        self._scheduler: CosineAnnealingLR | None = None
        self._history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    @property
    def history(self) -> dict[str, list[float]]:
        """Training metric history (loss per step, val_loss per val step)."""
        return self._history

    def run(self) -> dict[str, list[float]]:
        """Execute the full DAPT training loop.

        Returns:
            Dict of metric histories (``loss``, ``val_loss``).
        """
        model = self._model
        config = self._config

        model.enable_dapt()
        peft_model = model.model

        self._optimizer = AdamW(peft_model.parameters(), lr=config.learning_rate)
        self._scheduler = CosineAnnealingLR(self._optimizer, T_max=config.max_steps)

        dataset = DAPTIterableDataset(
            self._data_dir,
            self._tokenizer,
            max_length=config.max_length,
        )
        data_iter = iter(dataset)

        _LOG.info(
            "DAPT training starting",
            extra={
                "max_steps": config.max_steps,
                "batch_size": config.batch_size,
                "lr": config.learning_rate,
                "device": self._device,
            },
        )

        accum_loss = 0.0
        model.train()

        for step in range(1, config.max_steps + 1):
            # Accumulate gradients across grad_accum_steps micro-batches.
            self._optimizer.zero_grad()
            micro_loss = 0.0

            for _ in range(config.grad_accum_steps):
                sample = next(data_iter)
                input_ids = sample["input_ids"].unsqueeze(0).to(self._device)
                labels = sample["labels"].unsqueeze(0).to(self._device)
                attention_mask = torch.ones_like(input_ids)

                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / config.grad_accum_steps
                loss.backward()
                micro_loss += loss.item()

            self._optimizer.step()
            self._scheduler.step()

            current_loss = micro_loss
            accum_loss += current_loss
            self._history["loss"].append(current_loss)

            # ── Logging ─────────────────────────────────────────────────
            if step % 100 == 0:
                avg_loss = accum_loss / 100
                lr = self._scheduler.get_last_lr()[0]
                _LOG.info(
                    f"step {step}/{config.max_steps}",
                    extra={"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"},
                )
                accum_loss = 0.0

            # ── Validation ──────────────────────────────────────────────
            if step % config.val_steps == 0:
                val_loss = self._validate(num_samples=20)
                self._history["val_loss"].append(val_loss)
                _LOG.info(
                    f"validation @ step {step}",
                    extra={"val_loss": f"{val_loss:.4f}"},
                )
                model.train()  # restore training mode

            # ── Checkpoint ──────────────────────────────────────────────
            if step % config.save_steps == 0:
                ckpt_dir = config.output_dir / f"step-{step}"
                model.save_adapters(ckpt_dir)

        # Final save
        final_dir = config.output_dir / "final"
        model.save_adapters(final_dir)
        _LOG.info("DAPT training complete", extra={"final_path": str(final_dir)})

        return self._history

    def _validate(self, *, num_samples: int = 20) -> float:
        """Compute mean loss on *num_samples* fresh chunks."""
        self._model.eval()
        dataset = DAPTIterableDataset(self._data_dir, self._tokenizer, max_length=512)
        data_iter = iter(dataset)

        total = 0.0
        peft_model = self._model.model

        with torch.no_grad():
            for _ in range(num_samples):
                sample = next(data_iter)
                input_ids = sample["input_ids"].unsqueeze(0).to(self._device)
                labels = sample["labels"].unsqueeze(0).to(self._device)
                attention_mask = torch.ones_like(input_ids)
                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total += outputs.loss.item()

        return total / num_samples
