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
import signal
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
        """Yield training samples with full-length sequences.

        Concatenates text chunks to fill ``max_length`` tokens, applies
        T5 span-corruption noise, and yields (input_ids, labels) pairs.
        This ensures the model trains on long contexts rather than
        30-100 token fragments.
        """
        koine_stream = self._stream_sources(KOINE_SOURCES)
        classical_stream = self._stream_sources(CLASSICAL_SOURCES)

        # Token buffer: collect tokens until we have enough for one sample.
        buffer: list[int] = []

        def _fill_buffer(source_iter: Iterator[str], target_total: int) -> None:
            """Pull chunks until buffer reaches *target_total* tokens."""
            nonlocal buffer
            while len(buffer) < target_total:
                try:
                    chunk = next(source_iter)
                except StopIteration:
                    return
                ids = self._tokenizer.encode(
                    chunk, add_special_tokens=False, max_length=self._max_length,
                    truncation=True,
                )
                buffer.extend(ids)

        while True:
            # 70/30 mix: 7 Koine then 3 Classical, cycling.
            for source_iter, n_blocks in [(koine_stream, 7), (classical_stream, 3)]:
                for _ in range(n_blocks):
                    _fill_buffer(source_iter, self._max_length)

                    if len(buffer) < 8:  # not enough tokens
                        continue

                    # Take a full-length slice from the buffer.
                    seq = buffer[:self._max_length]
                    buffer = buffer[self._max_length:]

                    # Apply span-corruption noise.
                    input_ids = torch.tensor(seq, dtype=torch.long)
                    labels = input_ids.clone()
                    n_tokens = len(seq)
                    n_noise = max(1, int(n_tokens * self._noise_density))
                    noise_idx = torch.randperm(n_tokens)[:n_noise]
                    sentinel = getattr(self._tokenizer, "mask_token_id", None) or 4
                    input_ids[noise_idx] = sentinel

                    yield {"input_ids": input_ids, "labels": labels}

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
        use_amp: Enable Automatic Mixed Precision (FP16) for ~2× speedup.
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
    use_amp: bool = True
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

    # Sentinel file written before each checkpoint save — if it exists when
    # loading, the checkpoint is incomplete and should be skipped.
    _INCOMPLETE_MARKER = ".incomplete"

    def run(
        self,
        *,
        resume: bool = True,
        commit_volume: bool = False,
        volume: Any = None,  # Modal Volume object
    ) -> dict[str, list[float]]:
        """Execute the DAPT training loop with crash-safe checkpoints.

        Args:
            resume: If True, auto-detect and resume from the latest checkpoint.
            commit_volume: If True, call ``volume.commit()`` after each save
                           (Modal persistence — survives spot preemption).
            volume: Modal Volume object (required if *commit_volume* is True).

        Returns:
            Dict of metric histories (``loss``, ``val_loss``).
        """
        model = self._model
        config = self._config
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume or fresh start ──────────────────────────────────────
        start_step = 0
        if resume:
            start_step = self._find_latest_checkpoint()
            if start_step > 0:
                _LOG.info(f"resuming from step {start_step}")
                self._load_training_state(output_dir / f"step-{start_step}")
            else:
                _LOG.info("no checkpoint found — starting fresh")

        model.enable_dapt()
        peft_model = model.model

        if self._optimizer is None:
            self._optimizer = AdamW(peft_model.parameters(), lr=config.learning_rate)
        if self._scheduler is None:
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=config.max_steps)

        dataset = DAPTIterableDataset(
            self._data_dir, self._tokenizer, max_length=config.max_length,
        )
        data_iter = iter(dataset)

        # Skip ahead to resume point (replay data stream).
        for _ in range(start_step):
            next(data_iter)

        use_amp = config.use_amp and self._device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ── SIGTERM handler (spot preemption) ──────────────────────────
        interrupted = False

        def _handle_sigterm(signum: int, frame: Any) -> None:
            nonlocal interrupted
            _LOG.warning("SIGTERM received — saving emergency checkpoint")
            interrupted = True

        prev_handler = signal.signal(signal.SIGTERM, _handle_sigterm)

        _LOG.info(
            "DAPT training starting",
            extra={
                "start_step": start_step,
                "max_steps": config.max_steps,
                "batch_size": config.batch_size,
                "lr": config.learning_rate,
                "device": self._device,
                "amp": use_amp,
            },
        )

        accum_loss = 0.0
        model.train()

        try:
            for step in range(start_step + 1, config.max_steps + 1):
                if interrupted:
                    self._save_checkpoint(step, scaler, commit_volume, volume, emergency=True)
                    break

                self._optimizer.zero_grad()
                micro_loss = 0.0

                for _ in range(config.grad_accum_steps):
                    sample = next(data_iter)
                    input_ids = sample["input_ids"].unsqueeze(0).to(self._device)
                    labels = sample["labels"].unsqueeze(0).to(self._device)
                    attention_mask = torch.ones_like(input_ids)

                    with torch.amp.autocast("cuda") if use_amp else torch.no_grad():
                        outputs = peft_model(
                            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                        )
                    loss = outputs.loss / config.grad_accum_steps
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    micro_loss += loss.item()

                if scaler is not None:
                    scaler.step(self._optimizer)
                    scaler.update()
                else:
                    self._optimizer.step()
                self._scheduler.step()

                self._history["loss"].append(micro_loss)
                accum_loss += micro_loss

                if step % 100 == 0:
                    avg_loss = accum_loss / 100
                    lr = self._scheduler.get_last_lr()[0]
                    _LOG.info(f"step {step}/{config.max_steps}", extra={
                        "loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}",
                    })
                    accum_loss = 0.0

                if step % config.val_steps == 0:
                    val_loss = self._validate(num_samples=20)
                    self._history["val_loss"].append(val_loss)
                    _LOG.info(f"validation @ {step}", extra={"val_loss": f"{val_loss:.4f}"})
                    model.train()

                if step % config.save_steps == 0 and step > start_step:
                    self._save_checkpoint(step, scaler, commit_volume, volume)

        finally:
            signal.signal(signal.SIGTERM, prev_handler)

        # Final save (only if not interrupted)
        if not interrupted:
            self._save_checkpoint(config.max_steps, scaler, commit_volume, volume, final=True)
            _LOG.info("DAPT training complete", extra={"final_path": str(output_dir / "final")})
        else:
            _LOG.info("DAPT training interrupted — checkpoint saved, resume later")

        return self._history

    # ── Checkpoint persistence ───────────────────────────────────────────

    def _find_latest_checkpoint(self) -> int:
        """Return the highest completed checkpoint step, or 0 if none."""
        output_dir = self._config.output_dir
        if not output_dir.exists():
            return 0
        steps = []
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith("step-"):
                try:
                    s = int(d.name.split("-")[1])
                    # Skip incomplete checkpoints
                    if not (d / self._INCOMPLETE_MARKER).exists():
                        # Verify adapter file exists and is non-empty
                        adapter = d / "adapter_model.safetensors"
                        if adapter.exists() and adapter.stat().st_size > 1000:
                            steps.append(s)
                except (ValueError, IndexError):
                    continue
        return max(steps) if steps else 0

    def _save_checkpoint(
        self,
        step: int,
        scaler: Any,
        commit_volume: bool,
        volume: Any,
        *,
        emergency: bool = False,
        final: bool = False,
    ) -> None:
        """Save adapters + full training state to a crash-safe checkpoint."""
        output_dir = self._config.output_dir
        label = "final" if final else f"step-{step}"
        ckpt_dir = output_dir / label

        # Mark as incomplete before writing (crash-safe).
        marker = ckpt_dir / self._INCOMPLETE_MARKER
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("")

        # Save adapters.
        self._model.save_adapters(ckpt_dir)

        # Save full training state for exact resume.
        state = {
            "step": step,
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler": self._scheduler.state_dict() if self._scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "history": self._history,
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        # Remove incomplete marker — checkpoint is now valid.
        marker.unlink()

        # Commit Modal volume so checkpoint survives spot preemption.
        if commit_volume and volume is not None:
            try:
                volume.commit()
            except Exception as exc:
                _LOG.warning(f"volume commit failed: {exc}")

        tag = "emergency" if emergency else ("final" if final else "")
        _LOG.info(f"checkpoint saved {tag}", extra={"step": step, "path": str(ckpt_dir)})

    def _load_training_state(self, ckpt_dir: Path) -> None:
        """Restore optimizer, scheduler, scaler, and history from a checkpoint."""
        state_path = ckpt_dir / "training_state.pt"
        if not state_path.exists():
            _LOG.warning("no training state found — optimizer reset")
            return

        state = torch.load(state_path, map_location=self._device, weights_only=False)

        # Restore adapters
        self._model.load_adapters(ckpt_dir)

        # Restore optimizer
        if state.get("optimizer") and self._model.model is not None:
            self._optimizer = AdamW(self._model.model.parameters(), lr=self._config.learning_rate)
            self._optimizer.load_state_dict(state["optimizer"])

        # Restore scheduler
        if state.get("scheduler") and self._optimizer:
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=self._config.max_steps)
            if hasattr(self._scheduler, "load_state_dict"):
                self._scheduler.load_state_dict(state["scheduler"])

        # Restore history
        if state.get("history"):
            self._history = state["history"]

        _LOG.info("training state restored", extra={"step": state.get("step", "?")})

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
