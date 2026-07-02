"""KoineFormer: GreTa + LoRA domain-adapted for Koine Greek.

Loads the frozen GreTa T5 encoder-decoder and injects LoRA adapters
for parameter-efficient domain-adaptive pre-training on Koine Greek.
Produces ~3.7M trainable parameters out of 251M total (1.5%).

Usage:
    model = KoineFormer.from_pretrained()
    model.enable_dapt()          # unfreeze LoRA for DAPT training
    model.freeze()               # freeze everything for inference
    model.save_adapters(path)    # save ~18 MB of adapter weights
    model.load_adapters(path)    # restore adapters from disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

GRETA_MODEL_ID: str = "bowphs/GreTa"

# LoRA configuration matching the implementation plan specification.
# r=16, alpha=32 on attention projections (q, v, o) and feed-forward (wi, wo).
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v", "o", "wi", "wo"],
)

# Adapter weight filenames saved alongside the config.
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"


class KoineFormer:
    """Domain-adapted T5 encoder-decoder for Koine Greek.

    Wraps a frozen ``bowphs/GreTa`` checkpoint with LoRA adapters injected
    into attention projections and feed-forward layers.  The adapters are
    the ONLY trainable parameters during DAPT.

    Attributes:
        model: The underlying PeftModel (GreTa + LoRA).
        base_model: The frozen GreTa T5 (accessible via ``model.base_model``).
        model_id: HuggingFace model identifier.
        device: Torch device string (``"cpu"``, ``"cuda"``, or ``"mps"``).
    """

    def __init__(self, model: PeftModel, *, model_id: str = GRETA_MODEL_ID) -> None:
        self._model = model
        self._model_id = model_id
        self._device = str(next(model.parameters()).device)

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = GRETA_MODEL_ID,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KoineFormer:
        """Load GreTa and inject LoRA adapters.

        Args:
            model_id: HuggingFace model ID.
            device: Torch device.  Auto-detected if None.
            dtype: Model dtype.  ``torch.float32`` if None.

        Returns:
            Initialised KoineFormer with LoRA adapters attached.
        """
        if device is None:
            device = _detect_device()
        if dtype is None:
            dtype = torch.float32

        _LOG.info("loading GreTa", extra={"model_id": model_id, "device": device})
        base = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)

        # Silence the tied-weights warning (known GreTa checkpoint quirk).
        if hasattr(base.config, "tie_word_embeddings"):
            base.config.tie_word_embeddings = False

        _LOG.info("injecting LoRA adapters", extra={"r": 16, "alpha": 32})
        peft_model = get_peft_model(base, LORA_CONFIG)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        _LOG.info(
            "KoineFormer ready",
            extra={
                "trainable_params": trainable,
                "total_params": total,
                "pct_trainable": f"{100 * trainable / total:.1f}%",
            },
        )
        return cls(peft_model, model_id=model_id)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def model(self) -> PeftModel:
        """The underlying PeftModel (GreTa base + LoRA adapters)."""
        return self._model

    @property
    def model_id(self) -> str:
        """HuggingFace model identifier."""
        return self._model_id

    @property
    def device(self) -> str:
        """Torch device string."""
        return self._device

    # ── Training mode control ─────────────────────────────────────────────

    def enable_dapt(self) -> None:
        """Enable LoRA adapters for DAPT training; base GreTa stays frozen."""
        for name, param in self._model.named_parameters():
            param.requires_grad = "lora" in name
        self._model.train()
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        _LOG.info("DAPT mode enabled", extra={"trainable_params": trainable})

    def freeze(self) -> None:
        """Freeze all parameters for inference."""
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

    def train(self) -> None:
        """Set model to training mode (LoRA layers only)."""
        self._model.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self._model.eval()

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Execute a forward pass through the PEFT model.

        Delegates directly to ``PeftModel.forward``, which applies LoRA
        deltas on top of the frozen GreTa base.

        Args:
            input_ids: Tokenised input [batch, seq_len].
            attention_mask: Padding mask [batch, seq_len].
            labels: Target token IDs for loss computation.

        Returns:
            Dict with ``loss`` (if labels provided) and ``logits``.
        """
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate text from input token IDs.

        Args:
            input_ids: Tokenised prompt [batch, seq_len].
            max_new_tokens: Maximum tokens to generate.
            pad_token_id: Padding token ID.  Uses EOS if not set.
            **kwargs: Additional generation arguments passed to ``model.generate``.

        Returns:
            Tensor of generated token IDs [batch, gen_len].
        """
        if pad_token_id is None:
            pad_token_id = self._model.config.eos_token_id
        return self._model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    # ── Persistence ─────────────────────────────────────────────────────

    def save_adapters(self, path: Path | str) -> None:
        """Save LoRA adapter weights and config to disk (~18 MB).

        Args:
            path: Destination directory.  Created if it doesn't exist.
        """
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(dest))
        _LOG.info("adapters saved", extra={"path": str(dest)})

    def load_adapters(self, path: Path | str) -> None:
        """Load LoRA adapter weights from disk, replacing current adapters.

        Args:
            path: Directory containing ``adapter_model.safetensors`` and
                  ``adapter_config.json``.

        Raises:
            FileNotFoundError: If adapter files are missing.
        """
        src = Path(path)
        if not src.exists():
            msg = f"Adapter directory not found: {src}"
            raise FileNotFoundError(msg)
        # PeftModel.load_adapter handles the weight merging.
        self._model.load_adapter(str(src), "default")
        _LOG.info("adapters loaded", extra={"path": str(src)})

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return (
            f"KoineFormer(model={self._model_id!r}, "
            f"total_params={total:,}, trainable={trainable:,}, "
            f"device={self._device!r})"
        )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _detect_device() -> str:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
