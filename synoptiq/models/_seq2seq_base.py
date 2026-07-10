"""Shared loader for the Phase-5 seq2seq models (redactor + Fusion-in-Decoder).

Both the redaction operators and the FiD reconstructor are GreTa (optionally with
the decontaminated KoineFormer-NS knowledge merged in) plus a fresh task-specific
LoRA. This module centralises that assembly so the two model classes stay thin and
so a tiny stand-in T5 can be injected in tests without touching the loading path.
"""

from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from synoptiq.models.koineformer import GRETA_MODEL_ID
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)


def default_lora(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """Task LoRA for seq2seq generation — same target modules as the DAPT config."""
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q", "v", "o", "wi", "wo"],
    )


def load_greta_seq2seq(
    *,
    base_model_id: str = GRETA_MODEL_ID,
    init_adapters: Path | str | None = None,
    lora: LoraConfig | None = None,
    device: str = "cpu",
) -> tuple[object, AutoTokenizer]:
    """Load GreTa (+ optional merged NS adapters) and attach a fresh task LoRA.

    Args:
        base_model_id: HuggingFace id of the base encoder-decoder.
        init_adapters: Path to KoineFormer-NS adapters to *merge* into the base
            before the new LoRA (so the redactor starts from decontaminated Koine
            knowledge, not raw Classical GreTa). ``None`` uses the plain base.
        lora: Task LoRA config; ``default_lora()`` if omitted.
        device: Torch device.

    Returns:
        (peft_model, tokenizer). The tokenizer has a pad token and the model's
        embeddings are resized to match before LoRA is attached.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # GreTa's vocab already contains <pad>=0 and </s>=1 (matching config.pad_token_id
    # and config.eos_token_id / decoder_start_token_id), but the tokenizer object
    # leaves them unset. Point it at the EXISTING tokens — do NOT add a new [PAD] and
    # resize, which desyncs the tokenizer pad id from the model's pad/decoder-start id
    # and collapses generation to all-<pad> (empty output).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"  # id 0
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"   # id 1

    base = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
    if hasattr(base.config, "tie_word_embeddings"):
        base.config.tie_word_embeddings = False

    if init_adapters is not None:
        _LOG.info("merging init adapters into base", extra={"path": str(init_adapters)})
        base = PeftModel.from_pretrained(base, str(init_adapters)).merge_and_unload()

    model = get_peft_model(base, lora or default_lora())
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _LOG.info("seq2seq model ready", extra={"trainable_params": trainable, "device": device})
    return model, tokenizer
