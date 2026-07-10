"""Fusion-in-Decoder: reconstruct a source from several witnesses (Track A / Q).

Two witnesses (Matthew, Luke) are encoded *independently*, their encoder hidden
states are concatenated along the sequence axis, and a single decoder cross-attends
over the concatenation to generate the target (Mark, where ground truth exists; then
proto-Q on the double tradition). This is the Fusion-in-Decoder architecture of
Izacard & Grave (2021), and it is the study's Track-A deliverable and the 2SH branch
of the E1 channel test.

Encoding witnesses separately (rather than concatenating raw text) means the model
scales to variable numbers of witnesses and never has to attend across two long
inputs at encode time — the fusion happens only in the decoder's cross-attention.

The class wraps a T5 (GreTa + task LoRA); tests inject a tiny T5 so the forward and
fusion logic run on CPU with no download.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers.modeling_outputs import BaseModelOutput

from synoptiq.evaluation.scoring import sequence_nll
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

IGNORE_INDEX = -100


class FusionInDecoder:
    """Encode witnesses separately, fuse in the decoder, generate the source."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        init_adapters: Path | str | None = None,
        device: str | None = None,
        lora: Any = None,
    ) -> FusionInDecoder:
        """Load GreTa (+ optional merged NS adapters) with a fresh fusion LoRA."""
        from synoptiq.models._seq2seq_base import load_greta_seq2seq

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_greta_seq2seq(init_adapters=init_adapters, lora=lora, device=device)
        return cls(model, tokenizer, device=device)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    # ── Fusion core ─────────────────────────────────────────────────────────────

    def fuse_witnesses(
        self,
        witness_input_ids: Sequence[torch.Tensor],
        witness_masks: Sequence[torch.Tensor],
    ) -> tuple[BaseModelOutput, torch.Tensor]:
        """Encode each witness and concatenate along the sequence axis.

        Args:
            witness_input_ids: One ``[batch, L_i]`` tensor per witness.
            witness_masks: Matching ``[batch, L_i]`` attention masks.

        Returns:
            (encoder_outputs, combined_mask) where the encoder hidden state is
            ``[batch, sum_i L_i, hidden]`` and the mask is ``[batch, sum_i L_i]``.
        """
        if len(witness_input_ids) != len(witness_masks):
            msg = "witness_input_ids and witness_masks must have equal length"
            raise ValueError(msg)
        if not witness_input_ids:
            msg = "need at least one witness"
            raise ValueError(msg)

        encoder = self._model.get_encoder()
        hidden_states, masks = [], []
        for ids, mask in zip(witness_input_ids, witness_masks, strict=True):
            enc = encoder(input_ids=ids, attention_mask=mask)
            hidden_states.append(enc.last_hidden_state)
            masks.append(mask)
        combined = torch.cat(hidden_states, dim=1)
        combined_mask = torch.cat(masks, dim=1)
        return BaseModelOutput(last_hidden_state=combined), combined_mask

    def forward(
        self,
        witness_input_ids: Sequence[torch.Tensor],
        witness_masks: Sequence[torch.Tensor],
        labels: torch.Tensor,
    ) -> Any:
        """Teacher-forced forward over fused witnesses; returns HF output."""
        encoder_outputs, combined_mask = self.fuse_witnesses(witness_input_ids, witness_masks)
        return self._model(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            labels=labels,
        )

    @torch.no_grad()
    def score(
        self,
        witness_input_ids: Sequence[torch.Tensor],
        witness_masks: Sequence[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sequence mean NLL of the target given the fused witnesses."""
        self._model.eval()
        out = self.forward(witness_input_ids, witness_masks, labels)
        return sequence_nll(out.logits, labels, reduction="mean")

    @torch.no_grad()
    def generate(
        self,
        witness_input_ids: Sequence[torch.Tensor],
        witness_masks: Sequence[torch.Tensor],
        *,
        max_new_tokens: int = 256,
        num_beams: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate the reconstructed source from the fused witnesses."""
        self._model.eval()
        encoder_outputs, combined_mask = self.fuse_witnesses(witness_input_ids, witness_masks)
        return self._model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **kwargs,
        )

    # ── Encoding helper ─────────────────────────────────────────────────────────

    def encode_example(
        self,
        witness_texts: Sequence[str],
        target_text: str,
        *,
        max_source: int = 256,
        max_target: int = 256,
    ) -> dict[str, Any]:
        """Tokenise (list of witness strings, target string) into a batch of 1."""
        ids, masks = [], []
        for text in witness_texts:
            enc = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_source, padding=False
            )
            ids.append(enc["input_ids"].to(self._device))
            masks.append(enc["attention_mask"].to(self._device))
        tgt = self._tokenizer(
            target_text, return_tensors="pt", truncation=True, max_length=max_target, padding=False
        )
        labels = tgt["input_ids"].clone()
        labels[labels == self._tokenizer.pad_token_id] = IGNORE_INDEX
        return {
            "witness_input_ids": ids,
            "witness_masks": masks,
            "labels": labels.to(self._device),
        }

    def train(self) -> None:
        self._model.train()

    def eval(self) -> None:
        self._model.eval()

    def save_adapters(self, path: Path | str) -> None:
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(dest))
        _LOG.info("FiD adapters saved", extra={"path": str(dest)})

    def load_adapters(self, path: Path | str) -> None:
        self._model.load_adapter(str(path), adapter_name="default")
        _LOG.info("FiD adapters loaded", extra={"path": str(path)})

    def __repr__(self) -> str:
        return f"FusionInDecoder(device={self._device!r})"
