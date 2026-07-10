"""Redaction operators: seq2seq models of how one evangelist rewrites a source.

A ``Redactor`` maps source Greek to target Greek. Four instances, differing only in
training data, drive the source-criticism study (docs/SOURCE_CRITICISM_STUDY.md §3):

    R_Lk : source → Luke     (trained on Mk_p → Lk_p)   "how Luke redacts"
    R_Mt : source → Matthew  (trained on Mk_p → Mt_p)   "how Matthew redacts"
    G_Mt : Matthew → source  (trained on Mt_p → Mk_p)   single-witness reconstructor
    G_Lk : Luke → source     (trained on Lk_p → Mk_p)

The class is a thin wrapper over a T5 seq2seq (GreTa + task LoRA, optionally starting
from the decontaminated KoineFormer-NS). ``score`` returns teacher-forced per-sequence
NLL — the quantity every Track-B verdict is a paired difference of — reusing the
audited scoring backbone in ``synoptiq.evaluation.scoring``.

Construct with an injected model+tokenizer (tests use a tiny T5) or via
``Redactor.from_pretrained`` (loads GreTa).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from synoptiq.evaluation.scoring import sequence_nll
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

IGNORE_INDEX = -100


class Redactor:
    """A source→target seq2seq redaction operator with NLL scoring."""

    def __init__(self, model: Any, tokenizer: Any, *, name: str = "R", device: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._name = name
        self._device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        name: str = "R",
        init_adapters: Path | str | None = None,
        device: str | None = None,
        lora: Any = None,
    ) -> Redactor:
        """Load GreTa (+ optional merged NS adapters) with a fresh redaction LoRA."""
        from synoptiq.models._seq2seq_base import load_greta_seq2seq

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_greta_seq2seq(init_adapters=init_adapters, lora=lora, device=device)
        return cls(model, tokenizer, name=name, device=device)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_pair(
        self, source: str, target: str, *, max_source: int = 256, max_target: int = 256
    ) -> dict[str, torch.Tensor]:
        """Tokenise a (source, target) string pair into a training/scoring batch of 1.

        Target pad positions are masked to ``IGNORE_INDEX`` so they do not count
        toward the loss.
        """
        src = self._tokenizer(
            source, return_tensors="pt", truncation=True, max_length=max_source, padding=False
        )
        tgt = self._tokenizer(
            target, return_tensors="pt", truncation=True, max_length=max_target, padding=False
        )
        labels = tgt["input_ids"].clone()
        labels[labels == self._tokenizer.pad_token_id] = IGNORE_INDEX
        return {
            "input_ids": src["input_ids"].to(self._device),
            "attention_mask": src["attention_mask"].to(self._device),
            "labels": labels.to(self._device),
        }

    # ── Forward / score / generate ─────────────────────────────────────────────

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> Any:
        """Teacher-forced forward; returns the HF output (``.loss``, ``.logits``)."""
        return self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def score(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Per-sequence mean NLL (nats/token) of ``labels`` given the source.

        This is the conditional NLL ``−log p(target | source)`` the verdicts compare.
        """
        self._model.eval()
        out = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return sequence_nll(out.logits, labels, reduction="mean")

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, max_new_tokens: int = 256,
        num_beams: int = 4, **kwargs: Any,
    ) -> torch.Tensor:
        """Generate a target sequence from the source."""
        self._model.eval()
        return self._model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams, **kwargs,
        )

    # ── Mode + persistence ──────────────────────────────────────────────────────

    def train(self) -> None:
        self._model.train()

    def eval(self) -> None:
        self._model.eval()

    def save_adapters(self, path: Path | str) -> None:
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(dest))
        _LOG.info("redactor adapters saved", extra={"operator": self._name, "path": str(dest)})

    def load_adapters(self, path: Path | str) -> None:
        self._model.load_adapter(str(path), adapter_name="default")
        _LOG.info("redactor adapters loaded", extra={"operator": self._name, "path": str(path)})

    def __repr__(self) -> str:
        return f"Redactor(name={self._name!r}, device={self._device!r})"
