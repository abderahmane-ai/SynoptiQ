"""GreTa tokenizer wrapper for Koine Greek text.

Wraps the HuggingFace ``bowphs/GreTa`` T5 tokenizer and extends it
with nomina sacra special tokens and a Koine-aware encoding interface.

The GreTa tokenizer uses SentencePiece with a ~32,128 token vocabulary.
We add nomina sacra tokens as additional special tokens so the model can
learn manuscript-aware representations when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synoptiq.utils.types_ import TokenRecord

# Model ID on HuggingFace Hub — confirmed correct as of 2026-07
GRETA_MODEL_ID: str = "bowphs/GreTa"

# Nomina sacra as additional special tokens (abbreviated forms in Greek manuscripts)
# These are the normalized (de-accented, lowercase) forms.
NOMINA_SACRA_TOKENS: list[str] = [
    "<θς>",  # θεος (God)
    "<κς>",  # κυριος (Lord)
    "<ιης>",  # ιησους (Jesus)
    "<χς>",  # χριστος (Christ)
    "<πνα>",  # πνευμα (Spirit)
    "<πηρ>",  # πατηρ (Father)
    "<υς>",  # υιος (Son)
    "<σηρ>",  # σωτηρ (Savior)
    "<δαδ>",  # δαυιδ (David)
    "<ιηλ>",  # ισραηλ (Israel)
    "<ανος>",  # ανθρωπος (human/man)
    "<ουνος>",  # ουρανος (heaven)
]


class KoineTokenizer:
    """GreTa tokenizer extended with nomina sacra tokens and a Koine-aware API.

    This is a thin wrapper around HuggingFace AutoTokenizer that:
    1. Loads the ``bowphs/GreTa`` SentencePiece tokenizer
    2. Adds nomina sacra as extra special tokens
    3. Exposes ``encode_tokens`` and ``decode_ids`` for use with TokenRecord lists

    Attributes:
        tokenizer: The underlying HuggingFace tokenizer instance.
        model_id: The HuggingFace model ID used.
        vocab_size: Total vocabulary size (including added tokens).

    Example:
        >>> tok = KoineTokenizer.from_pretrained()
        >>> tokens = [{"text": "λόγος", ...}, ...]
        >>> ids = tok.encode_tokens(tokens)
    """

    def __init__(self, tokenizer: Any) -> None:  # noqa: F821
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = GRETA_MODEL_ID,
        *,
        cache_dir: Path | str | None = None,
        add_nomina_sacra: bool = True,
    ) -> KoineTokenizer:
        """Load the GreTa tokenizer from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (default: ``bowphs/GreTa``).
            cache_dir: Optional local cache directory.
            add_nomina_sacra: If True, adds nomina sacra as special tokens.

        Returns:
            Initialized KoineTokenizer.

        Raises:
            OSError: If the model cannot be downloaded or loaded.
        """
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        kwargs: dict[str, object] = {"use_fast": False}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

        if add_nomina_sacra:
            tokenizer.add_special_tokens({"additional_special_tokens": NOMINA_SACRA_TOKENS})

        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including added tokens."""
        return len(self._tokenizer)

    @property
    def tokenizer(self) -> Any:  # noqa: F821
        """The underlying HuggingFace tokenizer."""
        return self._tokenizer

    def encode_tokens(
        self,
        records: list[TokenRecord],
        *,
        max_length: int = 512,
        use_surface: bool = True,
    ) -> dict[str, list[int]]:
        """Encode a list of TokenRecords into model input IDs.

        Args:
            records: List of TokenRecord dicts from the corpus.
            max_length: Maximum sequence length (default 512).
            use_surface: If True, encode surface text; if False, encode
                normalized (de-accented) form.

        Returns:
            Dict with ``input_ids`` and ``attention_mask`` as int lists.
        """
        text = " ".join(r["text"] if use_surface else r["normalized"] for r in records)
        encoding = self._tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,  # Return plain lists
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    def decode_ids(self, input_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to Greek text.

        Args:
            input_ids: List of token IDs.
            skip_special_tokens: If True, remove padding and special tokens.

        Returns:
            Decoded Greek text string.
        """
        return self._tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)

    def tokenize_text(self, text: str) -> list[str]:
        """Tokenize a raw Greek string into subword tokens.

        Args:
            text: Raw Greek text string.

        Returns:
            List of subword token strings (with sentencepiece markers).
        """
        return self._tokenizer.tokenize(text)
