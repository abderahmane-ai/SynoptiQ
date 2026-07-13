"""Neural reading engine — analyse *arbitrary* Koine with the published Koine-T5.

The gold engine (:mod:`synoptiq.reader.gold`) only covers the canonical GNT/LXX. This
engine is the wedge: it tags and lemmatises any Koine a user pastes — papyri,
apocrypha, Apostolic Fathers, patristics — using
`ainouche-abderahmane/koine-t5 <https://huggingface.co/ainouche-abderahmane/koine-t5>`_,
then glosses each predicted lemma against a gold lexicon (typically
:meth:`GoldReader.lexicon`). Output is **predicted**, not gold: part of speech and
lemma only (Koine-T5 does not emit full morphology), flagged as such.

The inference recipe (tokenizer pad/eos, the 100 ``<extra_id>`` sentinels with *no*
embedding resize, POS upper-casing) mirrors the model card exactly. torch/transformers
are imported lazily so importing this module never pulls in the deep stack.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synoptiq.reader.gold import ReadResult, WordAnalysis
from synoptiq.utils.constants import POS_TAGSET
from synoptiq.utils.greek import is_greek

BASE_MODEL = "bowphs/GreTa"
ADAPTER = "ainouche-abderahmane/koine-t5"

GlossLookup = Callable[[str], tuple[str, str]]


def _pos_label(code: str) -> str:
    """Map a Koine-T5 POS code (upper-cased, e.g. ``"N-"``) to a friendly label."""
    return POS_TAGSET.get(code.upper(), POS_TAGSET.get(code.upper().ljust(2, "-"), code))


class NeuralReader:
    """Koine-T5 analyser for arbitrary Koine text.

    Args:
        gloss_lookup: ``lemma -> (gloss, strong)`` callable (e.g.
            ``GoldReader(...).gloss_for``); defaults to no glossing.
        base_model: Base GreTa model id.
        adapter: Koine-T5 LoRA adapter id.
        device: Torch device string; defaults to CUDA when available else CPU.
        max_seq_len: Truncation length for a single generate call.
    """

    def __init__(
        self,
        *,
        gloss_lookup: GlossLookup | None = None,
        base_model: str = BASE_MODEL,
        adapter: str = ADAPTER,
        device: str | None = None,
        max_seq_len: int = 256,
    ) -> None:
        self.base_model = base_model
        self.adapter = adapter
        self.max_seq_len = max_seq_len
        self._gloss = gloss_lookup or (lambda _lemma: ("", ""))
        self._device = device
        self._tok: Any = None
        self._model: Any = None

    # ── model loading ─────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        """True once the model + tokenizer are in memory."""
        return self._model is not None

    def load(self) -> NeuralReader:
        """Load GreTa + the Koine-T5 adapter (idempotent). Returns self for chaining."""
        if self.loaded:
            return self
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.base_model)
        tok.pad_token = "<pad>"
        tok.eos_token = "</s>"
        tok.add_special_tokens(
            {"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]}
        )
        base = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
        model = PeftModel.from_pretrained(base, self.adapter)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        self._tok, self._model, self._device = tok, model, device
        return self

    # ── generation ────────────────────────────────────────────────────────────

    def _generate(self, prefix: str, text: str) -> str:
        import torch

        inputs = self._tok(
            prefix + text, return_tensors="pt", truncation=True, max_length=self.max_seq_len
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_seq_len, num_beams=1, do_sample=False
            )
        return self._tok.decode(out[0], skip_special_tokens=True).strip()

    def _predict_aligned(self, prefix: str, words: list[str], *, upper: bool = False) -> list[str]:
        """Predict one value per word, guaranteeing 1:1 alignment.

        Tries a single batched pass first (fast); if the output token count does not
        match the input, falls back to per-word generation (correct on rare forms —
        the alignment drift that motivated this engine).
        """
        batch = self._generate(prefix, " ".join(words))
        toks = batch.upper().split() if upper else batch.split()
        if len(toks) == len(words):
            return toks
        out: list[str] = []
        for w in words:
            g = self._generate(prefix, w)
            out.append(g.upper() if upper else g)
        return out

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(self, text: str, *, max_words: int = 80) -> ReadResult:
        """Analyse arbitrary Koine text into predicted per-word records.

        Only Greek tokens are tagged; the first ``max_words`` are used. Each record's
        ``predicted`` flag is True and ``features`` is empty (Koine-T5 emits POS + lemma,
        not full morphology).

        Args:
            text: Koine Greek, whitespace-tokenised.
            max_words: Cap on the number of Greek words analysed (guards latency).

        Returns:
            A :class:`ReadResult` of predicted :class:`WordAnalysis` records.
        """
        if not self.loaded:
            self.load()
        words = [w for w in text.split() if is_greek(w)][:max_words]
        if not words:
            return ReadResult("(pasted text)", [])
        pos_codes = self._predict_aligned("pos: ", words, upper=True)
        lemmas = self._predict_aligned("lemma: ", words)
        records: list[WordAnalysis] = []
        for surface, code, lemma in zip(words, pos_codes, lemmas, strict=True):
            gloss, strong = self._gloss(lemma)
            records.append(
                WordAnalysis(
                    surface=surface,
                    lemma=lemma,
                    pos=_pos_label(code),
                    gloss=gloss,
                    strong=strong,
                    predicted=True,
                )
            )
        return ReadResult("(pasted text)", records)
