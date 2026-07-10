"""Build redaction pairs and fusion examples from the corpus (Phase-5 M2 training data).

Turns aligned pericopes into the (source → target) string pairs the redaction
operators train on, and the (witnesses → target) examples the Fusion-in-Decoder
trains on. Text is the whitespace-joined surface forms of a book's non-punctuation
tokens within a pericope, in canonical order — the same granularity the study scores.

Pure corpus bookkeeping; no torch. Restrict to specific pericope ids (e.g. a CV
fold's train/test split) via the ``ids`` argument.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synoptiq.data.corpus import Corpus


@dataclass(frozen=True)
class RedactionExample:
    """One (source, target) training pair for a redaction operator."""

    pericope_id: str
    source_book: str
    target_book: str
    source_text: str
    target_text: str


@dataclass(frozen=True)
class FusionExample:
    """One (witnesses → target) training example for the Fusion-in-Decoder."""

    pericope_id: str
    witness_texts: dict[str, str]  # book → text, e.g. {"Matthew": ..., "Luke": ...}
    target_text: str


def pericope_text(corpus: Corpus, book: str, pericope_id: str) -> str:
    """Whitespace-joined surface forms of a book's tokens in one pericope, in order."""
    rows: list[tuple[int, int, int, str]] = []
    for tok in corpus.get_tokens(book, pericope_id=pericope_id, exclude_punctuation=True):  # type: ignore[arg-type]
        rows.append(
            (int(tok["chapter"]), int(tok["verse"]), int(tok["position"]), str(tok["text"]))
        )
    rows.sort()
    return " ".join(text for *_, text in rows)


def _triple_pericope_ids(corpus: Corpus, ids: set[str] | None) -> list[str]:
    from synoptiq.data.study_design import full_triples

    triples = [u.pericope_id for u in full_triples(corpus)]
    if ids is not None:
        triples = [p for p in triples if p in ids]
    return triples


def redaction_pairs(
    corpus: Corpus,
    *,
    source_book: str,
    target_book: str,
    ids: set[str] | None = None,
) -> list[RedactionExample]:
    """(source_book → target_book) pairs over full triples (optionally id-restricted).

    Used to train the four operators: R_Lk = (Mark, Luke), R_Mt = (Mark, Matthew),
    G_Mt = (Matthew, Mark), G_Lk = (Luke, Mark).
    """
    out: list[RedactionExample] = []
    for pid in _triple_pericope_ids(corpus, ids):
        src = pericope_text(corpus, source_book, pid)
        tgt = pericope_text(corpus, target_book, pid)
        if src and tgt:
            out.append(
                RedactionExample(
                    pericope_id=pid,
                    source_book=source_book,
                    target_book=target_book,
                    source_text=src,
                    target_text=tgt,
                )
            )
    return out


def fusion_examples(
    corpus: Corpus,
    *,
    witnesses: tuple[str, ...] = ("Matthew", "Luke"),
    target: str = "Mark",
    ids: set[str] | None = None,
) -> list[FusionExample]:
    """(witnesses → target) examples over full triples (optionally id-restricted).

    Track A trains on ``(Matthew, Luke) → Mark`` where ground truth exists, then
    the trained model is applied to the double tradition to emit proto-Q.
    """
    out: list[FusionExample] = []
    for pid in _triple_pericope_ids(corpus, ids):
        texts = {b: pericope_text(corpus, b, pid) for b in witnesses}
        tgt = pericope_text(corpus, target, pid)
        if tgt and all(texts.values()):
            out.append(FusionExample(pericope_id=pid, witness_texts=texts, target_text=tgt))
    return out


def source_dropout_variants(example: FusionExample) -> list[FusionExample]:
    """All non-empty witness subsets of a fusion example (for source-dropout training).

    The capacity-fairness rule (§3) requires one-witness and two-witness conditionals
    to come from the *same* weights; training on every witness subset achieves that.
    Returns the full example plus each single-witness variant.
    """
    books = list(example.witness_texts)
    variants: list[FusionExample] = [example]
    if len(books) > 1:
        for b in books:
            variants.append(
                FusionExample(
                    pericope_id=example.pericope_id,
                    witness_texts={b: example.witness_texts[b]},
                    target_text=example.target_text,
                )
            )
    return variants


def grouped_ids(examples: list[RedactionExample] | list[FusionExample]) -> list[str]:
    """Pericope id per example — the grouping key for cluster-bootstrap CIs."""
    return [e.pericope_id for e in examples]


def per_pericope_counts(corpus: Corpus, ids: set[str] | None = None) -> dict[str, int]:
    """Target (Mark) token count per full-triple pericope — reconstruction weights."""
    counts: dict[str, int] = defaultdict(int)
    for pid in _triple_pericope_ids(corpus, ids):
        counts[pid] = len(pericope_text(corpus, "Mark", pid).split())
    return dict(counts)
