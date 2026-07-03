"""Evaluation benchmarks for KoineFormer.

Evaluates frozen encoder representations by training linear probes
on top of per-token hidden states for POS tagging and lemmatisation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# GreTa's SentencePiece tokenizer produces subwords prefixed with "▁"
# (U+2581, LOWER ONE EIGHTH BLOCK).  Tokens beginning with this mark
# the start of a new original word.
_SUBWORD_PREFIX = "▁"


@dataclass
class BenchmarkResult:
    """Results for a single model on a single task."""
    model_name: str
    task: str
    metric_name: str
    value: float
    n_samples: int


def evaluate_pos_tagging(
    model: Any,
    corpus: Any,
    tokenizer: Any,
    *,
    split: str = "test",
    max_verses: int = 500,
    device: str = "cpu",
    probe_epochs: int = 10,
) -> BenchmarkResult:
    """Evaluate POS accuracy via linear probe on per-token encoder outputs.

    Encodes verses with the frozen encoder, aligns subword hidden states
    back to original words using SentencePiece word boundaries, then
    trains a linear probe on the training split and evaluates on *split*.

    Args:
        model: KoineFormer instance.
        corpus: SynoptiQ Corpus.
        tokenizer: GreTa tokenizer (with pad token set).
        split: Corpus split to evaluate on (``"test"``).
        max_verses: Maximum verses to sample from each split (keeps eval fast).
        device: Torch device.
        probe_epochs: SGD epochs for the linear probe.

    Returns:
        BenchmarkResult with accuracy.
    """
    # ── Build POS vocabulary from training data ────────────────────────
    pos_to_idx: dict[str, int] = {}
    for token in corpus.get_tokens(split="train"):
        p = token.get("pos", "")
        if p and p not in pos_to_idx:
            pos_to_idx[p] = len(pos_to_idx)
    n_classes = len(pos_to_idx)
    _LOG.info("POS probe setup", extra={"n_classes": n_classes})

    # ── Extract per-token features ─────────────────────────────────────
    encoder = model.model.base_model.encoder
    encoder.eval()

    def _extract_features(data_split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (per_word_hidden, pos_labels) for verses in *data_split*."""
        all_hidden: list[torch.Tensor] = []
        all_labels: list[int] = []
        n_verses = 0

        for book in ("Matthew", "Mark", "Luke"):
            if max_verses and n_verses >= max_verses // 3:
                break
            tokens = corpus.get_tokens(book=book, split=data_split)
            verses: dict[tuple[int, int], list[Any]] = defaultdict(list)
            for t in tokens:
                if n_verses >= max_verses:
                    break
                key = (int(t["chapter"]), int(t["verse"]))
                verses[key].append(t)

            for _vref, verse_tokens in verses.items():
                if max_verses and n_verses >= max_verses:
                    break
                if len(verse_tokens) < 3:
                    continue

                n_verses += 1
                text = " ".join(str(t["text"]) for t in verse_tokens)
                encoded = tokenizer(
                    text, max_length=128, truncation=True,
                    padding="max_length", return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[0]  # [S, 768]

                # Align subwords → original words using SentencePiece prefix.
                # Decode each token; tokens starting with "▁" begin a new word.
                subword_texts = tokenizer.convert_ids_to_tokens(input_ids[0])
                word_boundaries: list[int] = []  # start indices of each original word
                for si, st in enumerate(subword_texts):
                    if st.startswith(_SUBWORD_PREFIX) or si == 0 or st in ("[PAD]",):
                        word_boundaries.append(si)

                # Map original words to mean-pooled subword hidden states.
                for wi, word_start in enumerate(word_boundaries):
                    if wi >= len(verse_tokens):
                        break
                    word_end = word_boundaries[wi + 1] if wi + 1 < len(word_boundaries) else len(subword_texts)
                    # Skip padding tokens
                    actual_end = min(word_end, attention_mask.sum().item())
                    if word_start >= actual_end:
                        continue

                    word_hidden = hidden[word_start:actual_end].mean(dim=0)
                    pos_tag = verse_tokens[wi].get("pos", "")
                    if pos_tag in pos_to_idx:
                        all_hidden.append(word_hidden.cpu())
                        all_labels.append(pos_to_idx[pos_tag])

        if not all_hidden:
            _LOG.warning(f"no features extracted for split={data_split}")
            return torch.zeros(0, 768), torch.zeros(0, dtype=torch.long)

        X = torch.stack(all_hidden)
        y = torch.tensor(all_labels, dtype=torch.long)
        _LOG.info(f"extracted {len(X)} tokens from {n_verses} verses", extra={"split": data_split})
        return X, y

    _LOG.info("extracting train features")
    X_train, y_train = _extract_features("train")
    _LOG.info(f"extracting {split} features")
    X_test, y_test = _extract_features(split)

    if len(X_train) == 0 or len(X_test) == 0:
        _LOG.error("no features extracted — check tokenizer/corpus alignment")
        return BenchmarkResult(model_name=model.model_id, task="POS", metric_name="accuracy",
                               value=0.0, n_samples=0)

    hidden_dim = X_train.shape[1]

    # ── Train linear probe ─────────────────────────────────────────────
    probe = torch.nn.Linear(hidden_dim, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    X_train_dev = X_train.to(device)
    y_train_dev = y_train.to(device)
    batch_size = 128

    probe.train()
    for epoch in range(probe_epochs):
        perm = torch.randperm(len(X_train_dev))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train_dev), batch_size):
            idx = perm[i:i + batch_size]
            batch_x, batch_y = X_train_dev[idx], y_train_dev[idx]
            opt.zero_grad()
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 3 == 0:
            _LOG.info(f"  probe epoch {epoch + 1}/{probe_epochs}: loss={total_loss / max(n_batches, 1):.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────
    probe.eval()
    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)
    with torch.no_grad():
        logits = probe(X_test_dev)
        preds = logits.argmax(dim=1)
        correct = (preds == y_test_dev).sum().item()

    accuracy = correct / max(len(y_test_dev), 1)
    _LOG.info(f"probe POS accuracy ({split}): {accuracy:.2%} ({correct}/{len(y_test_dev)} tokens)")

    return BenchmarkResult(
        model_name=model.model_id,
        task="POS tagging (linear probe)",
        metric_name="accuracy",
        value=accuracy,
        n_samples=len(y_test_dev),
    )


def evaluate_lemmatization(
    model: Any,
    corpus: Any,
    tokenizer: Any,
    *,
    split: str = "test",
    max_verses: int = 500,
    device: str = "cpu",
    probe_epochs: int = 15,
) -> BenchmarkResult:
    """Evaluate lemma accuracy via linear probe on per-token encoder outputs.

    Same extraction protocol as POS tagging, but predicts the dictionary
    headword (lemma) rather than the part of speech.  Lemmatisation is a
    harder task (~2,700 classes vs 13 for POS), making it a stronger test
    of encoder quality.
    """
    # Build lemma vocabulary from training data
    lemma_to_idx: dict[str, int] = {}
    for token in corpus.get_tokens(split="train"):
        l = token.get("lemma", "")
        if l and l not in lemma_to_idx:
            lemma_to_idx[l] = len(lemma_to_idx)
    n_classes = len(lemma_to_idx)
    _LOG.info("lemma probe setup", extra={"n_classes": n_classes})

    encoder = model.model.base_model.encoder
    encoder.eval()

    def _extract(data_split: str) -> tuple[torch.Tensor, torch.Tensor]:
        all_hidden: list[torch.Tensor] = []
        all_labels: list[int] = []
        n_verses = 0

        for book in ("Matthew", "Mark", "Luke"):
            if max_verses and n_verses >= max_verses // 3:
                break
            tokens = corpus.get_tokens(book=book, split=data_split)
            verses: dict[tuple[int, int], list[Any]] = defaultdict(list)
            for t in tokens:
                key = (int(t["chapter"]), int(t["verse"]))
                verses[key].append(t)

            for _vref, verse_tokens in verses.items():
                if max_verses and n_verses >= max_verses:
                    break
                if len(verse_tokens) < 3:
                    continue
                n_verses += 1

                text = " ".join(str(t["text"]) for t in verse_tokens)
                encoded = tokenizer(text, max_length=128, truncation=True,
                                    padding="max_length", return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[0]

                subword_texts = tokenizer.convert_ids_to_tokens(input_ids[0])
                word_boundaries = [si for si, st in enumerate(subword_texts)
                                   if st.startswith(_SUBWORD_PREFIX) or si == 0 or st == "[PAD]"]

                for wi, word_start in enumerate(word_boundaries):
                    if wi >= len(verse_tokens):
                        break
                    word_end = word_boundaries[wi + 1] if wi + 1 < len(word_boundaries) else len(subword_texts)
                    actual_end = min(word_end, attention_mask.sum().item())
                    if word_start >= actual_end:
                        continue
                    word_hidden = hidden[word_start:actual_end].mean(dim=0)
                    lemma = verse_tokens[wi].get("lemma", "")
                    if lemma in lemma_to_idx:
                        all_hidden.append(word_hidden.cpu())
                        all_labels.append(lemma_to_idx[lemma])

        if not all_hidden:
            return torch.zeros(0, 768), torch.zeros(0, dtype=torch.long)
        X = torch.stack(all_hidden)
        y = torch.tensor(all_labels, dtype=torch.long)
        _LOG.info(f"extracted {len(X)} lemma tokens from {n_verses} verses", extra={"split": data_split})
        return X, y

    _LOG.info("extracting train features")
    X_train, y_train = _extract("train")
    _LOG.info(f"extracting {split} features")
    X_test, y_test = _extract(split)

    if len(X_train) == 0 or len(X_test) == 0:
        return BenchmarkResult(model_name=model.model_id, task="Lemma", metric_name="accuracy",
                               value=0.0, n_samples=0)

    hidden_dim = X_train.shape[1]
    probe = torch.nn.Linear(hidden_dim, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    X_train_dev = X_train.to(device)
    y_train_dev = y_train.to(device)
    batch_size = 128

    probe.train()
    for epoch in range(probe_epochs):
        perm = torch.randperm(len(X_train_dev))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train_dev), batch_size):
            idx = perm[i:i + batch_size]
            batch_x, batch_y = X_train_dev[idx], y_train_dev[idx]
            opt.zero_grad()
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0:
            _LOG.info(f"  probe epoch {epoch + 1}/{probe_epochs}: loss={total_loss / max(n_batches, 1):.4f}")

    probe.eval()
    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)
    with torch.no_grad():
        logits = probe(X_test_dev)
        preds = logits.argmax(dim=1)
        correct = (preds == y_test_dev).sum().item()

    accuracy = correct / max(len(y_test_dev), 1)
    _LOG.info(f"probe lemma accuracy ({split}): {accuracy:.2%} ({correct}/{len(y_test_dev)} tokens)")

    return BenchmarkResult(
        model_name=model.model_id,
        task="Lemmatisation (linear probe)",
        metric_name="accuracy",
        value=accuracy,
        n_samples=len(y_test_dev),
    )
