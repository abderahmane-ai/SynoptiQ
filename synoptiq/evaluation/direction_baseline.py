"""Simple baselines for direction scoring — gate check before GPU training.

Provides cheap heuristics using the same frozen KoineFormer encoder:
1. Cosine similarity between mean-pooled passage embeddings
2. Logistic regression on pooled embeddings + simple features
3. Majority-class baseline

All baselines use the same train/val/test split as the DirectionScorer
for fair comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from synoptiq.data.corpus import Corpus
from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)


def _encode_passages(
    encoder: torch.nn.Module,
    corpus: Corpus,
    tokenizer: object,
    *,
    split: str = "train",
    max_length: int = 512,
    min_aligned_tokens: int = 5,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Encode all aligned passage pairs and return pooled embeddings + labels.

    Returns:
        X: [N, 1544] — concatenated pooled_A + pooled_B + simple features
        y: [N] — direction labels (0=A→B, 1=B→A, 2=independent)
        meta: list of dicts with book_a, book_b, pericope_id
    """
    from synoptiq.training.direction import DirectionDataset

    ds = DirectionDataset(
        corpus, tokenizer,
        split=split, max_length=max_length,
        min_aligned_tokens=min_aligned_tokens,
        use_scribal_noise=False,
    )

    all_features = []
    all_labels = []
    all_meta = []

    encoder.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            input_ids_a = sample["input_ids_a"].unsqueeze(0).to(device)
            mask_a = sample["attention_mask_a"].unsqueeze(0).to(device)
            input_ids_b = sample["input_ids_b"].unsqueeze(0).to(device)
            mask_b = sample["attention_mask_b"].unsqueeze(0).to(device)

            # Encode both passages
            h_a = encoder(input_ids=input_ids_a, attention_mask=mask_a)
            h_a = h_a.last_hidden_state  # [1, L_A, 768]
            h_b = encoder(input_ids=input_ids_b, attention_mask=mask_b)
            h_b = h_b.last_hidden_state  # [1, L_B, 768]

            # Mean-pool
            pooled_a = (h_a * mask_a.unsqueeze(-1)).sum(dim=1) / (mask_a.sum(dim=1, keepdim=True) + 1e-8)
            pooled_b = (h_b * mask_b.unsqueeze(-1)).sum(dim=1) / (mask_b.sum(dim=1, keepdim=True) + 1e-8)
            pooled_a = pooled_a.squeeze(0).cpu().numpy()
            pooled_b = pooled_b.squeeze(0).cpu().numpy()

            # Simple features
            cos_sim = np.dot(pooled_a, pooled_b) / (
                np.linalg.norm(pooled_a) * np.linalg.norm(pooled_b) + 1e-8
            )
            euclidean = np.linalg.norm(pooled_a - pooled_b)
            len_ratio = mask_a.sum().item() / (mask_b.sum().item() + 1e-8)
            vocab_overlap = _vocab_overlap(sample, tokenizer)

            features = np.concatenate([
                pooled_a, pooled_b,
                np.array([cos_sim, euclidean, len_ratio, vocab_overlap]),
            ])  # 768 + 768 + 4 = 1540

            all_features.append(features)
            all_labels.append(sample["direction_label"].item())
            all_meta.append({
                "book_a": sample.get("book_a", "?"),
                "book_b": sample.get("book_b", "?"),
            })

    return np.stack(all_features), np.array(all_labels), all_meta


def _vocab_overlap(sample: dict, tokenizer: object) -> float:
    """Compute token-level vocabulary overlap between two passages."""
    ids_a = set(sample["input_ids_a"][sample["attention_mask_a"].bool()].tolist())
    ids_b = set(sample["input_ids_b"][sample["attention_mask_b"].bool()].tolist())
    # Exclude padding and special tokens
    ids_a.discard(tokenizer.pad_token_id or 0)
    ids_b.discard(tokenizer.pad_token_id or 0)
    if not ids_a or not ids_b:
        return 0.0
    intersection = len(ids_a & ids_b)
    union = len(ids_a | ids_b)
    return intersection / union if union > 0 else 0.0


def evaluate_cosine_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict:
    """Cosine similarity as a direction heuristic.

    Hypothesis: if A copies B, the mean-pooled embeddings should be
    more similar than if they're independent. We use the cosine
    similarity feature (index 1536) to predict direction.
    """
    cos_idx = 1536  # First simple feature after 768+768 pooled dims

    # Simple threshold: higher cosine similarity → more likely A→B
    # Train: find optimal threshold on train set
    best_acc = 0.0
    best_threshold = 0.5
    for threshold in np.linspace(0.0, 1.0, 101):
        preds = np.where(X_train[:, cos_idx] > threshold, 0, 2)  # 0=A→B, 2=independent
        acc = (preds == y_train).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    test_preds = np.where(X_test[:, cos_idx] > best_threshold, 0, 2)
    test_acc = (test_preds == y_test).mean()

    return {
        "name": "cosine_similarity",
        "train_accuracy": float(best_acc),
        "test_accuracy": float(test_acc),
        "threshold": float(best_threshold),
    }


def evaluate_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict:
    """Logistic regression on pooled embeddings + simple features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train_scaled, y_train)

    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    return {
        "name": "logistic_regression",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }


def evaluate_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Always predict the majority class from training set."""
    from collections import Counter
    majority = Counter(y_train).most_common(1)[0][0]
    test_acc = (y_test == majority).mean()
    return {
        "name": "majority_class",
        "test_accuracy": float(test_acc),
        "majority_class": int(majority),
    }


def run_all_baselines(
    encoder: torch.nn.Module,
    corpus: Corpus,
    tokenizer: object,
    *,
    device: str = "cpu",
) -> dict:
    """Run all baselines and return a comparison table.

    Returns dict with keys: majority, cosine, logistic, comparison_table.
    """
    _LOG.info("encoding train passages...")
    X_train, y_train, _ = _encode_passages(encoder, corpus, tokenizer, split="train", device=device)
    _LOG.info("encoding test passages...")
    X_test, y_test, _ = _encode_passages(encoder, corpus, tokenizer, split="test", device=device)

    _LOG.info(f"train: {len(X_train)} samples, test: {len(X_test)} samples")
    _LOG.info(f"train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    _LOG.info(f"test label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    results = {}

    # 1. Random baseline (theoretical)
    results["random"] = {"name": "random (3-class)", "test_accuracy": 1.0 / 3.0}

    # 2. Majority class
    results["majority"] = evaluate_majority_baseline(y_train, y_test)

    # 3. Cosine similarity heuristic
    results["cosine"] = evaluate_cosine_baseline(X_train, y_train, X_test, y_test)

    # 4. Logistic regression
    results["logistic"] = evaluate_logistic_regression(X_train, y_train, X_test, y_test)

    # Build comparison table
    lines = [
        f"{'Baseline':<30s} {'Test Acc.':>10s}",
        f"{'─'*30} {'─'*10}",
    ]
    for key in ["random", "majority", "cosine", "logistic"]:
        r = results[key]
        lines.append(f"{r['name']:<30s} {r['test_accuracy']:>9.2%}")

    results["comparison_table"] = "\n".join(lines)
    return results


# ── Standalone runner ─────────────────────────────────────────────────────────


def main() -> None:
    """Run baselines and print comparison table."""
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).parent.parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from transformers import AutoTokenizer
    from synoptiq.data.corpus import Corpus
    from synoptiq.models.koineformer import KoineFormer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet",
        processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json",
        splits_path=processed / "splits.json",
    )

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    koine = KoineFormer.from_pretrained(device=device)
    dapt_path = Path("models/koineformer/dapt/final")
    if dapt_path.exists():
        koine.load_adapters(dapt_path)
        print("Using KoineFormer (DAPT) encoder")
    else:
        print("Using zero-shot GreTa encoder")
    koine.model.resize_token_embeddings(len(tokenizer))

    encoder = koine.model.base_model.encoder
    results = run_all_baselines(encoder, corpus, tokenizer, device=device)

    print("\n" + results["comparison_table"])
    print(f"\nGate: direction scorer must beat {results['logistic']['test_accuracy']:.2%} "
          f"(logistic regression on same encoder)")


if __name__ == "__main__":
    main()
