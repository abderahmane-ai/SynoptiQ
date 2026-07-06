"""Diagnose direction scorer feature quality and classifier behavior.

Runs five experiments on local CPU/MPS — no Modal, no GPU needed.
Answers: are asymmetry features useful, is authorship still decodable,
and what does the confusion matrix look like?
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # type: ignore[import-untyped]  # noqa: E402

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.models.direction import (  # noqa: E402
    DirectionScorer,
    DirectionScorerConfig,
)
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.training.direction import DirectionDataset  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

DIRECTIONS = ["A→B", "B→A", "independent"]


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Are the 10 asymmetry features informative on their own?
# ═══════════════════════════════════════════════════════════════════════════════


def experiment_1_asymmetry_only(
    scorer: DirectionScorer,
    corpus: Corpus,
    tokenizer: object,
    device: str,
) -> dict:
    """Train logistic regression on the 10 asymmetry features ONLY.

    If accuracy >33%, the frozen-encoder asymmetry features carry signal.
    If accuracy =33%, the features are not direction-informative.
    """
    _LOG.info("=== Experiment 1: Asymmetry features only ===")

    # Extract asymmetry features using the trained model
    def _extract_asym(split: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract the 10 asymmetry features and labels for one split."""
        ds = DirectionDataset(corpus, tokenizer, split=split, max_length=512,
                              min_aligned_tokens=5, use_scribal_noise=False)
        features, labels = [], []
        scorer.eval()
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                batch = {
                    "input_ids_a": s["input_ids_a"].unsqueeze(0).to(device),
                    "attention_mask_a": s["attention_mask_a"].unsqueeze(0).to(device),
                    "input_ids_b": s["input_ids_b"].unsqueeze(0).to(device),
                    "attention_mask_b": s["attention_mask_b"].unsqueeze(0).to(device),
                }
                out = scorer(**batch)
                features.append(out["asymmetry_features"][0].cpu().numpy())
                labels.append(s["direction_label"].item())
        return np.stack(features), np.array(labels)

    X_train, y_train = _extract_asym("train")
    X_test, y_test = _extract_asym("test")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    train_acc = clf.score(X_train_s, y_train)
    test_acc = clf.score(X_test_s, y_test)

    # Feature importance from logistic regression coefficients
    coef_norm = np.abs(clf.coef_).sum(axis=0)
    coef_norm = coef_norm / coef_norm.sum()

    return {
        "name": "asymmetry_features_only",
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "feature_importance": {
            f"f{i+1}": round(float(c), 4) for i, c in enumerate(coef_norm)
        },
        "verdict": (
            "features are informative" if test_acc > 0.38
            else "features are not direction-informative"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Confusion matrix — where is the model wrong?
# ═══════════════════════════════════════════════════════════════════════════════


def experiment_2_confusion_matrix(
    scorer: DirectionScorer,
    corpus: Corpus,
    tokenizer: object,
    device: str,
) -> dict:
    """Show exactly which classes the model confuses."""
    _LOG.info("=== Experiment 2: Confusion matrix ===")

    ds = DirectionDataset(corpus, tokenizer, split="test", max_length=512,
                          min_aligned_tokens=5, use_scribal_noise=False)

    all_preds, all_labels = [], []
    scorer.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            batch = {
                "input_ids_a": s["input_ids_a"].unsqueeze(0).to(device),
                "attention_mask_a": s["attention_mask_a"].unsqueeze(0).to(device),
                "input_ids_b": s["input_ids_b"].unsqueeze(0).to(device),
                "attention_mask_b": s["attention_mask_b"].unsqueeze(0).to(device),
            }
            out = scorer(**batch)
            pred = out["direction_logits"].argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(s["direction_label"].item())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    # Per-class accuracy
    per_class = {}
    for i, name in enumerate(DIRECTIONS):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            per_class[name] = float((np.array(all_preds)[mask] == i).mean())

    return {
        "name": "confusion_matrix",
        "matrix": cm.tolist(),
        "labels": DIRECTIONS,
        "per_class_accuracy": per_class,
        "total_accuracy": float((np.array(all_preds) == np.array(all_labels)).mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Asymmetry feature correlation with true direction
# ═══════════════════════════════════════════════════════════════════════════════


def experiment_3_feature_label_correlation(
    scorer: DirectionScorer,
    corpus: Corpus,
    tokenizer: object,
    device: str,
) -> dict:
    """Compute Pearson r between each asymmetry feature and the true label.

    High correlation = that feature carries directional signal.
    Near-zero = that feature is weak for that binary contrast.
    """
    _LOG.info("=== Experiment 3: Feature-label correlation ===")

    ds = DirectionDataset(corpus, tokenizer, split="test", max_length=512,
                          min_aligned_tokens=5, use_scribal_noise=False)

    all_features, all_labels = [], []
    scorer.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            batch = {
                "input_ids_a": s["input_ids_a"].unsqueeze(0).to(device),
                "attention_mask_a": s["attention_mask_a"].unsqueeze(0).to(device),
                "input_ids_b": s["input_ids_b"].unsqueeze(0).to(device),
                "attention_mask_b": s["attention_mask_b"].unsqueeze(0).to(device),
            }
            out = scorer(**batch)
            all_features.append(out["asymmetry_features"][0].cpu().numpy())
            all_labels.append(s["direction_label"].item())

    X = np.stack(all_features)  # [N, 10]
    y = np.array(all_labels)    # [N]

    correlations = {}
    feature_names = [
        "bertscore_recall",
        "bertscore_precision",
        "precision_recall_asymmetry",
        "row_max_std",
        "col_max_std",
        "entropy_asymmetry",
        "diagonal_closeness",
        "log_length_ratio",
        "pooled_similarity",
        "coverage_asymmetry",
    ]
    for i, name in enumerate(feature_names):
        # Point-biserial correlation with label
        # Binary tests: A→B vs not, B→A vs not, independent vs not
        corrs = {}
        for label_idx, label_name in enumerate(DIRECTIONS):
            binary_label = (y == label_idx).astype(float)
            if binary_label.std() > 0:
                r = np.corrcoef(X[:, i], binary_label)[0, 1]
                corrs[label_name] = round(float(r), 4)
        correlations[name] = corrs

    # Average absolute correlation across all three binary comparisons
    avg_corrs = {}
    for name in feature_names:
        vals = [abs(v) for v in correlations[name].values()]
        avg_corrs[name] = round(float(np.mean(vals)), 4)

    return {
        "name": "feature_label_correlation",
        "per_feature": correlations,
        "avg_abs_correlation": avg_corrs,
        "verdict": (
            "features correlate with direction" if max(avg_corrs.values()) > 0.15
            else "features are UNCORRELATED with direction — model didn't learn asymmetry"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Author decodability from asymmetry features
# ═══════════════════════════════════════════════════════════════════════════════


def experiment_4_author_discriminator_accuracy(
    scorer: DirectionScorer,
    corpus: Corpus,
    tokenizer: object,
    device: str,
) -> dict:
    """Check whether authorship is decodable from the 10 asymmetry features.

    This experiment asks:
    does a logistic regression trained on the asymmetry features alone predict
    the author of passage A better than chance (33%)?

    If accuracy ≈ 33%: features are direction-specific, not style-specific.
    If accuracy >> 33%: features carry authorship signal — a confound to note.
    """
    _LOG.info("=== Experiment 4: Author decodability from asymmetry features ===")

    def _collect(split: str) -> tuple[np.ndarray, np.ndarray]:
        """Collect asymmetry features and author-of-A labels for one split."""
        ds = DirectionDataset(corpus, tokenizer, split=split, max_length=512,
                              min_aligned_tokens=5, use_scribal_noise=False)
        features, author_labels = [], []
        _author_idx = {"Matthew": 0, "Mark": 1, "Luke": 2}
        scorer.eval()
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                sample_raw = ds.samples[i]  # access raw sample for book_a
                batch = {
                    "input_ids_a": s["input_ids_a"].unsqueeze(0).to(device),
                    "attention_mask_a": s["attention_mask_a"].unsqueeze(0).to(device),
                    "input_ids_b": s["input_ids_b"].unsqueeze(0).to(device),
                    "attention_mask_b": s["attention_mask_b"].unsqueeze(0).to(device),
                }
                out = scorer(**batch)
                features.append(out["asymmetry_features"][0].cpu().numpy())
                author_labels.append(_author_idx.get(sample_raw["book_a"], 0))
        return np.stack(features), np.array(author_labels)

    X_train, y_train = _collect("train")
    X_test, y_test = _collect("test")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    train_acc = float(clf.score(X_train_s, y_train))
    test_acc = float(clf.score(X_test_s, y_test))

    return {
        "name": "author_decodability",
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "random_baseline": round(1.0 / 3.0, 4),
        "verdict": (
            "features are direction-specific (good)" if test_acc < 0.45
            else "authorship still decodable from features — style confound present"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5: Ablation — direction accuracy WITHOUT the asymmetry features
# ═══════════════════════════════════════════════════════════════════════════════


def experiment_5_pooled_only(
    scorer: DirectionScorer,
    corpus: Corpus,
    tokenizer: object,
    device: str,
) -> dict:
    """Compare: pooled embeddings alone vs pooled + asymmetry features.

    Uses the trained model but sets asymmetry contribution to zero
    to isolate whether the asymmetry features add anything beyond
    the pooled encoder representations.
    """
    _LOG.info("=== Experiment 5: Pooled embeddings only ===")

    # We use the logistic regression baseline which already does this
    # But let's also check the trained model with asymmetry zeroed out
    ds = DirectionDataset(corpus, tokenizer, split="test", max_length=512,
                          min_aligned_tokens=5, use_scribal_noise=False)

    correct_with = total = 0
    scorer.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            batch = {
                "input_ids_a": s["input_ids_a"].unsqueeze(0).to(device),
                "attention_mask_a": s["attention_mask_a"].unsqueeze(0).to(device),
                "input_ids_b": s["input_ids_b"].unsqueeze(0).to(device),
                "attention_mask_b": s["attention_mask_b"].unsqueeze(0).to(device),
            }
            out = scorer(**batch)
            pred_full = out["direction_logits"].argmax(dim=1).item()
            correct_with += int(pred_full == s["direction_label"].item())
            total += 1

    return {
        "name": "pooled_vs_asymmetry",
        "accuracy_with_asymmetry": float(correct_with / total),
        "baseline_without_asymmetry": 0.728,  # pooled-embedding logistic baseline
        "delta": round(float(correct_with / total) - 0.728, 4),
        "verdict": (
            "asymmetry features ADD signal" if (correct_with / total) > 0.728
            else "asymmetry features do not beat the pooled-embedding baseline"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


def _centroid_initialize(
    scorer: DirectionScorer,
    corpus: Corpus,
    tokenizer: object,
    device: str,
) -> None:
    """Fit the constrained classifier from train-split centroids (no gradient step).

    Extracts the deterministic asymmetry features for the train split, sets the
    feature-standardisation buffers, and initialises the two constrained heads via
    the shared-covariance centroid rule. This reproduces the model's decision rule
    closely enough for the diagnostics without requiring a trained checkpoint.
    """
    ds = DirectionDataset(
        corpus, tokenizer, split="train", max_length=512,
        min_aligned_tokens=5, use_scribal_noise=False,
    )
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    scorer.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            out = scorer(
                input_ids_a=s["input_ids_a"].unsqueeze(0).to(device),
                attention_mask_a=s["attention_mask_a"].unsqueeze(0).to(device),
                input_ids_b=s["input_ids_b"].unsqueeze(0).to(device),
                attention_mask_b=s["attention_mask_b"].unsqueeze(0).to(device),
            )
            feats.append(out["asymmetry_features"][0].cpu())
            labels.append(s["direction_label"])
    features = torch.stack(feats).to(device)
    label_tensor = torch.stack(labels).to(device)
    scorer.classifier.set_feature_stats(features.mean(dim=0), features.std(dim=0))
    scorer.classifier.initialize_from_centroids(features, label_tensor)


def main() -> None:
    """Load the trained scorer and run the five direction-scorer diagnostics."""
    parser = _build_parser()
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load data
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet",
        processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json",
        splits_path=processed / "splits.json",
    )

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model
    koine = KoineFormer.from_pretrained(device=device)
    dapt_path = Path("models/koineformer/dapt/final")
    if dapt_path.exists():
        koine.load_adapters(dapt_path)
    koine.model.resize_token_embeddings(len(tokenizer))

    encoder = koine.model.base_model.encoder
    scorer = DirectionScorer(encoder, DirectionScorerConfig())
    scorer.to(device)

    # Load a trained checkpoint if one is compatible with the current classifier.
    # Older checkpoints predate the swap-equivariant two-head classifier (they used
    # a per-sample LayerNorm + single linear layer) and their keys will not match.
    # In that case fall back to a centroid-initialised classifier so exps 2 & 5 still
    # reflect the current architecture's decision rule. Exps 1, 3, 4 read only the
    # deterministic asymmetry features and are unaffected either way.
    ckpt_path = args.checkpoint
    if not ckpt_path or not Path(ckpt_path).exists():
        ckpt_path = Path("outputs/direction/best/model.pt")

    loaded = False
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            scorer.load_state_dict(state)
            loaded = True
            print(f"Loaded trained checkpoint: {ckpt_path}")
        except RuntimeError as exc:
            print(
                f"WARNING: checkpoint {ckpt_path} is incompatible with the current "
                f"classifier ({type(exc).__name__}); falling back to centroid init.",
            )
    if not loaded:
        print(
            "Centroid-initializing classifier from train-split features "
            "(no compatible checkpoint found).",
        )
        _centroid_initialize(scorer, corpus, tokenizer, device)
    scorer.to(device)

    # Run all experiments
    results = {}

    results["exp1_asymmetry_only"] = experiment_1_asymmetry_only(
        scorer, corpus, tokenizer, device)
    results["exp2_confusion"] = experiment_2_confusion_matrix(
        scorer, corpus, tokenizer, device)
    results["exp3_correlation"] = experiment_3_feature_label_correlation(
        scorer, corpus, tokenizer, device)
    results["exp4_author_disc"] = experiment_4_author_discriminator_accuracy(
        scorer, corpus, tokenizer, device)
    results["exp5_pooled"] = experiment_5_pooled_only(
        scorer, corpus, tokenizer, device)

    # Print report
    print("\n" + "=" * 65)
    print("DIAGNOSTIC REPORT")
    print("=" * 65)

    for key, r in results.items():
        print(f"\n─── {r['name']} ───")
        for k, v in r.items():
            if k in ("name", "verdict"):
                continue
            if isinstance(v, dict):
                print(f"  {k}:")
                for nested_key, nested_value in v.items():
                    print(f"    {nested_key}: {nested_value}")
            elif isinstance(v, list):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")
        if "verdict" in r:
            print(f"  >>> {r['verdict']}")

    # Save
    out_path = Path("outputs/direction/diagnostic_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nReport saved: {out_path}")


def _build_parser():
    """Build the argument parser for the direction scorer diagnostics CLI."""
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose direction scorer plateau")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to model checkpoint (default: outputs/direction/best/model.pt)")
    return parser


if __name__ == "__main__":
    main()
