"""Zero-shot conditional-NLL asymmetry probe for copying direction.

Compression view of causal direction (Marx & Vreeken 2017, MDL cause/effect):
if A is the source of B, then the description of the pair factorizes more cheaply
in the true direction. Using a seq2seq model as the codelength, we measure

    nll_asym = NLL(B | A) - NLL(A | B)          (per-token, length-normalized)

where NLL(Y | X) is the mean cross-entropy of generating passage Y with passage X
as the encoder input. KoineFormer is a T5 (encoder-decoder), so this needs no new
model code — just the existing labels= forward path.

This is a HYPOTHESIS TEST, not a committed feature. The synoptic asymmetry features
plateau at chance (see diagnose_direction.py); this asks whether the *generative*
direction carries signal the *similarity-geometry* features miss. We do NOT assume a
sign: a threshold/logistic classifier is fit on the train split and evaluated on test
with a pericope-grouped bootstrap CI. The sign the data prefers is reported.

Runs on CPU/MPS — no Modal, no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.training.direction import DirectionDataset  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

DIRECTIONS = ["A_to_B", "B_to_A", "independent"]


def _conditional_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,     # [1, L_src]
    attention_mask: torch.Tensor,
    label_ids: torch.Tensor,     # [1, L_tgt]
    label_mask: torch.Tensor,
) -> float:
    """Mean per-token NLL of generating the target given the source.

    Padding positions in the target are set to -100 so they are ignored by the
    HuggingFace cross-entropy (which averages over non-ignored tokens, giving a
    length-normalized codelength).
    """
    labels = label_ids.clone()
    labels[label_mask == 0] = -100
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.item())


def _collect_nll_asymmetry(
    model: torch.nn.Module,
    ds: DirectionDataset,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nll_asym, labels, pericope_ids) for every sample in the dataset."""
    asym: list[float] = []
    labels: list[int] = []
    pids: list[str] = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            ids_a = s["input_ids_a"].unsqueeze(0).to(device)
            mask_a = s["attention_mask_a"].unsqueeze(0).to(device)
            ids_b = s["input_ids_b"].unsqueeze(0).to(device)
            mask_b = s["attention_mask_b"].unsqueeze(0).to(device)

            nll_b_given_a = _conditional_nll(model, ids_a, mask_a, ids_b, mask_b)
            nll_a_given_b = _conditional_nll(model, ids_b, mask_b, ids_a, mask_a)

            asym.append(nll_b_given_a - nll_a_given_b)
            labels.append(int(s["direction_label"].item()))
            pids.append(ds.samples[i]["pericope_id"])
    return np.array(asym), np.array(labels), np.array(pids, dtype=object)


def _antisymmetry_check(asym: np.ndarray, labels: np.ndarray) -> dict:
    """Sanity check: A_to_B and B_to_A samples should have opposite-sign means.

    Swap augmentation guarantees nll_asym(B,A) = -nll_asym(A,B) exactly per pair,
    so the class means must be near-negatives of each other. A large residual would
    indicate a bug in how the two directions are scored.
    """
    mean_ab = float(asym[labels == 0].mean()) if (labels == 0).any() else float("nan")
    mean_ba = float(asym[labels == 1].mean()) if (labels == 1).any() else float("nan")
    mean_ind = float(asym[labels == 2].mean()) if (labels == 2).any() else float("nan")
    return {
        "mean_nll_asym_A_to_B": round(mean_ab, 4),
        "mean_nll_asym_B_to_A": round(mean_ba, 4),
        "mean_nll_asym_independent": round(mean_ind, 4),
        "antisymmetry_residual": round(abs(mean_ab + mean_ba), 4),
    }


def main() -> None:
    """Compute NLL-asymmetry on synoptic pairs and test its direction signal."""
    parser = argparse.ArgumentParser(description="Zero-shot NLL-asymmetry direction probe")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument(
        "--no-dapt", action="store_true",
        help="Use zero-shot GreTa instead of DAPT adapters (ablation).",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

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
    if dapt_path.exists() and not args.no_dapt:
        koine.load_adapters(dapt_path)
        print("Using KoineFormer (DAPT) decoder for conditional NLL.")
    else:
        print("Using zero-shot GreTa decoder for conditional NLL.")
    koine.model.resize_token_embeddings(len(tokenizer))
    model = koine.model

    train_ds = DirectionDataset(
        corpus, tokenizer, split="train", max_length=512,
        min_aligned_tokens=5, use_scribal_noise=False,
    )
    test_ds = DirectionDataset(
        corpus, tokenizer, split="test", max_length=512,
        min_aligned_tokens=5, use_scribal_noise=False,
    )

    _LOG.info("scoring train pairs...")
    asym_train, y_train, _ = _collect_nll_asymmetry(model, train_ds, device)
    _LOG.info("scoring test pairs...")
    asym_test, y_test, pid_test = _collect_nll_asymmetry(model, test_ds, device)

    # Point-biserial correlation of the raw scalar with each binary direction label.
    correlations = {}
    for idx, name in enumerate(DIRECTIONS):
        binary = (y_test == idx).astype(float)
        if binary.std() > 0 and np.std(asym_test) > 0:
            r, p = pointbiserialr(asym_test, binary)
            correlations[name] = {"r": round(float(r), 4), "p": round(float(p), 4)}

    # Single-scalar classifier. Features: [nll_asym, |nll_asym|]. The signed term
    # separates A_to_B from B_to_A; the magnitude term separates directed from
    # independent. Sign of the learned coefficient reveals which direction the
    # model finds cheaper to reconstruct.
    def _feat(a: np.ndarray) -> np.ndarray:
        return np.stack([a, np.abs(a)], axis=1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(_feat(asym_train))
    x_test = scaler.transform(_feat(asym_test))
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_train, y_train)
    preds_test = clf.predict(x_test)

    boot = accuracy_ci(
        y_test, preds_test, groups=pid_test,
        n_resamples=args.n_resamples, seed=42,
    )

    signed_coef = float(clf.coef_[0, 0])  # coefficient on nll_asym for class A_to_B

    # Directed-only axis: among KNOWN-DIRECTED pairs (exclude independent), does the
    # sign of nll_asym predict A_to_B vs B_to_A? This isolates the directional signal
    # from the independent class the single scalar cannot place. Threshold is fixed at
    # zero (the antisymmetric mid-point), not tuned, so there is no train leakage.
    directed_test = y_test != 2
    y_dir = y_test[directed_test]
    asym_dir = asym_test[directed_test]
    pid_dir = pid_test[directed_test]
    sign_pred = np.where(asym_dir < 0, 0, 1)  # <0 -> A_to_B, >=0 -> B_to_A
    directed_boot = accuracy_ci(
        y_dir, sign_pred, groups=pid_dir, n_resamples=args.n_resamples, seed=42,
    )

    results = {
        "name": "nll_asymmetry_probe",
        "encoder": "zero-shot" if (args.no_dapt or not dapt_path.exists()) else "dapt",
        "antisymmetry_check": _antisymmetry_check(
            np.concatenate([asym_train, asym_test]),
            np.concatenate([y_train, y_test]),
        ),
        "pointbiserial_correlation": correlations,
        "classifier": {
            "train_accuracy": round(float(clf.score(x_train, y_train)), 4),
            "test_accuracy": round(boot.accuracy, 4),
            "test_ci_low": round(boot.ci_low, 4),
            "test_ci_high": round(boot.ci_high, 4),
            "n_test_pericopes": boot.n_units,
            "n_test_samples": boot.n_samples,
            "A_to_B_nll_asym_coefficient": round(signed_coef, 4),
        },
        "directed_only_sign_classifier": {
            "description": "sign(nll_asym) predicts A_to_B vs B_to_A on directed pairs",
            "accuracy": round(directed_boot.accuracy, 4),
            "ci_low": round(directed_boot.ci_low, 4),
            "ci_high": round(directed_boot.ci_high, 4),
            "n_pericopes": directed_boot.n_units,
            "n_samples": directed_boot.n_samples,
            "binary_chance": 0.5,
        },
        "chance": round(1.0 / 3.0, 4),
        "pooled_lr_gate": 0.728,
    }

    # Three honest outcomes, not a binary. The point estimate and the (unclustered)
    # correlation can be strong while the pericope-grouped CI still includes chance
    # simply because the synoptic test set has very few independent pericopes.
    directed_point = directed_boot.accuracy
    directed_confirmed = directed_boot.ci_low > 0.5
    corr_significant = any(
        v["p"] < 0.05 for v in correlations.values()
    )
    if directed_confirmed:
        verdict = (
            "NLL asymmetry carries a CONFIRMED directed signal "
            "(directed-only accuracy CI above 50%)."
        )
    elif directed_point > 0.5 and corr_significant:
        verdict = (
            "NLL asymmetry shows a PROMISING directed signal "
            f"(directed accuracy {directed_point:.0%}, significant correlation), but "
            "the pericope-grouped CI includes chance — the synoptic test set is too "
            "small to confirm. Needs the external/synthetic sets (Stages 3-4)."
        )
    else:
        verdict = "NLL asymmetry does not separate direction above chance."
    results["verdict"] = verdict

    print("\n" + "=" * 65)
    print("NLL-ASYMMETRY PROBE")
    print("=" * 65)
    print(json.dumps(results, indent=2))

    out_path = Path("outputs/direction/nll_asymmetry_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved: {out_path}")
    print(f">>> {verdict}")


if __name__ == "__main__":
    main()
