"""Evaluate the NLL-asymmetry direction probe on the external known-direction set.

This is the cheat-proof check for Stage 2's finding. On the synoptics, source-in-A
pairs (A_to_B) had NEGATIVE nll_asym = NLL(B|A) - NLL(A|B). If the Jude -> 2 Peter
pairs (also source-in-A, but no synoptic author) show the SAME negative sign, the
probe transfers across corpora and is measuring copying direction, not Markan style.

Runs on CPU/MPS — no Modal, no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from synoptiq.data.external_pairs import load_external_pairs  # noqa: E402
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _conditional_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
) -> float:
    """Mean per-token NLL of generating the target given the source (pad -> -100)."""
    labels = label_ids.clone()
    labels[label_mask == 0] = -100
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.item())


def main() -> None:
    """Score NLL asymmetry on the external pairs and report sign transfer."""
    parser = argparse.ArgumentParser(description="External known-direction NLL probe")
    parser.add_argument(
        "--pairs", type=Path,
        default=Path("data/external/known_direction_pairs.json"),
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--no-dapt", action="store_true")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    koine = KoineFormer.from_pretrained(device=device)
    dapt_path = Path("models/koineformer/dapt/final")
    use_dapt = dapt_path.exists() and not args.no_dapt
    if use_dapt:
        koine.load_adapters(dapt_path)
        print("Using KoineFormer (DAPT) decoder.")
    else:
        print("Using zero-shot GreTa decoder.")
    koine.model.resize_token_embeddings(len(tokenizer))
    model = koine.model

    samples = load_external_pairs(args.pairs, tokenizer, max_length=512, augment_swap=True)

    asym: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    model.eval()
    with torch.no_grad():
        for s in samples:
            ids_a = s["input_ids_a"].unsqueeze(0).to(device)
            mask_a = s["attention_mask_a"].unsqueeze(0).to(device)
            ids_b = s["input_ids_b"].unsqueeze(0).to(device)
            mask_b = s["attention_mask_b"].unsqueeze(0).to(device)
            nll_b_a = _conditional_nll(model, ids_a, mask_a, ids_b, mask_b)
            nll_a_b = _conditional_nll(model, ids_b, mask_b, ids_a, mask_a)
            asym.append(nll_b_a - nll_a_b)
            labels.append(int(s["direction_label"].item()))
            groups.append(s["group"])

    asym_arr = np.array(asym)
    y = np.array(labels)
    grp = np.array(groups, dtype=object)

    # All external pairs are directed (no independent). Predict A_to_B if nll_asym<0
    # (the sign the synoptics preferred), B_to_A otherwise. Fixed threshold: no fit.
    directed = y != 2
    y_dir = y[directed]
    sign_pred = np.where(asym_arr[directed] < 0, 0, 1)
    boot = accuracy_ci(
        y_dir, sign_pred, groups=grp[directed], n_resamples=args.n_resamples, seed=42,
    )

    mean_ab = float(asym_arr[y == 0].mean()) if (y == 0).any() else float("nan")
    mean_ba = float(asym_arr[y == 1].mean()) if (y == 1).any() else float("nan")

    sign_transfers = mean_ab < 0 < mean_ba  # same convention as synoptics
    results = {
        "name": "external_nll_direction",
        "pairs_file": str(args.pairs),
        "encoder": "dapt" if use_dapt else "zero-shot",
        "n_samples": len(samples),
        "mean_nll_asym_source_in_A": round(mean_ab, 4),
        "mean_nll_asym_source_in_B": round(mean_ba, 4),
        "sign_convention_matches_synoptics": bool(sign_transfers),
        "directed_sign_accuracy": round(boot.accuracy, 4),
        "ci_low": round(boot.ci_low, 4),
        "ci_high": round(boot.ci_high, 4),
        "n_groups": boot.n_units,
        "binary_chance": 0.5,
    }
    results["verdict"] = (
        "Sign convention TRANSFERS to a non-synoptic corpus — evidence the probe "
        "tracks copying direction, not Markan style."
        if sign_transfers
        else "Sign convention does NOT transfer — probe signal may be corpus-specific."
    )

    print("\n" + "=" * 65)
    print("EXTERNAL KNOWN-DIRECTION NLL PROBE")
    print("=" * 65)
    print(json.dumps(results, indent=2))

    out_path = Path("outputs/direction/external_nll_direction.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved: {out_path}")
    print(f">>> {results['verdict']}")


if __name__ == "__main__":
    main()
