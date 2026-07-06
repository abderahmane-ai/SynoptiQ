"""Adversarial validation of the conditional-NLL direction signal (synoptic split).

Decides the direction component's design by measurement, not assumption. For every
synoptic test pair it computes the four NLL codelengths and compares three
signed direction scores (positive => A is the source):

  1. conditional_asym  — naive per-token NLL(A|B) - NLL(B|A)  (the Stage-2 probe)
  2. mdl               — length-fair [L(B->A) - L(A->B)] with marginal terms
  3. resid_conditional — conditional_asym with log length ratio regressed out

For each it reports directed-only sign accuracy (pericope/block-grouped bootstrap CI)
and, critically, the PARTIAL correlation with direction after controlling for length —
so we can see whether a score is measuring copying direction or merely "longer text".
Run with and without --no-dapt for the encoder ablation.

CPU/MPS, no Modal, no training.
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

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.evaluation.nll_direction import (  # noqa: E402
    DirectionCodelengths,
    _total_nll,
    make_empty_source,
)
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.training.direction import DirectionDataset  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


class _CachedScorer:
    """Scores pairs while caching conditional and marginal NLLs by passage bytes.

    Swap augmentation and triple-tradition combinatorics make the same passages
    recur across samples; caching cuts the forward passes roughly in half.
    """

    def __init__(self, model: torch.nn.Module, empty_source: tuple, device: str) -> None:
        self.model = model
        self.empty = empty_source
        self.device = device
        self._cond: dict[tuple, tuple[float, int]] = {}
        self._marg: dict[bytes, tuple[float, int]] = {}

    @staticmethod
    def _key(ids: torch.Tensor, mask: torch.Tensor) -> bytes:
        real = ids[0][mask[0].bool()]
        return real.cpu().numpy().tobytes()

    def score(self, ids_a, mask_a, ids_b, mask_b) -> DirectionCodelengths:  # noqa: ANN001
        ka, kb = self._key(ids_a, mask_a), self._key(ids_b, mask_b)

        def cond(src_ids, src_mask, tgt_ids, tgt_mask, sk, tk):  # noqa: ANN001
            key = (sk, tk)
            if key not in self._cond:
                self._cond[key] = _total_nll(self.model, src_ids, src_mask, tgt_ids, tgt_mask)
            return self._cond[key]

        def marg(tgt_ids, tgt_mask, tk):  # noqa: ANN001
            if tk not in self._marg:
                self._marg[tk] = _total_nll(self.model, *self.empty, tgt_ids, tgt_mask)
            return self._marg[tk]

        nll_b_given_a, n_b = cond(ids_a, mask_a, ids_b, mask_b, ka, kb)
        nll_a_given_b, n_a = cond(ids_b, mask_b, ids_a, mask_a, kb, ka)
        nll_b_marg, _ = marg(ids_b, mask_b, kb)
        nll_a_marg, _ = marg(ids_a, mask_a, ka)
        return DirectionCodelengths(
            nll_a_marg=nll_a_marg, nll_b_marg=nll_b_marg,
            nll_b_given_a=nll_b_given_a, nll_a_given_b=nll_a_given_b,
            n_a=n_a, n_b=n_b,
        )


def _collect(scorer: _CachedScorer, samples: list[dict], device: str) -> dict[str, np.ndarray]:
    """Score every sample; return per-variant score arrays + labels + groups + length."""
    cols = {k: [] for k in ("mdl", "conditional_asym", "log_len_ratio", "label", "group")}
    for s in samples:
        cl = scorer.score(
            s["input_ids_a"].unsqueeze(0).to(device), s["attention_mask_a"].unsqueeze(0).to(device),
            s["input_ids_b"].unsqueeze(0).to(device), s["attention_mask_b"].unsqueeze(0).to(device),
        )
        cols["mdl"].append(cl.mdl_score)
        cols["conditional_asym"].append(cl.conditional_asym_mean)
        cols["log_len_ratio"].append(cl.log_len_ratio)
        cols["label"].append(int(s["direction_label"].item()))
        cols["group"].append(s["group"])
    return {
        "mdl": np.array(cols["mdl"]),
        "conditional_asym": np.array(cols["conditional_asym"]),
        "log_len_ratio": np.array(cols["log_len_ratio"]),
        "label": np.array(cols["label"]),
        "group": np.array(cols["group"], dtype=object),
    }


def _residualize(x: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Return x with its OLS projection onto `on` (plus intercept) removed."""
    a = np.vstack([on, np.ones_like(on)]).T
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    return x - a @ coef


def _evaluate(data: dict[str, np.ndarray], n_resamples: int) -> dict:
    """Directed-only sign accuracy + length partial-correlation for each variant."""
    directed = data["label"] != 2
    y = data["label"][directed]
    grp = data["group"][directed]
    llr = data["log_len_ratio"][directed]
    y_sign = np.where(y == 0, 1.0, -1.0)  # A_to_B -> +1 (score should be positive)

    variants = {
        "conditional_asym": data["conditional_asym"][directed],
        "mdl": data["mdl"][directed],
        "resid_conditional": _residualize(data["conditional_asym"][directed], llr),
    }
    # How much of "direction" is just length in this split:
    length_only_r = float(np.corrcoef(llr, y_sign)[0, 1]) if np.std(llr) > 0 else float("nan")

    out = {"length_alone_corr_with_direction": round(length_only_r, 4), "variants": {}}
    for name, score in variants.items():
        pred = np.where(score > 0, 0, 1)  # >0 => A_to_B
        boot = accuracy_ci(y, pred, groups=grp, n_resamples=n_resamples, seed=42)
        raw_r = float(np.corrcoef(score, y_sign)[0, 1]) if np.std(score) > 0 else float("nan")
        # partial correlation with direction, controlling for length
        sr = _residualize(score, llr)
        yr = _residualize(y_sign, llr)
        partial_r = float(np.corrcoef(sr, yr)[0, 1]) if np.std(sr) > 0 else float("nan")
        out["variants"][name] = {
            "directed_accuracy": round(boot.accuracy, 4),
            "ci_low": round(boot.ci_low, 4),
            "ci_high": round(boot.ci_high, 4),
            "n_groups": boot.n_units,
            "n_samples": boot.n_samples,
            "corr_with_direction": round(raw_r, 4),
            "partial_corr_controlling_length": round(partial_r, 4),
        }
    return out


def main() -> None:
    """Run the multi-variant direction-signal analysis on synoptic + external data."""
    parser = argparse.ArgumentParser(description="Validate the NLL direction signal")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--no-dapt", action="store_true")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    koine = KoineFormer.from_pretrained(device=device)
    dapt_path = Path("models/koineformer/dapt/final")
    use_dapt = dapt_path.exists() and not args.no_dapt
    if use_dapt:
        koine.load_adapters(dapt_path)
    koine.model.resize_token_embeddings(len(tokenizer))
    model = koine.model
    model.eval()
    print(f"Encoder: {'DAPT' if use_dapt else 'zero-shot GreTa'}")

    scorer = _CachedScorer(model, make_empty_source(tokenizer, device), device)

    # Synoptic test split (DirectionDataset samples carry pericope_id; alias to 'group').
    syn = DirectionDataset(corpus, tokenizer, split="test", max_length=512,
                           min_aligned_tokens=5, use_scribal_noise=False)
    syn_samples = [
        {**syn[i], "group": syn.samples[i]["pericope_id"]} for i in range(len(syn))
    ]
    _LOG.info("scoring synoptic test pairs...")
    syn_data = _collect(scorer, syn_samples, device)

    # Synoptic only: this script's unique value is the multi-variant + length-control
    # + zero-shot comparison. External known-direction sets (Jude/2Pet, LXX Chronicles)
    # are handled by eval_external_direction.py.
    results = {
        "encoder": "dapt" if use_dapt else "zero-shot",
        "synoptic_test": _evaluate(syn_data, args.n_resamples),
    }

    print("\n" + "=" * 70)
    print(f"DIRECTION-SIGNAL ANALYSIS ({results['encoder']} encoder)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    suffix = "_zeroshot" if not use_dapt else ""
    out_path = Path(f"outputs/direction/signal_analysis{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
