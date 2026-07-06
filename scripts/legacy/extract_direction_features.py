"""Extract NLL-codelength direction features for the evaluation ladder.

Computes the FEATURE_NAMES vector (from the four NLL codelengths) for every pair in:
  - the synthetic redaction corpus  (train the component here; direction is clean)
  - the synoptic test split          (real, but direction confounded with authorship)
  - the external known-direction set (Jude -> 2 Peter)

Saves one .npz per regime (X, y, groups, split) under outputs/direction/features/.
This is the expensive step (T5 forward passes); training the component then reads the
cached features and is instant. Synthetic train is subsampled by group to bound cost.

CPU/MPS, no Modal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.data.external_pairs import load_external_pairs  # noqa: E402
from synoptiq.legacy.nll_direction import (  # noqa: E402
    CachedPairScorer,
    codelengths_to_features,
    make_empty_source,
)
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.legacy.direction_training import DirectionDataset  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

_DIR_TO_IDX = {"A_to_B": 0, "B_to_A": 1, "independent": 2}
_OUT = Path("outputs/direction/features")


def _encode(tokenizer: object, text: str, device: str, max_length: int) -> tuple:
    enc = tokenizer(  # type: ignore[operator]
        text, max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _extract(scorer: CachedPairScorer, records: list[dict], device: str) -> dict:
    """records: list of {ids_a, mask_a, ids_b, mask_b, label, group, split}."""
    feats, ys, groups, splits = [], [], [], []
    for i, r in enumerate(records):
        cl = scorer.codelengths(r["ids_a"], r["mask_a"], r["ids_b"], r["mask_b"])
        feats.append(codelengths_to_features(cl))
        ys.append(r["label"])
        groups.append(r["group"])
        splits.append(r["split"])
        if (i + 1) % 500 == 0:
            _LOG.info(f"  scored {i + 1}/{len(records)}")
    return {
        "X": np.stack(feats), "y": np.array(ys),
        "group": np.array(groups, dtype=object), "split": np.array(splits, dtype=object),
    }


def _synthetic_records(tokenizer, device, max_length, max_train_groups) -> list[dict]:  # noqa: ANN001
    data = json.loads(Path("data/synthetic/redaction_corpus.json").read_text(encoding="utf-8"))
    pairs = data["pairs"]
    # Subsample train by group to bound extraction cost; keep all val/test.
    train_groups = sorted({p["group"] for p in pairs if p["split"] == "train"})
    random.Random(0).shuffle(train_groups)
    keep = set(train_groups[:max_train_groups])
    recs = []
    for p in pairs:
        if p["split"] == "train" and p["group"] not in keep:
            continue
        ia, ma = _encode(tokenizer, p["text_a"], device, max_length)
        ib, mb = _encode(tokenizer, p["text_b"], device, max_length)
        recs.append({
            "ids_a": ia, "mask_a": ma, "ids_b": ib, "mask_b": mb,
            "label": _DIR_TO_IDX[p["direction"]], "group": p["group"], "split": p["split"],
        })
    return recs


def main() -> None:
    """Extract and cache direction features for all three regimes."""
    parser = argparse.ArgumentParser(description="Extract NLL direction features")
    parser.add_argument("--max-train-groups", type=int, default=700)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    koine = KoineFormer.from_pretrained(device=device)
    dapt = Path("models/koineformer/dapt/final")
    if dapt.exists():
        koine.load_adapters(dapt)
    koine.model.resize_token_embeddings(len(tokenizer))
    model = koine.model
    model.eval()

    scorer = CachedPairScorer(model, make_empty_source(tokenizer, device))
    _OUT.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic (train the component).
    _LOG.info("extracting synthetic redaction features...")
    syn_recs = _synthetic_records(tokenizer, device, args.max_length, args.max_train_groups)
    syn = _extract(scorer, syn_recs, device)
    np.savez(_OUT / "synthetic.npz", **syn)
    print(f"synthetic: {len(syn['y'])} pairs "
          f"(train={int((syn['split']=='train').sum())}, "
          f"val={int((syn['split']=='val').sum())}, test={int((syn['split']=='test').sum())})")

    # 2. Synoptic test (real, confounded).
    _LOG.info("extracting synoptic test features...")
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    ds = DirectionDataset(corpus, tokenizer, split="test", max_length=args.max_length,
                          min_aligned_tokens=5, use_scribal_noise=False)
    syn_test_recs = []
    for i in range(len(ds)):
        s = ds[i]
        syn_test_recs.append({
            "ids_a": s["input_ids_a"].unsqueeze(0).to(device),
            "mask_a": s["attention_mask_a"].unsqueeze(0).to(device),
            "ids_b": s["input_ids_b"].unsqueeze(0).to(device),
            "mask_b": s["attention_mask_b"].unsqueeze(0).to(device),
            "label": int(s["direction_label"].item()),
            "group": ds.samples[i]["pericope_id"], "split": "test",
        })
    syn_test = _extract(scorer, syn_test_recs, device)
    np.savez(_OUT / "synoptic_test.npz", **syn_test)
    print(f"synoptic_test: {len(syn_test['y'])} pairs")

    # 3. External (Jude -> 2 Peter).
    _LOG.info("extracting external features...")
    ext_samples = load_external_pairs(
        Path("data/external/known_direction_pairs.json"), tokenizer,
        max_length=args.max_length, augment_swap=True,
    )
    ext_recs = [{
        "ids_a": s["input_ids_a"].unsqueeze(0).to(device),
        "mask_a": s["attention_mask_a"].unsqueeze(0).to(device),
        "ids_b": s["input_ids_b"].unsqueeze(0).to(device),
        "mask_b": s["attention_mask_b"].unsqueeze(0).to(device),
        "label": int(s["direction_label"].item()),
        "group": s["group"], "split": "test",
    } for s in ext_samples]
    ext = _extract(scorer, ext_recs, device)
    np.savez(_OUT / "external.npz", **ext)
    print(f"external: {len(ext['y'])} pairs")
    print(f"\nFeatures saved under {_OUT}/")


if __name__ == "__main__":
    main()
