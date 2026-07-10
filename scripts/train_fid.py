"""Train the Fusion-in-Decoder to reconstruct Mark from Matthew+Luke (Track A, M2).

Trains on full-triple pericopes with source-dropout (so one- and two-witness
conditionals share weights), then evaluates reconstruction quality on the held-out
fold against copy-a-single-witness baselines, with pericope-grouped bootstrap CIs.
The trained model is later applied to the double tradition to emit proto-Q (M6).

Needs a GPU for a real run; ``--limit`` + ``--epochs 1`` gives a fast CPU smoke test.

Usage:
    python scripts/train_fid.py --init-adapters models/koineformer_ns/final --fold 0
    python scripts/train_fid.py --limit 4 --epochs 1 --device cpu   # smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from scripts._cli_utils import detect_device  # noqa: E402
from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.data.redaction import fusion_examples, source_dropout_variants  # noqa: E402
from synoptiq.data.study_design import build_folds, full_triples  # noqa: E402
from synoptiq.evaluation.bootstrap import statistic_ci  # noqa: E402
from synoptiq.evaluation.reconstruction import nearest_witness_baseline, token_f1  # noqa: E402
from synoptiq.models.fid import FusionInDecoder  # noqa: E402
from synoptiq.training._config import StudyConfig  # noqa: E402
from synoptiq.utils.greek import normalize_greek  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tokens", type=Path, default=Path("data/processed/tokens.parquet"))
    p.add_argument("--pericopes", type=Path, default=Path("data/processed/pericopes.parquet"))
    p.add_argument("--init-adapters", type=Path, default=None,
                   help="KoineFormer-NS adapters to start from (recommended)")
    p.add_argument("--fold", type=int, default=0, help="held-out fold index")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--limit", type=int, default=None, help="cap train examples (smoke test)")
    p.add_argument("--witnesses", nargs="+", default=["Matthew", "Luke"],
                   help="witness books encoded and fused (default: Track A = Matthew Luke)")
    p.add_argument("--target", default="Mark",
                   help="reconstruction/scoring target book (default: Track A = Mark)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs/study/fid"))
    return p


def _evaluate(fid: FusionInDecoder, examples: list, device: str) -> dict[str, object]:
    """Reconstruct Mark for each held-out example; F1 vs gold and vs baselines."""
    f1s, base_f1s, groups = [], [], []
    for ex in examples:
        batch = fid.encode_example(list(ex.witness_texts.values()), ex.target_text)
        gen = fid.generate(batch["witness_input_ids"], batch["witness_masks"], max_new_tokens=256)
        pred = fid.tokenizer.decode(gen[0], skip_special_tokens=True)
        f1s.append(token_f1(pred, ex.target_text, normalize=normalize_greek))
        base_f1s.append(
            nearest_witness_baseline(
                list(ex.witness_texts.values()), ex.target_text, normalize=normalize_greek
            )
        )
        groups.append(ex.pericope_id)
    ci = statistic_ci(f1s, groups=groups, n_resamples=2000) if f1s else None
    return {
        "n": len(f1s),
        "fid_mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "nearest_witness_mean_f1": sum(base_f1s) / len(base_f1s) if base_f1s else 0.0,
        "fid_f1_ci": [ci.ci_low, ci.ci_high] if ci else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device or detect_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus = Corpus.from_parquet(args.tokens, args.pericopes)
    plan = build_folds(
        full_triples(corpus), n_folds=StudyConfig.n_folds, seed=StudyConfig.fold_seed
    )
    train_ids = set(plan.train_ids(args.fold))
    test_ids = set(plan.test_ids(args.fold))

    witnesses = tuple(args.witnesses)
    train_ex: list = []
    for ex in fusion_examples(corpus, witnesses=witnesses, target=args.target, ids=train_ids):
        train_ex.extend(source_dropout_variants(ex))
    if args.limit:
        train_ex = train_ex[: args.limit]
    test_ex = fusion_examples(corpus, witnesses=witnesses, target=args.target, ids=test_ids)

    _LOG.info("FiD training", extra={
        "fold": args.fold, "witnesses": witnesses, "target": args.target,
        "train_examples": len(train_ex), "test_pericopes": len(test_ex),
        "device": device, "init_adapters": str(args.init_adapters),
    })

    fid = FusionInDecoder.from_pretrained(init_adapters=args.init_adapters, device=device)
    optimizer = AdamW([p for p in fid.model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        fid.train()
        random.shuffle(train_ex)
        total = 0.0
        for ex in train_ex:
            batch = fid.encode_example(list(ex.witness_texts.values()), ex.target_text)
            optimizer.zero_grad()
            out = fid.forward(batch["witness_input_ids"], batch["witness_masks"], batch["labels"])
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
        _LOG.info(f"epoch {epoch + 1}/{args.epochs}",
                  extra={"mean_loss": total / max(1, len(train_ex))})

    metrics = _evaluate(fid, test_ex, device)
    metrics["witnesses"] = list(witnesses)
    metrics["target"] = args.target
    fid.save_adapters(args.out / f"fold{args.fold}")
    (args.out / f"reconstruction_fold{args.fold}.json").write_text(json.dumps(metrics, indent=2))

    fused = "+".join(witnesses)
    print(f"\n=== FiD reconstruction: {fused} -> {args.target} (held-out fold {args.fold}) ===")
    print(f"  FiD ({fused} fusion)  mean F1 = {metrics['fid_mean_f1']:.3f}  "
          f"CI {metrics['fid_f1_ci']}")
    print(f"  nearest-witness       mean F1 = {metrics['nearest_witness_mean_f1']:.3f}")
    print(f"  wrote: {args.out}/reconstruction_fold{args.fold}.json")
    print("  (reconstruction F1 is Track A's grade; for E2 the verdict comes from "
          "scoring — run scripts/run_mai_test.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
