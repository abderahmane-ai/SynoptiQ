"""Train the four redaction operators R_Lk / R_Mt / G_Mt / G_Lk (Track B, M2).

Each operator is a source→target seq2seq trained on full-triple pericopes (train fold),
then its held-out conditional NLL is compared to two baselines: copying the source
verbatim, and a source-free target LM (empty source). An operator must beat both or it
has learned nothing (the M2 acceptance gate; failing it starts the K1 kill-clock).

Needs a GPU for a real run; ``--limit`` + ``--epochs 1`` gives a fast CPU smoke test.

Usage:
    python scripts/train_redactors.py --init-adapters models/koineformer_ns/final --fold 0
    python scripts/train_redactors.py --only R_Lk --limit 4 --epochs 1 --device cpu
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
from synoptiq.data.redaction import redaction_pairs  # noqa: E402
from synoptiq.data.study_design import build_folds, full_triples  # noqa: E402
from synoptiq.models.redactor import Redactor  # noqa: E402
from synoptiq.training._config import StudyConfig  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# name → (source_book, target_book)
_OPERATORS: dict[str, tuple[str, str]] = {
    "R_Lk": ("Mark", "Luke"),
    "R_Mt": ("Mark", "Matthew"),
    "G_Mt": ("Matthew", "Mark"),
    "G_Lk": ("Luke", "Mark"),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tokens", type=Path, default=Path("data/processed/tokens.parquet"))
    p.add_argument("--pericopes", type=Path, default=Path("data/processed/pericopes.parquet"))
    p.add_argument("--init-adapters", type=Path, default=None)
    p.add_argument("--only", choices=list(_OPERATORS), default=None, help="train one operator")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs/study/redactors"))
    return p


def _mean_nll(redactor: Redactor, pairs: list, *, source_free: bool = False) -> float:
    total, n = 0.0, 0
    for pr in pairs:
        source = "" if source_free else pr.source_text
        batch = redactor.encode_pair(source, pr.target_text)
        nll = redactor.score(batch["input_ids"], batch["attention_mask"], batch["labels"])
        total += float(nll.mean())
        n += 1
    return total / max(1, n)


def _train_one(
    name: str, corpus: Corpus, plan: object, args: argparse.Namespace, device: str
) -> dict:
    src, tgt = _OPERATORS[name]
    train_ids = set(plan.train_ids(args.fold))  # type: ignore[attr-defined]
    test_ids = set(plan.test_ids(args.fold))    # type: ignore[attr-defined]
    train_pairs = redaction_pairs(corpus, source_book=src, target_book=tgt, ids=train_ids)
    test_pairs = redaction_pairs(corpus, source_book=src, target_book=tgt, ids=test_ids)
    if args.limit:
        train_pairs = train_pairs[: args.limit]

    _LOG.info(f"training {name} ({src}->{tgt})",
              extra={"train": len(train_pairs), "test": len(test_pairs)})
    redactor = Redactor.from_pretrained(name=name, init_adapters=args.init_adapters, device=device)
    optimizer = AdamW([p for p in redactor.model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        redactor.train()
        random.shuffle(train_pairs)
        total = 0.0
        for pr in train_pairs:
            batch = redactor.encode_pair(pr.source_text, pr.target_text)
            optimizer.zero_grad()
            out = redactor.forward(batch["input_ids"], batch["attention_mask"], batch["labels"])
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
        _LOG.info(f"{name} epoch {epoch + 1}/{args.epochs}",
                  extra={"mean_loss": total / max(1, len(train_pairs))})

    nll = _mean_nll(redactor, test_pairs)
    # Mismatched-source control: pair each target with a *different* pericope's
    # source. If the operator uses real source content (not just target-language
    # style), the true source must beat a wrong one.
    rotated = test_pairs[1:] + test_pairs[:1]
    mismatch = [type(p)(p.pericope_id, src, tgt, other.source_text, p.target_text)
                for p, other in zip(test_pairs, rotated, strict=True)]
    nll_mismatch = _mean_nll(redactor, mismatch) if len(test_pairs) > 1 else float("nan")
    nll_free = _mean_nll(redactor, test_pairs, source_free=True)
    redactor.save_adapters(args.out / name)
    passed = nll < nll_free and (not len(test_pairs) > 1 or nll < nll_mismatch)
    _LOG.info(
        f"{name} eval",
        extra={"nll": nll, "nll_mismatch": nll_mismatch, "nll_free": nll_free, "passed": passed},
    )
    return {"operator": name, "held_out_nll": nll, "mismatched_source_nll": nll_mismatch,
            "source_free_nll": nll_free, "beats_baselines": passed}


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

    names = [args.only] if args.only else list(_OPERATORS)
    results = [_train_one(name, corpus, plan, args, device) for name in names]
    (args.out / f"redactors_fold{args.fold}.json").write_text(json.dumps(results, indent=2))

    print("\n=== REDACTION OPERATORS (held-out NLL, nats/token) ===")
    for r in results:
        flag = "PASS" if r["beats_baselines"] else "FAIL"
        print(f"  {r['operator']:5s}  nll={r['held_out_nll']:.3f}  "
              f"mismatch={r['mismatched_source_nll']:.3f}  "
              f"free={r['source_free_nll']:.3f}  [{flag}]")
    print(f"  wrote: {args.out}/redactors_fold{args.fold}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
