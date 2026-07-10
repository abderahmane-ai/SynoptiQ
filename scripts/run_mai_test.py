"""E2 minor-agreement information test on the triple tradition (M4, with the G3 null floor).

Uses a source-dropout FiD trained on (Mark, Matthew) -> Luke to compute, per held-out
pericope, whether adding Matthew to Mark reduces the NLL of Luke *beyond a matched control*:

    base_p    = NLL(Lk | Mk)                    # Mark alone
    added_p   = NLL(Lk | Mk, Mt)                # Mark + the real Matthew
    control_p = NLL(Lk | Mk, Mt_other)          # Mark + a mismatched Matthew (same length)
    excess_p  = control_p - added_p             # real Matthew beats a control ⇒ genuine info

Then the verdict (clustered bootstrap of mean excess) and the difference-in-differences
between the Mark-Q overlap partition and the rest. All from the *validated scoring path*
(FiD.score / teacher-forced NLL) — generation is not used, so the Track-A reconstruction
result does not bear on this.

Needs the trained FiD for the fold (from `train_fid.py --witnesses Mark Matthew --target Luke`).
Full CV = run per fold 0..4 and pool the JSONs. ``--limit`` gives a fast CPU smoke test.

Usage:
    python scripts/run_mai_test.py --fid-adapters outputs/study/fid_mai/fold0 --fold 0
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

from scripts._cli_utils import detect_device  # noqa: E402
from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.data.redaction import pericope_text  # noqa: E402
from synoptiq.data.study_design import build_folds, full_triples, overlap_partition  # noqa: E402
from synoptiq.models.fid import FusionInDecoder  # noqa: E402
from synoptiq.training._config import StudyConfig  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tokens", type=Path, default=Path("data/processed/tokens.parquet"))
    p.add_argument("--pericopes", type=Path, default=Path("data/processed/pericopes.parquet"))
    p.add_argument("--fid-adapters", type=Path, required=True,
                   help="FiD trained on (Mark,Matthew)->Luke with source-dropout, this fold")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--target", default="Luke")
    p.add_argument("--base", default="Mark")
    p.add_argument("--added", default="Matthew")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs/study/mai"))
    return p


def _score(fid: FusionInDecoder, witness_texts: list[str], target_text: str) -> float:
    batch = fid.encode_example(witness_texts, target_text)
    nll = fid.score(batch["witness_input_ids"], batch["witness_masks"], batch["labels"])
    return float(nll.mean())


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    from synoptiq.evaluation.verdict import did_contrast, minor_agreement_test, null_threshold

    args = _build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device or detect_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus = Corpus.from_parquet(args.tokens, args.pericopes)
    triples = full_triples(corpus)
    plan = build_folds(triples, n_folds=StudyConfig.n_folds, seed=StudyConfig.fold_seed)
    test_ids = plan.test_ids(args.fold)
    if args.limit:
        test_ids = test_ids[: args.limit]
    overlap_core, _ = overlap_partition(triples, scope="core")
    overlap_set = set(overlap_core)

    _LOG.info("E2 minor-agreement test", extra={
        "fold": args.fold, "held_out": len(test_ids), "device": device,
        "target": args.target, "base": args.base, "added": args.added,
    })

    fid = FusionInDecoder.from_pretrained(device=device)
    fid.load_adapters(args.fid_adapters)
    fid.eval()

    # Rotated (mismatched) "added" witness for the control and the null floor.
    rotated = test_ids[1:] + test_ids[:1]
    rotated2 = test_ids[2:] + test_ids[:2]

    rows = []
    for pid, ctrl_pid, null_pid in zip(test_ids, rotated, rotated2, strict=True):
        base_t = pericope_text(corpus, args.base, pid)
        added_t = pericope_text(corpus, args.added, pid)
        ctrl_t = pericope_text(corpus, args.added, ctrl_pid)
        null_t = pericope_text(corpus, args.added, null_pid)
        target_t = pericope_text(corpus, args.target, pid)
        base_nll = _score(fid, [base_t], target_t)
        added_nll = _score(fid, [base_t, added_t], target_t)
        control_nll = _score(fid, [base_t, ctrl_t], target_t)
        null_added_nll = _score(fid, [base_t, null_t], target_t)
        rows.append({
            "pericope_id": pid, "overlap": pid in overlap_set,
            "base": base_nll, "added": added_nll, "control": control_nll,
            "null_added": null_added_nll,
        })

    base = [r["base"] for r in rows]
    added = [r["added"] for r in rows]
    control = [r["control"] for r in rows]
    groups = [r["pericope_id"] for r in rows]

    verdict = minor_agreement_test(base, added, control, groups=groups)
    # G3 null floor: excess with two *different* control witnesses (no real signal).
    null_excess = [r["control"] - r["null_added"] for r in rows]
    floor = null_threshold(null_excess)

    excess = [c - a for c, a in zip(control, added, strict=True)]
    ov = [e for e, r in zip(excess, rows, strict=True) if r["overlap"]]
    rest = [e for e, r in zip(excess, rows, strict=True) if not r["overlap"]]
    did = did_contrast(ov, rest) if ov and rest else None

    report = {
        "fold": args.fold, "n": len(rows),
        "excess_lift": verdict.excess.estimate,
        "excess_lift_ci": [verdict.excess.ci_low, verdict.excess.ci_high],
        "prob_positive": verdict.excess.prob_positive,
        "significant_vs_zero": verdict.significant,
        "g3_null_threshold": floor.threshold,
        "clears_g3_floor": verdict.excess.estimate > floor.threshold,
        "did_overlap_vs_rest": did.estimate if did else None,
        "did_ci": [did.ci_low, did.ci_high] if did else None,
        "n_overlap": len(ov), "n_rest": len(rest),
        "rows": rows,  # per-pericope NLLs — pooled across folds by scripts/pool_mai.py
    }
    (args.out / f"mai_fold{args.fold}.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== E2 minor-agreement test — {args.added}->{args.target} beyond {args.base} "
          f"(fold {args.fold}, N={len(rows)}) ===")
    print(f"  excess lift (nats/tok): {verdict.excess.estimate:+.4f}  "
          f"CI [{verdict.excess.ci_low:+.4f}, {verdict.excess.ci_high:+.4f}]  "
          f"P(>0)={verdict.excess.prob_positive:.2f}")
    print(f"  G3 null floor: {floor.threshold:.4f}   clears floor: {report['clears_g3_floor']}")
    if did:
        print(f"  DiD overlap({len(ov)}) vs rest({len(rest)}): {did.estimate:+.4f}  "
              f"CI [{did.ci_low:+.4f}, {did.ci_high:+.4f}]")
    print(f"  wrote: {args.out}/mai_fold{args.fold}.json")
    print("  NOTE: single fold — pool folds 0..4 for the preregistered CV verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
