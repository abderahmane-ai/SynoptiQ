"""Pool the per-fold E2 runs into the preregistered cross-validated verdict (M4).

Each fold's ``mai_fold{k}.json`` holds per-pericope NLL rows scored by that fold's held-out
FiD. Concatenating all folds gives the excess lift over every full triple (N≈65) and — the
point of the whole design — all five Mark-Q overlap pericopes in one difference-in-differences.

This is where the actual verdict is read: the excess lift with its pericope-clustered CI, the
G3 null floor it must clear, and the overlap-vs-rest DiD. Nothing here needs a GPU.

Usage:
    python scripts/pool_mai.py --glob 'outputs/study/mai/mai_fold*.json'
"""

from __future__ import annotations

import argparse
from glob import glob
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.evaluation.verdict import (  # noqa: E402
    did_contrast,
    minor_agreement_test,
    null_threshold,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--glob", default="outputs/study/mai/mai_fold*.json",
                   help="glob for the per-fold JSONs written by run_mai_test.py")
    p.add_argument("--out", type=Path, default=Path("outputs/study/mai/mai_pooled.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = sorted(glob(args.glob))
    if not paths:
        print(f"no fold files matched {args.glob!r}")
        return 2

    rows: list[dict] = []
    for path in paths:
        rows.extend(json.loads(Path(path).read_text())["rows"])

    base = [r["base"] for r in rows]
    added = [r["added"] for r in rows]
    control = [r["control"] for r in rows]
    groups = [r["pericope_id"] for r in rows]
    excess = [c - a for c, a in zip(control, added, strict=True)]
    null_excess = [r["control"] - r["null_added"] for r in rows]

    verdict = minor_agreement_test(base, added, control, groups=groups)
    floor = null_threshold(null_excess)
    ov = [e for e, r in zip(excess, rows, strict=True) if r["overlap"]]
    rest = [e for e, r in zip(excess, rows, strict=True) if not r["overlap"]]
    did = did_contrast(ov, rest) if len(ov) >= 2 and rest else None

    clears = verdict.excess.estimate > floor.threshold
    report = {
        "n_folds": len(paths), "n_pericopes": len(rows),
        "excess_lift": verdict.excess.estimate,
        "excess_lift_ci": [verdict.excess.ci_low, verdict.excess.ci_high],
        "prob_positive": verdict.excess.prob_positive,
        "g3_null_threshold": floor.threshold,
        "clears_g3_floor": clears,
        "n_overlap": len(ov), "n_rest": len(rest),
        "did_overlap_vs_rest": did.estimate if did else None,
        "did_ci": [did.ci_low, did.ci_high] if did else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\n=== E2 POOLED VERDICT — Matthew->Luke beyond Mark ({len(paths)} folds, "
          f"N={len(rows)} pericopes) ===")
    print(f"  excess lift (nats/tok): {verdict.excess.estimate:+.4f}  "
          f"CI [{verdict.excess.ci_low:+.4f}, {verdict.excess.ci_high:+.4f}]  "
          f"P(>0)={verdict.excess.prob_positive:.2f}")
    print(f"  G3 null floor: {floor.threshold:.4f}   CLEARS FLOOR: {clears}")
    if did:
        tag = "overlap-concentrated (2SH-like)" if did.ci_low > 0.0 else "not overlap-concentrated"
        print(f"  DiD overlap({len(ov)}) vs rest({len(rest)}): {did.estimate:+.4f}  "
              f"CI [{did.ci_low:+.4f}, {did.ci_high:+.4f}]  ({tag})")
    else:
        print(f"  DiD: need >=2 overlap pericopes (have {len(ov)})")
    print("\n  READING: a calibrated minor-agreement signal requires CLEARS FLOOR = True.")
    print(f"  wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
