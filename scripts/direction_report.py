"""Reproduce the validated direction findings in one place (robustness evidence).

  1. AGREEMENT STRUCTURE (triple tradition): Markan hub, Griesbach-conflator kill, the
     Mark-Q overlaps — the theory-neutral signal underpinning the triangulated regime.
  2. STYLE-CONFOUND CONTROL: the demoted connective canon's Markan-priority signal survives
     matching global καί-density (so even the weak feature is per-edit, not pure style).

The negative results (global passage scores are confounded with length / Markan style) and
the full narrative live in docs/DIRECTION_SCORER_FINDINGS.md. CPU only.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.direction.alignment3 import align_three  # noqa: E402
from synoptiq.direction.features import agreement_spectrum, connective_vote  # noqa: E402
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _kai_density(toks) -> float:  # noqa: ANN001
    n = len(toks)
    return sum(1 for t in toks if (t.get("normalized") or "").lower() == "και") / n if n else 0.0


def _agreement_structure(corpus: Corpus) -> dict:
    total: Counter[str] = Counter()
    rows = []
    for per in corpus.iter_pericopes(tradition="triple"):
        t = per["tokens"]
        mt, mk, lk = t.get("Matthew", []), t.get("Mark", []), t.get("Luke", [])
        if not (mt and mk and lk):
            continue
        sp = agreement_spectrum(
            align_three(list(mt), list(mk), list(lk)), list(mt), list(mk), list(lk)).as_dict()
        total.update(sp)
        rows.append((per["pericope_id"], sp["mt_lk_against_mark"]))
    hub = total["mt_mk"] + total["mk_lk"]
    return {
        "totals": dict(total),
        "markan_hub_ratio": round(hub / max(total["mt_lk_against_mark"], 1), 2),
        "markan_singular_rate": round(
            total["mark_singular"] / max(total["n_content_columns"], 1), 3),
        "top_overlaps": [p for p, _ in sorted(rows, key=lambda r: -r[1])[:5]],
    }


def _style_control(corpus: Corpus) -> dict:
    votes, adk, groups = [], [], []
    for per in corpus.iter_pericopes(tradition="triple"):
        t = per["tokens"]
        mk = t.get("Mark", [])
        if len(mk) < 5:
            continue
        for other in ("Matthew", "Luke"):
            to = t.get(other, [])
            if len(to) < 5:
                continue
            votes.append(connective_vote(list(mk), list(to)))       # + => Mark primitive
            adk.append(abs(_kai_density(mk) - _kai_density(to)))
            groups.append(per["pericope_id"])
    votes, adk, groups = np.array(votes), np.array(adk), np.array(groups, dtype=object)
    fired = votes != 0
    y = np.zeros(int(fired.sum()), dtype=int)                       # truth: Mark source
    pred = np.where(votes[fired] > 0, 0, 1)
    boot = accuracy_ci(y, pred, groups=groups[fired], n_resamples=2000, seed=1)
    matched = fired & (adk <= np.median(adk[fired]))
    ym = np.zeros(int(matched.sum()), dtype=int)
    pm = np.where(votes[matched] > 0, 0, 1)
    bm = accuracy_ci(ym, pm, groups=groups[matched], n_resamples=2000, seed=1)
    return {
        "overall": {"acc": round(boot.accuracy, 3),
                    "ci": [round(boot.ci_low, 3), round(boot.ci_high, 3)], "n": int(fired.sum())},
        "matched_kai_density": {"acc": round(bm.accuracy, 3),
                                "ci": [round(bm.ci_low, 3), round(bm.ci_high, 3)],
                                "n": int(matched.sum())},
    }


def main() -> None:
    """Print and save the reproduction report."""
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    report = {"agreement_structure": _agreement_structure(corpus),
              "style_confound_control": _style_control(corpus)}
    out = Path("outputs/direction/direction_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    a = report["agreement_structure"]
    s = report["style_confound_control"]
    print("\n" + "=" * 74)
    print("DIRECTION FINDINGS — REPRODUCTION REPORT")
    print("=" * 74)
    print("\n1. AGREEMENT STRUCTURE (triple tradition, theory-neutral):")
    print("    Markan-hub ratio (Mark-involving : Mt-Lk-against-Mark) = "
          f"{a['markan_hub_ratio']} : 1")
    print(f"    Markan-singular rate = {a['markan_singular_rate']} (Griesbach conflator ~ 0)")
    print(f"    top Mt-Lk-against-Mark pericopes (Mark-Q overlaps): {a['top_overlaps']}")
    print("\n2. STYLE-CONFOUND CONTROL (connective canon, demoted feature):")
    print(f"    overall Markan-priority acc = {s['overall']['acc']} CI{s['overall']['ci']} "
          f"n={s['overall']['n']}")
    print(f"    matched καί-density         = {s['matched_kai_density']['acc']} "
          f"CI{s['matched_kai_density']['ci']} n={s['matched_kai_density']['n']} "
          "(would be ~0.5 if pure style)")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
