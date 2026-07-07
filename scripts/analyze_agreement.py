"""Three-way agreement structure over the synoptic triple tradition (theory-neutral).

Builds a Mark-anchored three-way alignment of every triple-tradition pericope and reports
the positional agreement spectrum. This is the robust, non-stylistic replacement for the
canon-based RPM: it reads copying structure off the *pattern of agreements* alone.

Fingerprints:
  - Markan hub: agreements involving Mark (mt_mk + mk_lk) >> Mt-Lk-against-Mark.
  - Griesbach kill: Markan-singular rate is far from 0 (Mark is not a conflator).
  - Mark-Q overlaps: pericopes with high Mt-Lk-against-Mark (should be the known ones,
    e.g. the Temptation and John the Baptist's preaching).

CPU only, deterministic.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.evaluation.agreement import agreement_spectrum, align_three  # noqa: E402


def main() -> None:
    """Compute and report the corpus-wide agreement spectrum."""
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )

    total: Counter[str] = Counter()
    rows: list[tuple[str, dict]] = []
    for per in corpus.iter_pericopes(tradition="triple"):
        toks = per["tokens"]
        mt, mk, lk = toks.get("Matthew", []), toks.get("Mark", []), toks.get("Luke", [])
        if not (mt and mk and lk):
            continue
        sp = agreement_spectrum(
            align_three(list(mt), list(mk), list(lk)), list(mt), list(mk), list(lk),
        ).as_dict()
        total.update(sp)
        rows.append((per["pericope_id"], sp))

    hub = total["mt_mk"] + total["mk_lk"]
    report = {
        "corpus_totals": dict(total),
        "markan_hub_ratio": round(hub / max(total["mt_lk_against_mark"], 1), 2),
        "markan_singular_rate": round(
            total["mark_singular"] / max(total["n_content_columns"], 1), 3),
        "top_mt_lk_against_mark": [
            {"pericope": pid, **{k: s[k] for k in ("mt_lk_against_mark", "triple", "mk_lk")}}
            for pid, s in sorted(rows, key=lambda r: -r[1]["mt_lk_against_mark"])[:8]
        ],
    }
    out = Path("outputs/direction/agreement_structure.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 74)
    print("THREE-WAY AGREEMENT STRUCTURE (triple tradition, content words)")
    print("=" * 74)
    for k in ("triple", "mt_mk", "mk_lk", "mt_lk_against_mark",
              "mark_singular", "matthew_singular", "luke_singular", "n_content_columns"):
        print(f"    {k:22s}: {total[k]}")
    print(f"\n  Markan-hub ratio (Mark-involving : Mt-Lk-against-Mark) = "
          f"{report['markan_hub_ratio']} : 1")
    print(f"  Markan-singular rate = {report['markan_singular_rate']} "
          "(Griesbach conflator would be ~ 0)")
    print("\n  Top Mt-Lk-against-Mark pericopes (expect the Mark-Q overlaps):")
    for e in report["top_mt_lk_against_mark"]:
        print(f"    {e['pericope']}: against-Mark={e['mt_lk_against_mark']} "
              f"triple={e['triple']} mk_lk={e['mk_lk']}")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
