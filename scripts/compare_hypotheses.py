"""Phase 6 — consume the DirectionScorer's per-pericope scores → posterior over the stemmata.

Reads the ``outputs/direction/scores.json`` artifact emitted by
``scripts/train_direction_scorer.py`` and pools it through
``synoptiq.bayesian.rooting`` into a posterior over the four hypotheses (2SH / Farrer /
Griesbach / Augustinian) plus the Farrer-vs-Q Bayes factor. Markan-priority relationships
(Mt-Mk, Mk-Lk) are read from the triple tradition (triangulated regime); the Farrer/Q
discriminator (Mt-Lk) is read from the double tradition. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.bayesian.rooting import (  # noqa: E402
    bayes_factor,
    posterior_over_stemmata,
    relationship_counts,
)
from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def main() -> None:
    """Load emitted scores, build relationship counts, report the stemma posterior."""
    scores_path = Path("outputs/direction/scores.json")
    if not scores_path.exists():
        _LOG.error("run scripts/train_direction_scorer.py first to emit scores.json")
        return
    scores = json.loads(scores_path.read_text())

    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    tradition = {per["pericope_id"]: per["tradition"] for per in corpus.iter_pericopes()}

    # Markan relationships from triple tradition; Mt-Lk (Farrer/Q) from double tradition.
    def keep(ds: dict) -> bool:
        trad = tradition.get(ds["pericope_id"])
        pair = {ds["book_a"], ds["book_b"]}
        if pair == {"Matthew", "Luke"}:
            return trad == "double"
        return trad == "triple"

    counts = relationship_counts([ds for ds in scores if keep(ds)])
    posterior = posterior_over_stemmata(counts)
    bf_double = bayes_factor(
        {r: c for r, c in counts.items() if r == ("Matthew", "Luke")}, "Farrer", "2SH")

    report = {"posterior": posterior, "farrer_vs_q_bayes_factor": round(bf_double, 3)}
    out = Path("outputs/direction/hypothesis_posterior.json")
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 74)
    print("PHASE 6 — STEMMA POSTERIOR (from DirectionScorer output)")
    print("=" * 74)
    print("\nPer-relationship counts (k = first-book-source votes, of n non-abstaining):")
    for label, c in posterior["_relationships"].items():
        print(f"    {label:16s} k={c['k']}/n={c['n']}  frac_first={c['frac_first']}")
    print("\nStemma posterior (uniform prior):")
    for h in ("2SH", "Farrer", "Griesbach", "Augustinian"):
        print(f"    {h:12s} P={posterior[h]['posterior']:.3f}")
    print(f"\nFarrer vs Q (BF Farrer:2SH on Mt-Lk double tradition): {bf_double:.3f}")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
