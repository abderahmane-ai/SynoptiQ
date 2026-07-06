"""R5 / H5 — root the synoptic tree: pool per-pericope RPM votes into a stemma posterior.

Applies the RPM connective-smoothing canon (unsupervised on the synoptics; sign fixed by
textual criticism, validated only on external LXX/Jude) to every synoptic pericope, tallies
per-relationship directional votes, and computes a Bayesian posterior over the four rooted
stemmata (2SH / Farrer / Griesbach / Augustinian). Reports two things:

  1. MARKAN PRIORITY (triple tradition): do Mt-Mk and Mk-Lk vote Mark = source?
  2. FARRER vs Q (double tradition): does Mt-Lk on the Q material show a *consistent* Mt->Lk
     direction (Farrer) or no consistent direction (2SH / Q)?  Reported as a Bayes factor.

CPU only, deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.bayesian.rooting import (  # noqa: E402
    RelationshipCount,
    bayes_factor,
    posterior_over_stemmata,
)
from synoptiq.data.alignment import align_tokens  # noqa: E402
from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.data.frequency import build_frequency_table  # noqa: E402
from synoptiq.evaluation.variants import featurize_pair  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _vote(tokens_a, tokens_b, freq) -> int | None:  # noqa: ANN001
    """Connective-canon vote for the ordered pair (A, B).

    Returns +1 if A is the source, -1 if B is the source, None if the canon is silent
    (no connective substitution in this pericope => abstain).
    """
    try:
        align = align_tokens(tokens_a, tokens_b)
    except ValueError:
        return None
    pv = featurize_pair(tokens_a, tokens_b, align, freq)
    s = pv.sum_feature("connective_smooth")  # positive => A primitive (LXX-validated sign)
    if s > 0:
        return 1
    if s < 0:
        return -1
    return None


def _count(corpus: Corpus, relationship: tuple[str, str], tradition: str,
           freq) -> RelationshipCount:  # noqa: ANN001
    """Tally votes for one relationship over pericopes of one tradition."""
    a, b = relationship
    k = n = 0
    for per in corpus.iter_pericopes(tradition=tradition):
        toks = per["tokens"]
        ta, tb = toks.get(a, []), toks.get(b, [])
        if len(ta) < 5 or len(tb) < 5:
            continue
        v = _vote(list(ta), list(tb), freq)
        if v is None:
            continue
        n += 1
        if v > 0:
            k += 1
    return RelationshipCount(relationship, k, n)


def main() -> None:
    """Compute the stemma posterior and the Farrer-vs-Q Bayes factor."""
    freq = build_frequency_table()
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )

    mt_mk = _count(corpus, ("Matthew", "Mark"), "triple", freq)
    mk_lk = _count(corpus, ("Mark", "Luke"), "triple", freq)
    mt_lk_triple = _count(corpus, ("Matthew", "Luke"), "triple", freq)
    mt_lk_double = _count(corpus, ("Matthew", "Luke"), "double", freq)

    # Primary rooting: Markan-priority relationships from the triple tradition, and the
    # Farrer/Q discriminator (Mt-Lk) from the DOUBLE tradition (the Q material proper).
    counts = {
        ("Matthew", "Mark"): mt_mk,
        ("Mark", "Luke"): mk_lk,
        ("Matthew", "Luke"): mt_lk_double,
    }
    posterior = posterior_over_stemmata(counts)

    # Farrer vs 2SH differ ONLY on Mt-Lk => isolate that relationship for the verdict.
    farrer_vs_q_double = bayes_factor({("Matthew", "Luke"): mt_lk_double}, "Farrer", "2SH")
    farrer_vs_q_triple = bayes_factor({("Matthew", "Luke"): mt_lk_triple}, "Farrer", "2SH")

    report = {
        "relationship_votes": {
            "Matthew-Mark (triple)": vars(mt_mk),
            "Mark-Luke (triple)": vars(mk_lk),
            "Matthew-Luke (triple)": vars(mt_lk_triple),
            "Matthew-Luke (double/Q)": vars(mt_lk_double),
        },
        "stemma_posterior": posterior,
        "farrer_vs_q_bayes_factor": {
            "double_tradition_Q": round(farrer_vs_q_double, 3),
            "triple_tradition": round(farrer_vs_q_triple, 3),
        },
    }
    out = Path("outputs/direction/stemma_rooting.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print("R5 / H5 — SYNOPTIC ROOTING (RPM connective canon, unsupervised on synoptics)")
    print("=" * 78)
    print("\nPer-relationship votes (k = pericopes voting FIRST book is source, of n non-silent):")
    for label, rc in report["relationship_votes"].items():
        frac = rc["k"] / rc["n"] if rc["n"] else float("nan")
        first = label.split("-")[0]
        print(f"    {label:26s} k={rc['k']:3d}/n={rc['n']:3d}  "
              f"frac({first}=source)={frac:.3f}")

    print("\nStemma posterior (Mt-Mk & Mk-Lk triple + Mt-Lk double, uniform prior):")
    for h in ("2SH", "Farrer", "Griesbach", "Augustinian"):
        print(f"    {h:12s} P={posterior[h]['posterior']:.3f}  "
              f"logE={posterior[h]['log_evidence']:.2f}")

    print("\nFARRER vs Q (Bayes factor Farrer:2SH on Mt-Lk only; >1 favours Farrer):")
    print(f"    double tradition (Q material): BF = {farrer_vs_q_double:.3f}")
    print(f"    triple tradition:              BF = {farrer_vs_q_triple:.3f}")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
