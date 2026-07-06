"""R6 — does the Markan-priority signal survive controlling for Mark's καί-heavy style?

The one confound a reviewer will press: RPM votes "Mark is the source" on Mark–Matthew and
Mark–Luke because the connective canon marks the καί-rich text as primitive, and Mark simply
uses καί more *globally*. If so, the signal is Markan style, not per-edit direction.

This control settles it. For every triple-tradition Mark–X pericope we compute:
  - the RPM connective vote (positive => Mark is the source), and
  - Δκαί = global καί-density(Mark) − global καί-density(X).
We then stratify pericopes into tertiles by |Δκαί| and measure directed accuracy (truth: Mark
is the source) with pericope-grouped bootstrap CIs in each stratum. If the signal were the
global-density confound it would vanish in the matched (low-|Δκαί|) stratum. We also report the
raw per-edit vote ratio overall and on the matched pericopes.

CPU only, deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.alignment import align_tokens  # noqa: E402
from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.data.frequency import build_frequency_table  # noqa: E402
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.evaluation.variants import extract_variants, featurize_pair, variant_features  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _kai_density(toks) -> float:  # noqa: ANN001
    n = len(toks)
    if not n:
        return 0.0
    return sum(1 for t in toks if (t.get("normalized") or "").lower() == "και") / n


def _acc_ci(mask: np.ndarray, votes: np.ndarray, groups: np.ndarray) -> dict:
    """Directed accuracy + pericope-grouped CI on a stratum (truth: Mark=source)."""
    v = votes[mask]
    g = groups[mask]
    keep = v != 0                       # drop abstentions (silent canon)
    v, g = v[keep], g[keep]
    if len(v) == 0:
        return {"acc": None, "ci": [None, None], "n": 0}
    y = np.zeros(len(v), dtype=int)      # truth: Mark is source (label 0)
    pred = np.where(v > 0, 0, 1)         # vote>0 => Mark source
    boot = accuracy_ci(y, pred, groups=g, n_resamples=3000, seed=1)
    return {"acc": round(boot.accuracy, 3),
            "ci": [round(boot.ci_low, 3), round(boot.ci_high, 3)], "n": int(len(v))}


def main() -> None:
    """Run the καί-density-stratified Markan-priority control."""
    freq = build_frequency_table()
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )

    votes, dkai, groups = [], [], []
    edit_pos = edit_neg = 0
    for per in corpus.iter_pericopes(tradition="triple"):
        toks = per["tokens"]
        tm = toks.get("Mark", [])
        if len(tm) < 5:
            continue
        for other in ("Matthew", "Luke"):
            to = toks.get(other, [])
            if len(to) < 5:
                continue
            try:
                align = align_tokens(list(tm), list(to))
            except ValueError:
                continue
            pv = featurize_pair(list(tm), list(to), align, freq)   # X = Mark
            votes.append(pv.sum_feature("connective_smooth"))
            dkai.append(_kai_density(tm) - _kai_density(to))
            groups.append(per["pericope_id"])
            for v in extract_variants(list(tm), list(to), align):
                s = variant_features(v, freq)["connective_smooth"]
                edit_pos += s > 0
                edit_neg += s < 0

    votes = np.array(votes, dtype=float)
    adk = np.abs(np.array(dkai, dtype=float))
    groups = np.array(groups, dtype=object)

    # Tertiles of |Δκαί|: T1 = most matched, T3 = most divergent style.
    q1, q2 = np.quantile(adk, [1 / 3, 2 / 3])
    strata = {
        "T1_matched (|Δκαί| low)": adk <= q1,
        "T2_middle": (adk > q1) & (adk <= q2),
        "T3_divergent (|Δκαί| high)": adk > q2,
    }
    # Per-edit ratio on the most-matched half (|Δκαί| below median).
    matched_half = adk <= np.median(adk)
    m_pos = m_neg = 0
    # recount edits on matched-half pericopes
    idx = 0
    for per in corpus.iter_pericopes(tradition="triple"):
        toks = per["tokens"]
        tm = toks.get("Mark", [])
        if len(tm) < 5:
            continue
        for other in ("Matthew", "Luke"):
            to = toks.get(other, [])
            if len(to) < 5:
                continue
            try:
                align = align_tokens(list(tm), list(to))
            except ValueError:
                continue
            if matched_half[idx]:
                for v in extract_variants(list(tm), list(to), align):
                    s = variant_features(v, freq)["connective_smooth"]
                    m_pos += s > 0
                    m_neg += s < 0
            idx += 1

    report = {
        "overall": _acc_ci(np.ones(len(votes), bool), votes, groups),
        "by_kai_density_stratum": {k: _acc_ci(m, votes, groups) for k, m in strata.items()},
        "per_edit_vote_ratio": {
            "all": {"mark_source": int(edit_pos), "other_source": int(edit_neg),
                    "ratio": round(edit_pos / max(edit_neg, 1), 2)},
            "matched_half": {"mark_source": int(m_pos), "other_source": int(m_neg),
                             "ratio": round(m_pos / max(m_neg, 1), 2)},
        },
    }
    out = Path("outputs/direction/style_confound_control.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print("R6 — MARKAN PRIORITY vs καί-STYLE CONFOUND (control)")
    print("=" * 78)
    o = report["overall"]
    print(f"\nOverall Markan-priority directed accuracy: {o['acc']} CI{o['ci']} n={o['n']}")
    print("\nStratified by |Δκαί-density| (if this were pure style, T1 would collapse to ~0.5):")
    for k, v in report["by_kai_density_stratum"].items():
        print(f"    {k:28s} acc={v['acc']} CI{v['ci']} n={v['n']}")
    r = report["per_edit_vote_ratio"]
    print("\nPer-edit vote ratio (Mark-source : other-source):")
    print(f"    all edits     {r['all']['mark_source']}:{r['all']['other_source']} "
          f"= {r['all']['ratio']}:1")
    mh = r["matched_half"]
    print(f"    matched half  {mh['mark_source']}:{mh['other_source']} = {mh['ratio']}:1")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
