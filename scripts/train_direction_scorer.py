"""Calibrate the DirectionScorer and run the gated validation (G1-G3), then emit scores.

  G1 (triangulated): per-pericope directed accuracy recovering Mark-as-source on the triple
      tradition, with a confidence-based abstention curve. This is the robust regime.
  G2 (pair-only): external known-direction pairs (Jude->2Peter, LXX Chronicles) on BOTH
      length polarities — a length prior would flip; a real signal does not.
  G3 (calibration): accuracy rises with confidence (the abstention curve is the evidence).

Then writes per-pericope DirectionScores for the whole corpus to outputs/direction/scores.json
— the artifact the Phase-6 comparison (scripts/compare_hypotheses.py) consumes. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.direction.features import pair_features  # noqa: E402
from synoptiq.direction.scorer import DirectionScorer  # noqa: E402
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.utils.greek import normalize_greek  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _word_tokens(text: str) -> list[dict]:
    return [{"normalized": normalize_greek(w), "lemma": normalize_greek(w),
             "pos": "X-", "is_punctuation": False} for w in text.split() if w]


def _external(path: Path) -> list[dict]:
    """External known-direction pairs; text_a is the source (X)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in data["pairs"]:
        src, cpy = _word_tokens(p["text_a"]), _word_tokens(p["text_b"])
        if len(src) < 5 or len(cpy) < 5:
            continue
        out.append({"src": src, "cpy": cpy,
                    "polarity": "copy_shorter" if len(cpy) < len(src) else "copy_longer"})
    return out


def _fit(scorer_data: list[dict]) -> DirectionScorer:
    """Fit pair-only weights on external pairs (X=src => positive)."""
    phi = [pair_features(d["src"], d["cpy"]) for d in scorer_data]
    return DirectionScorer().fit(phi, [1] * len(phi))


def _abstention_curve(correct: np.ndarray, conf: np.ndarray) -> dict:
    order = np.argsort(-conf)
    cov = {}
    for frac in (1.0, 0.5, 0.25):
        k = max(1, int(len(order) * frac))
        cov[f"cov{int(frac * 100)}"] = round(float(correct[order[:k]].mean()), 3)
    return cov


def _g1_triangulated(scorer: DirectionScorer, corpus: Corpus) -> dict:
    y, pred, conf, groups = [], [], [], []
    for per in corpus.iter_pericopes(tradition="triple"):
        t = per["tokens"]
        mt, mk, lk = t.get("Matthew", []), t.get("Mark", []), t.get("Luke", [])
        if not (mt and mk and lk):
            continue
        for bname, b, c in [("Matthew", mt, lk), ("Luke", lk, mt)]:
            r = scorer.score_pair(per["pericope_id"], "Mark", list(mk), bname, list(b),
                                  tokens_c=list(c))
            y.append(0)                                       # truth: Mark (A) is source
            pred.append(0 if r["predicted_direction"] == "A_to_B" else 1)
            conf.append(r["confidence"])
            groups.append(per["pericope_id"])
    y, pred = np.array(y), np.array(pred)
    correct = (pred == y).astype(int)
    boot = accuracy_ci(y, pred, groups=np.array(groups, dtype=object), n_resamples=2000, seed=1)
    return {"acc": round(boot.accuracy, 3), "ci": [round(boot.ci_low, 3), round(boot.ci_high, 3)],
            "n": len(y), "abstention": _abstention_curve(correct, np.array(conf))}


def _g2_pair_only(scorer: DirectionScorer, ext: list[dict], label: str) -> dict:
    """Directed accuracy on NON-ABSTAINING pairs + coverage (the pair-only regime abstains
    heavily where the weak features are silent — that is the honest, expected behaviour)."""
    out = {}
    for pol in ("copy_shorter", "copy_longer"):
        sub = [d for d in ext if d["polarity"] == pol]
        if not sub:
            continue
        correct = []
        for d in sub:
            r = scorer.score_pair(label, "SRC", d["src"], "CPY", d["cpy"])   # pair-only
            if r["predicted_direction"] == "independent":
                continue                                                     # abstained
            correct.append(int(r["predicted_direction"] == "A_to_B"))        # SRC is source
        cov = len(correct) / len(sub) if sub else 0.0
        out[pol] = {"acc_when_confident": round(float(np.mean(correct)), 3) if correct else None,
                    "coverage": round(cov, 3), "n": len(sub)}
    return out


def _emit_scores(scorer: DirectionScorer, corpus: Corpus, out: Path) -> int:
    scores = []
    for per in corpus.iter_pericopes():
        t = per["tokens"]
        books = [b for b in ("Matthew", "Mark", "Luke") if t.get(b)]
        for i in range(len(books)):
            for j in range(i + 1, len(books)):
                a, b = books[i], books[j]
                third = next((x for x in books if x not in (a, b)), None)
                tc = t.get(third) if third else None
                scores.append(scorer.score_pair(
                    per["pericope_id"], a, list(t[a]), b, list(t[b]),
                    tokens_c=list(tc) if tc else None))
    out.write_text(json.dumps(scores, indent=2))
    return len(scores)


def main() -> None:
    """Calibrate, validate (G1-G3), and emit the per-pericope scores artifact."""
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    jude = _external(Path("data/external/known_direction_pairs.json"))
    lxx = _external(Path("data/external/lxx_chronicles_pairs.json"))
    scorer = _fit(lxx + jude)

    g1 = _g1_triangulated(scorer, corpus)
    g2 = {"jude_2peter": _g2_pair_only(scorer, jude, "jude"),
          "lxx_chronicles": _g2_pair_only(scorer, lxx, "lxx")}
    out = Path("outputs/direction/scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_scores = _emit_scores(scorer, corpus, out)

    report = {"learned_weights": scorer.weights, "G1_triangulated": g1, "G2_pair_only": g2}
    (out.parent / "direction_validation.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 74)
    print("DIRECTION SCORER — CALIBRATION + GATED VALIDATION")
    print("=" * 74)
    print(f"\nLearned weights: {scorer.weights}")
    print(f"\nG1 (triangulated, Mark=source): acc={g1['acc']} CI{g1['ci']} n={g1['n']}")
    print(f"    abstention curve: {g1['abstention']}  (G3: accuracy rises with confidence)")
    print("\nG2 (pair-only, both length polarities — a length prior would flip):")
    for corp, res in g2.items():
        print(f"    {corp}: {res}")
    print(f"\nEmitted {n_scores} per-pericope DirectionScores -> {out}")


if __name__ == "__main__":
    main()
