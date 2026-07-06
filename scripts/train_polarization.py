"""Train the Redactional Polarization Model and test aggregation + abstention (H2/H3).

Builds summed variant-feature vectors for real known-direction pairs, fits the linear
polarization aggregator, and evaluates:

  - learned weights (interpretability: connective-smoothing should dominate; the
    length-confounded lectio-brevior weight should collapse toward zero);
  - POLARITY TRANSFER (H3): train on copy-longer LXX, test on copy-shorter LXX and vice
    versa — a length prior fails this, a real canon transfers;
  - CROSS-CORPUS transfer: train LXX, test Jude->2Peter and the synoptic directed pairs;
  - ABSTENTION (H2): accuracy at 100 / 50 / 25 % coverage — does keeping only the most
    confident pericopes raise accuracy?

CPU only; no neural model.
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
from synoptiq.evaluation.variants import featurize_pair  # noqa: E402
from synoptiq.models.polarization import PolarizationScorer  # noqa: E402
from synoptiq.utils.greek import normalize_greek  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# local_brevior is EXCLUDED: H1 proved it is a pure length proxy (0.0 vs 1.0 across length
# polarities), and its raw magnitude is large enough to dominate the score even at a tiny
# learned weight. We keep only the confound-free canons and scale them (no centering, to
# preserve swap antisymmetry) so weight magnitude reflects true importance.
MODEL_FEATURES: tuple[str, ...] = ("harder_reading", "connective_smooth")


def _phi(tokens_src, tokens_cpy, freq) -> np.ndarray | None:  # noqa: ANN001
    """Summed per-variant feature vector for one pair (X=src). None if unalignable."""
    try:
        align = align_tokens(tokens_src, tokens_cpy)
    except ValueError:
        return None
    pv = featurize_pair(tokens_src, tokens_cpy, align, freq)
    if not pv.variants:
        return None
    return np.array([pv.sum_feature(n) for n in MODEL_FEATURES], dtype=float)


def _word_tokens(text: str) -> list[dict]:
    return [{"normalized": normalize_greek(w), "text": w} for w in text.split() if w]


def _external_set(path: Path, freq) -> dict:  # noqa: ANN001
    data = json.loads(path.read_text(encoding="utf-8"))
    phis, groups, polarity = [], [], []
    for p in data["pairs"]:
        src, cpy = _word_tokens(p["text_a"]), _word_tokens(p["text_b"])
        if len(src) < 5 or len(cpy) < 5:
            continue
        v = _phi(src, cpy, freq)
        if v is None:
            continue
        phis.append(v)
        groups.append(p.get("group", p["id"]))
        polarity.append("copy_shorter" if len(cpy) < len(src) else "copy_longer")
    return {"phi": np.array(phis), "group": np.array(groups, dtype=object),
            "polarity": np.array(polarity, dtype=object)}


def _synoptic_set(freq) -> dict:  # noqa: ANN001
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    phis, groups = [], []
    for pid, book_a, tokens_a, book_b, tokens_b, _al in corpus.iter_direction_pairs():
        if book_a == "Mark":
            src, cpy = list(tokens_a), list(tokens_b)
        elif book_b == "Mark":
            src, cpy = list(tokens_b), list(tokens_a)
        else:
            continue
        v = _phi(src, cpy, freq)
        if v is None:
            continue
        phis.append(v)
        groups.append(pid)
    return {"phi": np.array(phis), "group": np.array(groups, dtype=object)}


def _eval(scorer: PolarizationScorer, s: dict, n_resamples: int = 2000) -> dict:
    """Directed accuracy (all pairs have X=src) + abstention curve."""
    if len(s["phi"]) == 0:
        return {}
    score = scorer.score(s["phi"])
    correct = (score > 0).astype(int)
    y = np.zeros(len(score), dtype=int)  # truth: X is source (label 0)
    pred = np.where(score > 0, 0, 1)
    boot = accuracy_ci(y, pred, groups=s["group"], n_resamples=n_resamples, seed=1)
    # Abstention: keep the most confident coverage fraction.
    order = np.argsort(-np.abs(score))
    cov = {}
    for frac in (1.0, 0.5, 0.25):
        k = max(1, int(len(score) * frac))
        cov[f"cov{int(frac * 100)}"] = round(float(correct[order[:k]].mean()), 3)
    return {"acc": round(boot.accuracy, 3), "ci": [round(boot.ci_low, 3), round(boot.ci_high, 3)],
            "n": len(score), "abstention_accuracy": cov}


def main() -> None:
    """Fit RPM on LXX and run the H2/H3 transfer + abstention tests."""
    freq = build_frequency_table()
    _LOG.info("building feature vectors...")
    lxx = _external_set(Path("data/external/lxx_chronicles_pairs.json"), freq)
    jude = _external_set(Path("data/external/known_direction_pairs.json"), freq)
    syn = _synoptic_set(freq)

    # Scale features by their std on the training corpus (no centering => antisymmetry
    # preserved), so weight magnitude reflects true importance, not raw feature scale.
    scale = lxx["phi"].std(axis=0) + 1e-9
    for s in (lxx, jude, syn):
        s["phi"] = s["phi"] / scale

    short_mask = lxx["polarity"] == "copy_shorter"
    long_mask = lxx["polarity"] == "copy_longer"

    def _subset(s: dict, mask: np.ndarray) -> dict:
        return {"phi": s["phi"][mask], "group": s["group"][mask]}

    ones = lambda s: np.ones(len(s["phi"]), dtype=int)  # noqa: E731  (X=src in every pair)

    # Fit on all LXX (mixed polarity) — this is the real model.
    full = PolarizationScorer(MODEL_FEATURES).fit(lxx["phi"], ones(lxx))

    report = {
        "learned_weights_full": full.weight_dict(),
        # H3: the mixed-trained model must work on BOTH length polarities. A length prior
        # would score one polarity well and the other at 0; a real canon is even-handed.
        "both_polarity_H3": {
            "lxx_copy_shorter": _eval(full, _subset(lxx, short_mask)),
            "lxx_copy_longer": _eval(full, _subset(lxx, long_mask)),
        },
        "cross_corpus": {
            "train_lxx_test_jude_2peter": _eval(full, jude),
            "train_lxx_test_synoptic": _eval(full, syn),
            "lxx_in_sample": _eval(full, lxx),
        },
    }

    out = Path("outputs/direction/polarization_rpm.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 74)
    print("RPM — AGGREGATION + ABSTENTION (H2/H3)")
    print("=" * 74)
    print(f"\nLearned weights (positive => X-source evidence): {report['learned_weights_full']}")
    print("\nBOTH-POLARITY (H3) — length prior scores one side ~0, a real canon is even-handed:")
    for k, v in report["both_polarity_H3"].items():
        print(f"    {k:28s} acc={v.get('acc')} CI{v.get('ci')} n={v.get('n')}")
    print("\nCROSS-CORPUS + ABSTENTION (H2):")
    for k, v in report["cross_corpus"].items():
        print(f"    {k:28s} acc={v.get('acc')} CI{v.get('ci')} "
              f"n={v.get('n')} abstention={v.get('abstention_accuracy')}")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
