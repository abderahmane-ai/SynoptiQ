"""Train the Redactional Polarization Model and test aggregation, abstention, fatigue.

Builds per-pair feature vectors for real known-direction pairs, fits the linear
polarization aggregator, and evaluates a ladder of hypotheses:

  - H2 (aggregation): does summing per-variant polarizations with abstention beat the
    dead global scores and pass the both-polarity criterion?
  - H3 (transfer): the mixed-polarity-trained model must work on BOTH length polarities
    (a length prior scores one side ~0); train LXX, test Jude->2Peter and synoptics.
  - H4 (fatigue increment): does folding editorial fatigue (`intro_lateness`, the one
    fatigue feature that survived length-decorrelation) in as a SECOND canon add signal
    over the connective-smoothing canon alone? Compared head-to-head here.

Three models are fit and run through the identical ladder:
  - "canon"          — static textual-criticism canons (harder_reading + connective_smooth)
  - "canon+fatigue"  — canons + the positional fatigue term (the R4 candidate)
  - "fatigue_only"   — intro_lateness alone (isolates fatigue's standalone power)

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
from synoptiq.evaluation.fatigue import compute_fatigue_features  # noqa: E402
from synoptiq.evaluation.variants import featurize_pair  # noqa: E402
from synoptiq.models.polarization import PolarizationScorer  # noqa: E402
from synoptiq.utils.greek import normalize_greek  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# The full per-pair feature vector. `local_brevior` is EXCLUDED throughout: H1 proved it a
# pure length proxy (0.0 vs 1.0 across polarity) whose raw magnitude dominates the score.
#   - harder_reading, connective_smooth : summed per-variant canon features (variants.py)
#   - intro_lateness                    : within-pair positional fatigue (fatigue.py), the
#                                         one fatigue feature that survived length-decorrelation
# All three share the sign convention "positive => X (the first passage) is the source", so
# every column negates under an X<->Y swap and the aggregate stays antisymmetric.
FULL_FEATURES: tuple[str, ...] = ("harder_reading", "connective_smooth", "intro_lateness")

# Models compared head-to-head on the identical ladder (this is the H4 test).
MODELS: dict[str, tuple[str, ...]] = {
    "canon": ("harder_reading", "connective_smooth"),
    "canon+fatigue": ("harder_reading", "connective_smooth", "intro_lateness"),
    "fatigue_only": ("intro_lateness",),
}


def _phi(tokens_src, tokens_cpy, freq) -> np.ndarray | None:  # noqa: ANN001
    """Full per-pair feature vector over FULL_FEATURES (X=src). None if unalignable."""
    try:
        align = align_tokens(tokens_src, tokens_cpy)
    except ValueError:
        return None
    pv = featurize_pair(tokens_src, tokens_cpy, align, freq)
    if not pv.variants:
        return None
    fat = compute_fatigue_features(tokens_src, tokens_cpy, align)
    feats = {
        "harder_reading": pv.sum_feature("harder_reading"),
        "connective_smooth": pv.sum_feature("connective_smooth"),
        "intro_lateness": fat.intro_lateness_asym,
    }
    return np.array([feats[n] for n in FULL_FEATURES], dtype=float)


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


def _eval(scorer: PolarizationScorer, phi: np.ndarray, groups: np.ndarray,
          n_resamples: int = 2000) -> dict:
    """Directed accuracy (all pairs have X=src) + abstention curve for a sliced phi."""
    if len(phi) == 0:
        return {}
    score = scorer.score(phi)
    correct = (score > 0).astype(int)
    y = np.zeros(len(score), dtype=int)  # truth: X is source (label 0)
    pred = np.where(score > 0, 0, 1)
    boot = accuracy_ci(y, pred, groups=groups, n_resamples=n_resamples, seed=1)
    order = np.argsort(-np.abs(score))
    cov = {}
    for frac in (1.0, 0.5, 0.25):
        k = max(1, int(len(score) * frac))
        cov[f"cov{int(frac * 100)}"] = round(float(correct[order[:k]].mean()), 3)
    return {"acc": round(boot.accuracy, 3), "ci": [round(boot.ci_low, 3), round(boot.ci_high, 3)],
            "n": len(score), "abstention_accuracy": cov}


def _run_model(feats: tuple[str, ...], lxx: dict, jude: dict, syn: dict,
               short_mask: np.ndarray, long_mask: np.ndarray) -> dict:
    """Fit one model on its feature subset (all mixed LXX) and run the full ladder."""
    cols = [FULL_FEATURES.index(f) for f in feats]

    def sl(s: dict) -> np.ndarray:
        return s["phi"][:, cols]

    ones = np.ones(len(lxx["phi"]), dtype=int)  # X=src in every pair
    scorer = PolarizationScorer(feats).fit(sl(lxx), ones)
    return {
        "features": list(feats),
        "learned_weights": scorer.weight_dict(),
        "both_polarity_H3": {
            "lxx_copy_shorter": _eval(scorer, sl(lxx)[short_mask], lxx["group"][short_mask]),
            "lxx_copy_longer": _eval(scorer, sl(lxx)[long_mask], lxx["group"][long_mask]),
        },
        "cross_corpus": {
            "train_lxx_test_jude_2peter": _eval(scorer, sl(jude), jude["group"]),
            "train_lxx_test_synoptic": _eval(scorer, sl(syn), syn["group"]),
            "lxx_in_sample": _eval(scorer, sl(lxx), lxx["group"]),
        },
    }


def main() -> None:
    """Fit the RPM model family and run the H2/H3/H4 ladder."""
    freq = build_frequency_table()
    _LOG.info("building feature vectors...")
    lxx = _external_set(Path("data/external/lxx_chronicles_pairs.json"), freq)
    jude = _external_set(Path("data/external/known_direction_pairs.json"), freq)
    syn = _synoptic_set(freq)

    # Scale each column by its std on the training corpus (no centering => antisymmetry
    # preserved), so weight magnitude reflects true importance, not raw feature scale.
    scale = lxx["phi"].std(axis=0) + 1e-9
    for s in (lxx, jude, syn):
        s["phi"] = s["phi"] / scale

    short_mask = lxx["polarity"] == "copy_shorter"
    long_mask = lxx["polarity"] == "copy_longer"

    report = {name: _run_model(feats, lxx, jude, syn, short_mask, long_mask)
              for name, feats in MODELS.items()}

    out = Path("outputs/direction/polarization_rpm.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print("RPM — AGGREGATION + ABSTENTION + FATIGUE (H2/H3/H4)")
    print("=" * 78)
    for name, r in report.items():
        print(f"\n### MODEL: {name}   features={r['features']}")
        print(f"  weights (positive => X-source): {r['learned_weights']}")
        print("  both-polarity (H3) — a length prior scores one side ~0:")
        for k, v in r["both_polarity_H3"].items():
            print(f"    {k:22s} acc={v.get('acc')} CI{v.get('ci')} n={v.get('n')}")
        print("  cross-corpus + abstention (H2):")
        for k, v in r["cross_corpus"].items():
            print(f"    {k:26s} acc={v.get('acc')} CI{v.get('ci')} "
                  f"n={v.get('n')} abst={v.get('abstention_accuracy')}")

    # H4 verdict: does adding fatigue raise synoptic directed accuracy / coverage?
    base = report["canon"]["cross_corpus"]["train_lxx_test_synoptic"]
    aug = report["canon+fatigue"]["cross_corpus"]["train_lxx_test_synoptic"]
    print("\n" + "-" * 78)
    print("H4 VERDICT (synoptic directed accuracy, canon vs canon+fatigue):")
    print(f"    canon         acc={base.get('acc')} @25%={base['abstention_accuracy']['cov25']}")
    print(f"    canon+fatigue acc={aug.get('acc')} @25%={aug['abstention_accuracy']['cov25']}")

    # H4b — the decisive fairness check. A second canon can only add value on pericopes the
    # first canon is SILENT about (no connective variant fires). Does fatigue beat chance
    # THERE? If not, no combination scheme can rescue it on the synoptics.
    conn_idx = FULL_FEATURES.index("connective_smooth")
    fat_idx = FULL_FEATURES.index("intro_lateness")
    conn = syn["phi"][:, conn_idx]
    silent = conn == 0.0                      # connective canon says nothing
    fired = ~silent
    fat_score = syn["phi"][:, fat_idx]        # fatigue's own (signed) vote
    def _acc(mask: np.ndarray, score: np.ndarray) -> tuple[float, int]:
        if not mask.any():
            return float("nan"), 0
        return round(float((score[mask] > 0).mean()), 3), int(mask.sum())
    fat_silent = _acc(silent, fat_score)
    fat_fired = _acc(fired, fat_score)
    conn_fired = _acc(fired, conn)
    print("\n" + "-" * 78)
    print("H4b — fatigue's power exactly where the connective canon is SILENT:")
    print(f"    connective FIRES: n={fired.sum()}  connective acc={conn_fired[0]}  "
          f"fatigue acc(here)={fat_fired[0]}")
    print(f"    connective SILENT: n={silent.sum()}  fatigue acc(here)={fat_silent[0]} "
          f"(chance=0.5 => no complementary coverage)")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
