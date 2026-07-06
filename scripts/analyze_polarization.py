"""H1 gate: do the textual-criticism canons actually polarize direction on real data?

For each polarization feature (harder_reading = lectio difficilior, local_brevior, and
connective_smooth), we align every real known-direction pair, extract variants, aggregate
the signed feature over variants (positive => X is the source), and test whether its sign
predicts the true direction.

The decisive split is by LENGTH POLARITY: a genuine directional canon works whether the
copy is shorter (compression) or longer (expansion); a length proxy only works on one side.
We therefore bucket all external pairs into copy-shorter vs copy-longer and require a
feature to clear chance on BOTH. Reports block-grouped bootstrap CIs.

GATE: if no feature clears chance on both polarity buckets, stop (another rigorous negative).
Otherwise the survivors define the RPM feature set for R3.

CPU only; no model.
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
from synoptiq.evaluation.variants import FEATURE_NAMES, featurize_pair  # noqa: E402
from synoptiq.utils.greek import normalize_greek  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _word_tokens(text: str) -> list[dict]:
    return [{"normalized": normalize_greek(w), "text": w} for w in text.split() if w]


def _pair_scores(tokens_src, tokens_cpy, freq) -> dict[str, float] | None:  # noqa: ANN001
    """Mean signed feature scores for one aligned pair (X=src). None if unalignable."""
    try:
        align = align_tokens(tokens_src, tokens_cpy)
    except ValueError:
        return None
    pv = featurize_pair(tokens_src, tokens_cpy, align, freq)
    if not pv.variants:
        return None
    return {name: pv.mean_feature(name) for name in FEATURE_NAMES}


def _external_records(path: Path, freq) -> list[dict]:  # noqa: ANN001
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for p in data["pairs"]:
        src = _word_tokens(p["text_a"])   # A = source by construction
        cpy = _word_tokens(p["text_b"])
        if len(src) < 5 or len(cpy) < 5:
            continue
        scores = _pair_scores(src, cpy, freq)
        if scores is None:
            continue
        polarity = "copy_shorter" if len(cpy) < len(src) else "copy_longer"
        group = p.get("group", p["id"])
        # swap-augment: X=src (y=0), X=cpy (y=1, features negate).
        records.append({"scores": scores, "y": 0, "group": group, "polarity": polarity})
        records.append({"scores": {k: -v for k, v in scores.items()},
                        "y": 1, "group": group, "polarity": polarity})
    return records


def _synoptic_records(freq) -> list[dict]:  # noqa: ANN001
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    records: list[dict] = []
    for pid, book_a, tokens_a, book_b, tokens_b, alignment in corpus.iter_direction_pairs():
        if book_a == "Mark":
            src, cpy = list(tokens_a), list(tokens_b)
        elif book_b == "Mark":
            src, cpy = list(tokens_b), list(tokens_a)
        else:
            continue
        aligned = [(i, j) for i, j in alignment]
        if not aligned:
            continue
        # Re-align on the fly (corpus alignment matched on lemma+pos; we want variants).
        scores = _pair_scores(src, cpy, freq)
        if scores is None:
            continue
        records.append({"scores": scores, "y": 0, "group": pid, "polarity": "synoptic"})
        records.append({"scores": {k: -v for k, v in scores.items()},
                        "y": 1, "group": pid, "polarity": "synoptic"})
    return records


def _accuracy(records: list[dict], feature: str, n_resamples: int = 2000) -> dict:
    y = np.array([r["y"] for r in records])
    groups = np.array([r["group"] for r in records], dtype=object)
    vals = np.array([r["scores"][feature] for r in records])
    pred = np.where(vals > 0, 0, 1)   # >0 => predict X is source
    boot = accuracy_ci(y, pred, groups=groups, n_resamples=n_resamples, seed=1)
    return {"acc": round(boot.accuracy, 3),
            "ci": [round(boot.ci_low, 3), round(boot.ci_high, 3)],
            "n_pairs": len(records) // 2}


def main() -> None:
    """Run the H1 canon-polarization gate."""
    freq = build_frequency_table()
    _LOG.info("scoring external Jude->2Peter...")
    ext = _external_records(Path("data/external/known_direction_pairs.json"), freq)
    _LOG.info("scoring external LXX Chronicles...")
    ext += _external_records(Path("data/external/lxx_chronicles_pairs.json"), freq)
    _LOG.info("scoring synoptic directed pairs...")
    syn = _synoptic_records(freq)

    shorter = [r for r in ext if r["polarity"] == "copy_shorter"]
    longer = [r for r in ext if r["polarity"] == "copy_longer"]

    report = {"buckets": {}, "verdicts": {}}
    buckets = {
        "external_copy_shorter": shorter,
        "external_copy_longer": longer,
        "external_all": ext,
        "synoptic_confounded": syn,
    }
    for bname, recs in buckets.items():
        report["buckets"][bname] = {
            f: _accuracy(recs, f) for f in FEATURE_NAMES
        } if recs else {}

    # Gate verdict per feature: clears chance (CI) on the SAME side in BOTH polarity buckets.
    for f in FEATURE_NAMES:
        s = report["buckets"]["external_copy_shorter"].get(f)
        lo = report["buckets"]["external_copy_longer"].get(f)
        if not s or not lo:
            report["verdicts"][f] = "insufficient data"
            continue

        def _side(d: dict) -> int:
            if d["ci"][0] > 0.5:
                return 1
            if d["ci"][1] < 0.5:
                return -1
            return 0
        ss, ls = _side(s), _side(lo)
        if ss != 0 and ss == ls:
            report["verdicts"][f] = "PASSES H1 (clears chance on both polarities, same side)"
        elif (s["acc"] - 0.5) * (lo["acc"] - 0.5) > 0 and min(s["acc"], lo["acc"]) > 0.5:
            report["verdicts"][f] = "suggestive (both sides >0.5) but a CI includes chance"
        else:
            report["verdicts"][f] = "fails (length-confounded or no signal)"

    out = Path("outputs/direction/polarization_h1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 74)
    print("H1 GATE — DO THE TEXTUAL-CRITICISM CANONS POLARIZE DIRECTION?")
    print("=" * 74)
    for bname, feats in report["buckets"].items():
        if not feats:
            continue
        n = next(iter(feats.values()))["n_pairs"]
        print(f"\n{bname}  (pairs={n})")
        for f in FEATURE_NAMES:
            d = feats[f]
            print(f"    {f:20s} acc={d['acc']:.3f} CI{d['ci']}")
    print("\nVERDICTS (both-polarity gate):")
    for f, v in report["verdicts"].items():
        print(f"    {f:20s} {v}")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
