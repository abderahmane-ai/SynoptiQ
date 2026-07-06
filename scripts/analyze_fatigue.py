"""Test editorial-fatigue directional features against the both-polarity criterion.

A signed fatigue feature (positive => A is the source) is only a real direction signal
if its sign predicts the true direction CONSISTENTLY on both external known-direction
sets of OPPOSITE length polarity:

  Jude -> 2 Peter        (copy LONGER  — expansion)
  LXX Samuel-Kings -> Chronicles (copy SHORTER — compression)

Passing both proves the signal is not a length prior (the failure mode of every global
score). Neither set involves a synoptic author, so it is not Markan style either. The
synoptic triple tradition is reported too, but as the confounded case.

Every feature is antisymmetric under an A<->B swap; we swap-augment each set so the
directed-accuracy test is balanced, and report pericope/block-grouped bootstrap CIs.

CPU only; no model, no Modal — just alignment + counting.
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
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.evaluation.fatigue import compute_fatigue_features  # noqa: E402
from synoptiq.utils.greek import normalize_greek  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

_SIGNED = ("intro_lateness_asym", "orphan_asym", "coverage_asym")


def _word_tokens(text: str) -> list[dict]:
    """Turn a raw Greek string into minimal token dicts for surface alignment."""
    return [{"normalized": normalize_greek(w), "text": w, "is_punctuation": False}
            for w in text.split() if w]


def _records_from_pair(
    src_tokens: list[dict], cpy_tokens: list[dict], group: str,
) -> list[dict]:
    """Two swap-augmented records for one known-direction pair (src is the source).

    Orientation 1: A=src (truth: A is source, y=0). Orientation 2: A=cpy (y=1).
    Features negate under swap, so this is a balanced, fair sign test.
    """
    try:
        align = align_tokens(src_tokens, cpy_tokens)
    except ValueError:
        return []
    swapped = [(j, i) for (i, j) in align]
    f_src = compute_fatigue_features(src_tokens, cpy_tokens, align).as_dict()
    f_cpy = compute_fatigue_features(cpy_tokens, src_tokens, swapped).as_dict()
    return [
        {"feats": f_src, "y": 0, "group": group},   # A is source
        {"feats": f_cpy, "y": 1, "group": group},   # A is copy
    ]


def _evaluate_dataset(records: list[dict], n_resamples: int = 2000) -> dict:
    """Directed accuracy per signed feature (fixed convention: positive => A source)."""
    y = np.array([r["y"] for r in records])
    groups = np.array([r["group"] for r in records], dtype=object)
    out = {"n_pairs": len(records) // 2, "n_groups": int(len(set(groups.tolist())))}
    for feat in _SIGNED:
        vals = np.array([r["feats"][feat] for r in records])
        pred = np.where(vals > 0, 0, 1)                 # >0 => predict A is source
        boot = accuracy_ci(y, pred, groups=groups, n_resamples=n_resamples, seed=1)
        # mean feature value when A is truly the source (sign should be consistent
        # across datasets if the feature is genuinely directional).
        mean_when_a_source = float(vals[y == 0].mean()) if (y == 0).any() else float("nan")
        out[feat] = {
            "acc": round(boot.accuracy, 3),
            "ci": [round(boot.ci_low, 3), round(boot.ci_high, 3)],
            "mean_when_A_source": round(mean_when_a_source, 4),
        }
    support = np.array([r["feats"]["n_shared_content"] for r in records])
    out["mean_shared_content"] = round(float(support.mean()), 1)
    return out


def _external_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for p in data["pairs"]:
        src = _word_tokens(p["text_a"])   # A = source by construction
        cpy = _word_tokens(p["text_b"])
        if len(src) < 5 or len(cpy) < 5:
            continue
        records.extend(_records_from_pair(src, cpy, p.get("group", p["id"])))
    return records


def _synoptic_records() -> list[dict]:
    processed = Path("data/processed")
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet", processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json", splits_path=processed / "splits.json",
    )
    records: list[dict] = []
    for pid, book_a, tokens_a, book_b, tokens_b, alignment in corpus.iter_direction_pairs():
        # Directed pairs only: Mark is the source under Markan priority.
        if book_a == "Mark":
            src, cpy = tokens_a, tokens_b
        elif book_b == "Mark":
            src, cpy = tokens_b, tokens_a
        else:
            continue  # Matthew<->Luke = independent, skip
        aligned = [(i, j) for i, j in alignment if i is not None or j is not None]
        if len([1 for i, j in aligned if i is not None and j is not None]) < 5:
            continue
        records.extend(_records_from_pair(list(src), list(cpy), pid))
    return records


def main() -> None:
    """Run the fatigue both-polarity test across the three datasets."""
    _LOG.info("aligning + scoring external Jude->2Peter (copy longer)...")
    jude = _evaluate_dataset(_external_records(Path("data/external/known_direction_pairs.json")))
    _LOG.info("aligning + scoring external LXX Chronicles (copy shorter)...")
    lxx = _evaluate_dataset(_external_records(Path("data/external/lxx_chronicles_pairs.json")))
    _LOG.info("aligning + scoring synoptic directed pairs (confounded)...")
    syn = _evaluate_dataset(_synoptic_records())

    report = {
        "criterion": (
            "a real signal has acc>0.5 (same side) on BOTH jude (copy longer) and "
            "lxx (copy shorter); that rules out a length prior."
        ),
        "datasets": {
            "jude_2peter_copy_longer": jude,
            "lxx_chronicles_copy_shorter": lxx,
            "synoptic_directed_CONFOUNDED": syn,
        },
    }

    # Verdict per feature (honest): a real signal needs, on BOTH external sets, a
    # bootstrap CI that clears chance on the SAME side. Point estimates alone on 6-14
    # pairs are too noisy to trust.
    verdicts = {}
    for feat in _SIGNED:
        def _clears(d: dict, f: str = feat) -> int:
            lo, hi = d[f]["ci"]
            if lo > 0.5:
                return 1   # confidently A-source-positive
            if hi < 0.5:
                return -1  # confidently the opposite sign
            return 0       # includes chance
        jc, lc = _clears(jude), _clears(lxx)
        if jc != 0 and jc == lc:
            verdicts[feat] = "PASSES both-polarity test (CIs clear chance, same side)"
        elif (jude[feat]["acc"] - 0.5) * (lxx[feat]["acc"] - 0.5) > 0:
            verdicts[feat] = "suggestive (same side) but CI includes chance — needs more data"
        else:
            verdicts[feat] = "fails (length-confounded or no signal)"
    report["verdicts"] = verdicts

    out = Path("outputs/direction/fatigue_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72)
    print("EDITORIAL-FATIGUE DIRECTIONAL TEST")
    print("=" * 72)
    for name, d in report["datasets"].items():
        print(f"\n{name}  (pairs={d['n_pairs']}, shared-content~{d['mean_shared_content']})")
        for feat in _SIGNED:
            f = d[feat]
            print(f"    {feat:22s} acc={f['acc']:.3f} CI{f['ci']}  "
                  f"mean|A=src={f['mean_when_A_source']:+.4f}")
    print("\nVERDICTS:")
    for feat, v in verdicts.items():
        print(f"    {feat:22s} {v}")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
