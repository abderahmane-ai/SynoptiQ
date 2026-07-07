"""Identifiability ceiling: which static feature recovers KNOWN direction on BOTH polarities?

The honest test of what per-pair direction detection can achieve. On the known-direction
external corpora (Jude→2 Peter, copy longer; LXX Kings→Chronicles, copy shorter) it sweeps a
battery of candidate directional features (each signed so >0 predicts "A is the source") and
reports directed accuracy on each length polarity separately.

A feature is a **confound** if it flips sign across polarities (works on one, backwards on the
other — it is really tracking length). It is **genuinely directional** only if it is
correctly-signed (>0.5) on BOTH polarities.

Result (2026-07): every length/lexical feature FLIPS; the connective canon is BACKWARDS on
external data (it only "worked" on the synoptics via Markan style); the ONLY correctly-signed
features on both polarities are editorial fatigue (`intro_lateness`, ~0.61/0.65) and markedness
(~0.53/0.58) — both real but weak/underpowered on ~62 pairs. CPU only.
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

from synoptiq.data.alignment import align_tokens  # noqa: E402
from synoptiq.data.frequency import build_frequency_table  # noqa: E402
from synoptiq.direction.features import connective_vote, intro_lateness  # noqa: E402
from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.utils.greek import normalize_greek  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)
_FREQ = build_frequency_table()


def _toks(text: str) -> list[dict]:
    return [{"normalized": normalize_greek(w), "lemma": normalize_greek(w),
             "pos": "X-", "is_punctuation": False} for w in text.split() if w]


def _load(path: Path) -> list[tuple]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for x in data["pairs"]:
        a, b = _toks(x["text_a"]), _toks(x["text_b"])          # a = SOURCE, b = COPY
        if len(a) < 5 or len(b) < 5:
            continue
        out.append((a, b, "copy_shorter" if len(b) < len(a) else "copy_longer", x["id"]))
    return out


def _w(t: dict) -> str:
    return t["normalized"]


def _ttr(ts: list[dict]) -> float:
    ws = [_w(t) for t in ts]
    return len(set(ws)) / len(ws) if ws else 0.0


def _hapax(ts: list[dict]) -> float:
    c = Counter(_w(t) for t in ts)
    return sum(1 for v in c.values() if v == 1) / len(ts) if ts else 0.0


def _marked(ts: list[dict]) -> float:
    ws = [_w(t) for t in ts]
    return sum(_FREQ.markedness(x) for x in ws) / len(ws) if ws else 0.0


def _shared(a: list[dict], b: list[dict]) -> int:
    try:
        al = align_tokens(a, b)
    except ValueError:
        return 0
    return sum(1 for i, j in al if i is not None and j is not None)


def _features(a: list[dict], b: list[dict]) -> dict[str, float]:
    """Signed candidate features (>0 => A is the source)."""
    sh = _shared(a, b)
    return {
        "len_ratio (control)": float(len(a) - len(b)),
        "ttr_diff": _ttr(a) - _ttr(b),
        "hapax_diff": _hapax(a) - _hapax(b),
        "markedness_diff": _marked(a) - _marked(b),
        "coverage_asym": (sh / len(a)) - (sh / len(b)),
        "connective_smooth": connective_vote(a, b),
        "fatigue_intro_late": intro_lateness(a, b),
    }


def _acc(vals: list[float], groups: list) -> tuple[float, float, float, int]:
    v = np.array(vals)
    y = np.zeros(len(v), dtype=int)                 # truth: A (source); feature>0 predicts it
    pred = np.where(v > 0, 0, 1)
    b = accuracy_ci(y, pred, groups=np.array(groups, dtype=object), n_resamples=3000, seed=1)
    return b.accuracy, b.ci_low, b.ci_high, len(v)


def main() -> None:
    """Run the both-polarity identifiability sweep."""
    pairs = (_load(Path("data/external/known_direction_pairs.json"))
             + _load(Path("data/external/lxx_chronicles_pairs.json")))
    names = list(_features(pairs[0][0], pairs[0][1]))
    data = {n: {"copy_shorter": [], "copy_longer": []} for n in names}
    grp = {"copy_shorter": [], "copy_longer": []}
    for a, b, pol, pid in pairs:
        f = _features(a, b)
        for n in names:
            data[n][pol].append(f[n])
        grp[pol].append(pid)

    report = {}
    print("\n" + "=" * 84)
    print("IDENTIFIABILITY CEILING — which feature recovers KNOWN direction on BOTH polarities?")
    print("=" * 84)
    print(f"\n{'feature':22} | copy_shorter        | copy_longer         | verdict")
    print("-" * 84)
    for n in names:
        s = _acc(data[n]["copy_shorter"], grp["copy_shorter"])
        ll = _acc(data[n]["copy_longer"], grp["copy_longer"])
        correct = s[0] > 0.5 and ll[0] > 0.5
        sig = s[1] > 0.5 and ll[1] > 0.5
        backwards = s[0] < 0.5 and ll[0] < 0.5
        verdict = ("DIRECTIONAL" if (correct and sig) else "directional?" if correct
                   else "BACKWARDS" if backwards else "FLIPS(confound)")
        report[n] = {"copy_shorter": round(s[0], 3), "copy_longer": round(ll[0], 3),
                     "verdict": verdict}
        print(f"{n:22} | {s[0]:.2f} [{s[1]:.2f},{s[2]:.2f}] n={s[3]:<2} | "
              f"{ll[0]:.2f} [{ll[1]:.2f},{ll[2]:.2f}] n={ll[3]:<2} | {verdict}")
    out = Path("outputs/direction/identifiability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\nGenuinely directional = correctly-signed (>0.5) on BOTH polarities.")
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
