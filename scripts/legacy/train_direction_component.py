"""Train the MDL direction component on synthetic data and evaluate the full ladder.

The component (MDLDirectionHead) is trained ONLY on the synthetic same-author,
length-decorrelated redaction corpus — so it cannot exploit authorship or length — then
evaluated, frozen, on:

  synthetic val / test  (held-out AUTHORS: does the learned direction generalize?)
  external Jude->2Peter  (real, known direction, no synoptic author)
  synoptic test          (real, but direction confounded with Markan authorship)

Every metric is reported with a group-clustered bootstrap CI, and each eval set also
gets a length partial-correlation so we can see the direction score is not just length.
Reads cached features from outputs/direction/features/ (see extract_direction_features.py).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.evaluation.bootstrap import accuracy_ci  # noqa: E402
from synoptiq.legacy.nll_direction import FEATURE_NAMES  # noqa: E402
from synoptiq.legacy.direction import MDLDirectionHead  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)
_FEAT = Path("outputs/direction/features")
_LLR_IDX = FEATURE_NAMES.index("log_len_ratio")


def _load(name: str) -> dict:
    d = np.load(_FEAT / name, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _residualize(x: np.ndarray, on: np.ndarray) -> np.ndarray:
    a = np.vstack([on, np.ones_like(on)]).T
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    return x - a @ coef


def _train_head(x_tr: np.ndarray, y_tr: np.ndarray, steps: int = 1500) -> MDLDirectionHead:
    head = MDLDirectionHead()
    head.set_feature_stats(
        torch.tensor(x_tr.mean(0), dtype=torch.float32),
        torch.tensor(x_tr.std(0), dtype=torch.float32),
    )
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    opt = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    head.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(head(xt), yt)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def _predict(head: MDLDirectionHead, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = head(torch.tensor(x, dtype=torch.float32))
    probs = torch.softmax(logits, dim=1).numpy()
    return logits.numpy(), probs.argmax(1)


def _evaluate(head: MDLDirectionHead, data: dict, name: str, n_resamples: int) -> dict:
    x, y, grp = data["X"], data["y"], data["group"]
    logits, preds = _predict(head, x)
    # 3-class accuracy
    boot3 = accuracy_ci(y, preds, groups=grp, n_resamples=n_resamples, seed=1)
    # directed-only: predict A_to_B vs B_to_A by the sign of the direction logit d
    directed = y != 2
    yd, gd = y[directed], grp[directed]
    dir_logit = logits[directed, 0] - logits[directed, 1]  # 2d
    dpred = np.where(dir_logit > 0, 0, 1)
    bootd = accuracy_ci(yd, dpred, groups=gd, n_resamples=n_resamples, seed=1)
    # length control on the directed axis
    llr = x[directed, _LLR_IDX]
    y_sign = np.where(yd == 0, 1.0, -1.0)
    raw_r = float(np.corrcoef(dir_logit, y_sign)[0, 1]) if np.std(dir_logit) > 0 else float("nan")
    partial = float(np.corrcoef(_residualize(dir_logit, llr), _residualize(y_sign, llr))[0, 1]) \
        if np.std(_residualize(dir_logit, llr)) > 0 else float("nan")
    return {
        "set": name,
        "n_samples": int(len(y)),
        "n_groups": int(len(set(grp.tolist()))),
        "three_class_acc": round(boot3.accuracy, 4),
        "three_class_ci": [round(boot3.ci_low, 4), round(boot3.ci_high, 4)],
        "directed_acc": round(bootd.accuracy, 4),
        "directed_ci": [round(bootd.ci_low, 4), round(bootd.ci_high, 4)],
        "directed_corr": round(raw_r, 4),
        "directed_partial_corr_controlling_length": round(partial, 4),
    }


def main() -> None:
    """Train the component on synthetic data and evaluate the full ladder."""
    n_resamples = 2000
    syn = _load("synthetic.npz")
    tr = syn["split"] == "train"
    va = syn["split"] == "val"
    te = syn["split"] == "test"

    _LOG.info(f"training on {int(tr.sum())} synthetic pairs...")
    head = _train_head(syn["X"][tr], syn["y"][tr])

    ladder = [
        _evaluate(head, {k: syn[k][va] for k in ("X", "y", "group")},
                  "synthetic_val_heldout_authors", n_resamples),
        _evaluate(head, {k: syn[k][te] for k in ("X", "y", "group")},
                  "synthetic_test_heldout_authors", n_resamples),
        _evaluate(head, _load("external.npz"), "external_jude_2peter_COPY_LONGER", n_resamples),
    ]
    if (_FEAT / "lxx.npz").exists():
        ladder.append(
            _evaluate(head, _load("lxx.npz"), "external_lxx_chronicles_COPY_SHORTER", n_resamples),
        )
    ladder.append(
        _evaluate(head, _load("synoptic_test.npz"), "synoptic_test_CONFOUNDED", n_resamples),
    )

    # Weight interpretation: which features the direction head relies on.
    signed_w = head.direction_head.weight.detach().numpy().ravel()
    signed_names = ["cond_pair(0-1)", "marg_pair(3-4)", "infogain_pair(6-7)",
                    "cond_asym", "marg_asym", "infogain_asym", "mdl_score", "log_len_ratio"]
    weights = {n: round(float(w), 4) for n, w in zip(signed_names, signed_w)}

    def _acc(substr: str) -> float:
        return next((e["directed_acc"] for e in ladder if substr in e["set"]), float("nan"))

    syn_acc = _acc("synthetic_test")
    jude_acc = _acc("jude")           # copy LONGER
    chron_acc = _acc("chronicles")    # copy SHORTER
    # A genuine detector works on BOTH length polarities. A length prior does the
    # opposite on the two: high where the copy compresses, low where it expands.
    length_prior_signature = (chron_acc > 0.7 and jude_acc < 0.4)
    if syn_acc > 0.8 and length_prior_signature:
        verdict = (
            f"NO TRANSFERABLE DIRECTION SIGNAL. Synthetic held-out {syn_acc:.0%} is a "
            f"generator artifact (redactional smoothing). On real data behaviour is a pure "
            f"LENGTH prior: Chronicles copy-shorter {chron_acc:.0%} vs Jude/2Pet copy-longer "
            f"{jude_acc:.0%} — the model predicts 'shorter = copy', right for compression, "
            f"wrong for expansion. Short-passage direction is confounded (length + Markan "
            f"style); the bottleneck is the problem, not the model."
        )
    elif syn_acc > 0.8 and jude_acc > 0.6 and chron_acc > 0.6:
        verdict = "TRANSFERS on both length polarities — genuine direction signal."
    else:
        verdict = (
            f"Mixed: synthetic {syn_acc:.0%}, jude {jude_acc:.0%}, chronicles {chron_acc:.0%}."
        )

    report = {
        "component": "MDLDirectionHead trained on synthetic redaction corpus",
        "train_n": int(tr.sum()),
        "direction_head_weights": weights,
        "ladder": ladder,
        "verdict": verdict,
        "reading": (
            "synthetic held-out AUTHORS = does it learn generalizable direction; "
            "external = real known direction; synoptic = confounded with Mark."
        ),
    }
    out = Path("outputs/direction/component_report.json")
    out.write_text(json.dumps(report, indent=2))

    # Save the component weights.
    torch.save(head.state_dict(), Path("outputs/direction/mdl_direction_head.pt"))

    print(json.dumps(report, indent=2))
    print(f"\nReport: {out}\nWeights: outputs/direction/mdl_direction_head.pt")


if __name__ == "__main__":
    main()
