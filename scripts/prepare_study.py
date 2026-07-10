"""Set up the source-criticism study: census, folds, preregistration freeze, and power.

Two subcommands, both GPU-free — they operate on the corpus and its structure only:

    freeze  — compute the census, build the deterministic k-folds, and write the
              preregistration freeze block (config / folds / overlap hashes). Run this
              once; the hashes lock the design (see docs/SOURCE_CRITICISM_STUDY.md §10)
              before any double-tradition scoring.

    power   — the preregistered power analysis. Sweeps the per-token signal-to-noise
              ratio over the corpus's real per-pericope weights and reports each test's
              minimum detectable effect (MDE). This is kill criterion K2's input.

Usage:
    python scripts/prepare_study.py freeze
    python scripts/prepare_study.py freeze --n-folds 5 --seed 20260707
    python scripts/prepare_study.py power
    python scripts/prepare_study.py power --n-sims 2000 --sim-resamples 800
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.data.study_design import (  # noqa: E402
    build_folds,
    census,
    config_hash,
    double_tradition_units,
    fold_hash,
    full_triples,
    overlap_hash,
    overlap_partition,
)
from synoptiq.evaluation.model_comparison import mde_did, mde_lift, simulate_lift_power  # noqa: E402
from synoptiq.training._config import StudyConfig  # noqa: E402

_BETWEEN_GRID = (0.0, 0.10, 0.20, 0.30)  # tau: between-pericope effect heterogeneity (sigma units)


def _load_corpus(tokens: Path, pericopes: Path) -> Corpus:
    return Corpus.from_parquet(
        tokens,
        pericopes,
        alignments_path=tokens.parent / "alignments.json",
        splits_path=tokens.parent / "splits.json",
    )


# ── freeze ────────────────────────────────────────────────────────────────────


def cmd_freeze(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = _load_corpus(args.tokens, args.pericopes)

    c = census(corpus)
    (args.out / "census.json").write_text(json.dumps(c.to_dict(), indent=2, ensure_ascii=False))

    triples = full_triples(corpus)
    plan = build_folds(triples, n_folds=args.n_folds, seed=args.seed)
    plan_json = {"n_folds": plan.n_folds, "seed": plan.seed, "assignment": plan.assignment}
    (args.out / "folds.json").write_text(json.dumps(plan_json, indent=2))

    core_ov, core_rest = overlap_partition(triples, scope="core")
    ext_ov, _ = overlap_partition(triples, scope="extended")

    config = StudyConfig(n_folds=args.n_folds, fold_seed=args.seed)
    freeze = {
        "config_hash": config_hash(dataclasses.asdict(config)),
        "folds_hash": fold_hash(plan),
        "overlap_core_hash": overlap_hash("core"),
        "overlap_extended_hash": overlap_hash("extended"),
        "n_folds": args.n_folds,
        "fold_seed": args.seed,
    }
    (args.out / "freeze.json").write_text(json.dumps(freeze, indent=2))

    fold_sizes = {k: len(plan.test_ids(k)) for k in range(plan.n_folds)}
    print("\n=== CENSUS ===")
    print(f"full triples (E2 / operator-training units): {c.n_full_triples}")
    print(f"double tradition (E1 held-out, scored once): {c.n_double}")
    print(f"triple token totals:  {c.triple_token_totals}")
    print(f"double token totals:  {c.double_token_totals}")
    print(f"triple genres:        {c.triple_genres}")
    print(f"double genres:        {c.double_genres}")
    print(f"unlearnable double-tradition genres (no triple analog): {c.unlearnable_double_genres}")
    print(f"Mark-Q overlap (core / extended present): {c.n_overlap_core} / {c.n_overlap_extended}")
    print("\n=== FOLDS ===")
    print(f"{plan.n_folds}-fold over {c.n_full_triples} full triples, seed={args.seed}")
    print(f"fold sizes: {fold_sizes}")
    print(f"core overlap ids:      {core_ov}")
    print(f"extended-only overlap: {sorted(set(ext_ov) - set(core_ov))}")
    print(f"E2 DiD partition (core): overlap={len(core_ov)}  rest={len(core_rest)}")
    print("\n=== FREEZE BLOCK (docs/SOURCE_CRITICISM_STUDY.md §10) ===")
    for k, v in freeze.items():
        print(f"  {k}: {v}")
    print(f"\nwrote: {args.out}/census.json, folds.json, freeze.json")
    return 0


# ── power ─────────────────────────────────────────────────────────────────────


def cmd_power(args: argparse.Namespace) -> int:  # noqa: PLR0915
    args.out.mkdir(parents=True, exist_ok=True)
    fast = {"n_sims": args.n_sims, "sim_resamples": args.sim_resamples}

    corpus = _load_corpus(args.tokens, args.pericopes)
    triples = full_triples(corpus)
    doubles = double_tradition_units(corpus)
    core_ov, core_rest = overlap_partition(triples, scope="core")

    tri_w = {u.pericope_id: float(u.token_counts["Luke"]) for u in triples}
    e2_weights = np.array([tri_w[p] for p in tri_w])
    ov_weights = np.array([tri_w[p] for p in core_ov])
    rest_weights = np.array([tri_w[p] for p in core_rest])
    e1_weights = np.array([float(u.token_counts["Luke"]) for u in doubles])

    report: dict[str, object] = {
        "settings": {"target_power": args.target_power, **fast},
        "sample_sizes": {
            "E2_full_triples": int(e2_weights.size),
            "E2_overlap_core": int(ov_weights.size),
            "E2_rest": int(rest_weights.size),
            "E1_double_tradition": int(e1_weights.size),
        },
        "mde": {"E2_lift": {}, "E2_did": {}, "E1_channel": {}},
        "power_curve_reference": {},
    }

    print("\n=== POWER ANALYSIS ===")
    print(f"sims={args.n_sims}  resamples/sim={args.sim_resamples}  "
          f"target power={args.target_power}")
    print(f"N: E2 full triples={e2_weights.size}, overlap(core)={ov_weights.size}, "
          f"rest={rest_weights.size}, E1 double tradition={e1_weights.size}")
    print(f"median Luke target tokens: E2={np.median(e2_weights):.0f}  "
          f"E1={np.median(e1_weights):.0f}")

    grid = np.linspace(0.0, 1.5, 31)
    print("\n--- Minimum detectable effect (per-token snr) at 80% power ---")
    header = (f"{'tau (between-SD)':>16} | {'E2 lift N=' + str(e2_weights.size):>14} | "
              f"{'E2 DiD (5 vs ' + str(rest_weights.size) + ')':>18} | "
              f"{'E1 chan N=' + str(e1_weights.size):>14}")
    print(header)

    def fmt(x: float) -> str:
        return "unreach." if not np.isfinite(x) else f"{x:.3f}"

    for tau in _BETWEEN_GRID:
        m_lift = mde_lift(e2_weights, between_sd=tau, target_power=args.target_power,
                          snr_grid=grid, **fast)
        m_did = mde_did(ov_weights, rest_weights, between_sd=tau, target_power=args.target_power,
                        snr_grid=grid, **fast)
        m_e1 = mde_lift(e1_weights, between_sd=tau, target_power=args.target_power,
                        snr_grid=grid, **fast)
        report["mde"]["E2_lift"][f"{tau}"] = m_lift.mde_snr  # type: ignore[index]
        report["mde"]["E2_did"][f"{tau}"] = m_did.mde_snr    # type: ignore[index]
        report["mde"]["E1_channel"][f"{tau}"] = m_e1.mde_snr  # type: ignore[index]
        row = (f"{tau:>16.2f} | {fmt(m_lift.mde_snr):>14} | "
               f"{fmt(m_did.mde_snr):>18} | {fmt(m_e1.mde_snr):>14}")
        print(row)

    print("\n--- Detection rate at a reference effect (snr=0.3, tau=0.15) ---")
    r_lift = simulate_lift_power(e2_weights, snr=0.3, between_sd=0.15, **fast)
    r_e1 = simulate_lift_power(e1_weights, snr=0.3, between_sd=0.15, **fast)
    report["power_curve_reference"] = {
        "E2_lift": {"detection": r_lift.detection_rate, "fpr": r_lift.false_positive_rate},
        "E1_channel": {"detection": r_e1.detection_rate, "fpr": r_e1.false_positive_rate},
    }
    print(f"  E2 lift (N={e2_weights.size}):  detection={r_lift.detection_rate:.2f}  "
          f"false-positive={r_lift.false_positive_rate:.3f}")
    print(f"  E1 chan (N={e1_weights.size}):  detection={r_e1.detection_rate:.2f}  "
          f"false-positive={r_e1.false_positive_rate:.3f}")

    (args.out / "power_analysis.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote: {args.out}/power_analysis.json")
    print("\n=== READING (preregistered) ===")
    print("• The DiD contrast (5 overlap pericopes) is the binding bottleneck — its MDE is")
    print("  the largest of the three; treat it as the study's true resolution.")
    print("• E1 at the double-tradition N needs a materially larger per-token effect than E2;")
    print("  if the gates demonstrate an effect below E1's MDE, K2 fires and E1 is not run.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tokens", type=Path, default=Path("data/processed/tokens.parquet"))
    p.add_argument("--pericopes", type=Path, default=Path("data/processed/pericopes.parquet"))
    p.add_argument("--out", type=Path, default=Path("outputs/study"))
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze", help="census + folds + preregistration freeze block")
    f.add_argument("--n-folds", type=int, default=StudyConfig.n_folds)
    f.add_argument("--seed", type=int, default=StudyConfig.fold_seed)
    f.set_defaults(func=cmd_freeze)

    pw = sub.add_parser("power", help="preregistered power analysis / MDE curves")
    pw.add_argument("--n-sims", type=int, default=1500)
    pw.add_argument("--sim-resamples", type=int, default=600)
    pw.add_argument("--target-power", type=float, default=StudyConfig.power_target)
    pw.set_defaults(func=cmd_power)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
