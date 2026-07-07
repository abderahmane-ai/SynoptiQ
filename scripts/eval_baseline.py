"""Paper A baseline evaluation pipeline.

Evaluates KoineFormer (after DAPT) against zero-shot GreTa on the
SynoptiQ test set.  Produces the comparison tables for Paper A.

Usage:
    # Evaluate zero-shot GreTa (before DAPT — always available):
    python scripts/eval_baseline.py --zero-shot

    # Evaluate DAPT'd KoineFormer:
    python scripts/eval_baseline.py --dapt-checkpoint outputs/dapt/final

    # Full comparison (both):
    python scripts/eval_baseline.py --dapt-checkpoint outputs/dapt/final --zero-shot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._cli_utils import detect_device  # noqa: E402
from transformers import AutoTokenizer  # type: ignore[import-untyped]  # noqa: E402

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.evaluation import evaluate_lemmatization, evaluate_pos_tagging  # noqa: E402
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the baseline evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Paper A baseline evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Root data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"),
                        help="Directory for evaluation results")
    parser.add_argument("--dapt-checkpoint", type=Path, default=None,
                        help="Path to DAPT LoRA adapters")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Evaluate zero-shot GreTa (no DAPT)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device")
    return parser


def main() -> int:
    """Run zero-shot vs DAPT baseline evaluation. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or detect_device()

    processed = data_dir / "processed"
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet",
        processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json",
        splits_path=processed / "splits.json",
    )

    results: dict[str, dict[str, float]] = {}

    if args.zero_shot:
        _LOG.info("=== ZERO-SHOT GRETA ===")
        model = KoineFormer.from_pretrained(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model.model_id)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.model.resize_token_embeddings(len(tokenizer))

        pos_result = evaluate_pos_tagging(model, corpus, tokenizer, split="test", device=device)
        lemma_result = evaluate_lemmatization(model, corpus, tokenizer, split="test", device=device)
        results["GreTa (zero-shot)"] = {
            "model": pos_result.model_name,
            "pos_accuracy": pos_result.value,
            "lemma_accuracy": lemma_result.value,
            "n_pos_samples": pos_result.n_samples,
            "n_lemma_samples": lemma_result.n_samples,
        }
        _LOG.info(f"zero-shot POS: {pos_result.value:.2%}  |  Lemma: {lemma_result.value:.2%}")

    if args.dapt_checkpoint:
        _LOG.info("=== DAPT KOINEFORMER ===")
        model = KoineFormer.from_pretrained(device=device)
        model.load_adapters(args.dapt_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model.model_id)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.model.resize_token_embeddings(len(tokenizer))

        pos_result = evaluate_pos_tagging(model, corpus, tokenizer, split="test", device=device)
        lemma_result = evaluate_lemmatization(model, corpus, tokenizer, split="test", device=device)
        results["KoineFormer (DAPT)"] = {
            "model": pos_result.model_name,
            "pos_accuracy": pos_result.value,
            "lemma_accuracy": lemma_result.value,
            "n_pos_samples": pos_result.n_samples,
            "n_lemma_samples": lemma_result.n_samples,
        }
        _LOG.info(f"DAPT POS: {pos_result.value:.2%}  |  Lemma: {lemma_result.value:.2%}")

    # Write results
    results_path = output_dir / "baseline_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _LOG.info("results saved", extra={"path": str(results_path)})

    # Print comparison table
    if len(results) == 2:
        zs_pos = results["GreTa (zero-shot)"]["pos_accuracy"]
        dapt_pos = results["KoineFormer (DAPT)"]["pos_accuracy"]
        zs_lemma = results["GreTa (zero-shot)"]["lemma_accuracy"]
        dapt_lemma = results["KoineFormer (DAPT)"]["lemma_accuracy"]
        pos_delta = dapt_pos - zs_pos
        lemma_delta = dapt_lemma - zs_lemma
        print(f"\n  {'Model':<25s} {'POS':>8s}  {'Lemma':>8s}")
        print(f"  {'─'*25} {'─'*8}  {'─'*8}")
        print(f"  {'GreTa (zero-shot)':<25s} {zs_pos:>7.2%}  {zs_lemma:>7.2%}")
        print(f"  {'KoineFormer (DAPT)':<25s} {dapt_pos:>7.2%}  {dapt_lemma:>7.2%}")
        print(f"  {'Δ':<25s} {pos_delta:>+7.2%}  {lemma_delta:>+7.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
