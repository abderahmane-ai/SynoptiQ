"""Phase 2B: Multi-task LoRA fine-tuning CLI.

Trains task-specific heads on the frozen KoineFormer encoder using the
SynoptiQ Corpus annotations (POS, dependency, lemmatisation, pericope).

Usage:
    # Full multi-task training:
    python scripts/train_multitask.py --dapt-checkpoint outputs/dapt/final

    # Quick smoke test:
    python scripts/train_multitask.py --smoke-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # type: ignore[import-untyped]  # noqa: E402

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.training.multitask import MultiTaskTrainer, MultiTaskTrainingConfig  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KoineFormer multi-task LoRA fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Root data directory with processed/ subdir")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multitask"),
                        help="Directory for multi-task head checkpoints")
    parser.add_argument("--dapt-checkpoint", type=Path, default=None,
                        help="Path to DAPT LoRA adapters (loads DAPT'd KoineFormer)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (auto-detected)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 1 epoch with 100 samples to verify pipeline")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.smoke_test:
        _LOG.info("smoke test mode")
        args.epochs = 1

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    device = args.device or _detect_device()

    # Load corpus
    processed = data_dir / "processed"
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet",
        processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json",
        splits_path=processed / "splits.json",
    )

    # Load KoineFormer with DAPT adapters if available
    _LOG.info("loading KoineFormer")
    model = KoineFormer.from_pretrained(device=device)
    if args.dapt_checkpoint:
        model.load_adapters(args.dapt_checkpoint)
        _LOG.info("DAPT adapters loaded", extra={"path": str(args.dapt_checkpoint)})

    tokenizer = AutoTokenizer.from_pretrained(model.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = MultiTaskTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=output_dir,
    )

    trainer = MultiTaskTrainer(model, corpus, tokenizer, config, device=device)
    metrics = trainer.run()

    _LOG.info("multi-task complete", extra={"metrics": str(metrics)})
    return 0


def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    sys.exit(main())
