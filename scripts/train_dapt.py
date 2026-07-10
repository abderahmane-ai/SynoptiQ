"""Phase 2A: KoineFormer DAPT training CLI.

Trains LoRA adapters on GreTa using T5 span corruption on Koine Greek
text with a Classical Greek replay buffer.

Usage:
    # Full DAPT training (20K steps, ~10-14 hours on A10G):
    python scripts/train_dapt.py

    # Quick smoke test (100 steps on CPU):
    python scripts/train_dapt.py --smoke-test

    # Resume from checkpoint:
    python scripts/train_dapt.py --resume outputs/dapt/step-2000

    # Custom hyperparameters:
    python scripts/train_dapt.py --lr 5e-5 --max-steps 10000 --batch-size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._cli_utils import detect_device  # noqa: E402
from transformers import AutoTokenizer  # type: ignore[import-untyped]  # noqa: E402

from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.training.dapt import DAPTConfig, DAPTTrainer  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the DAPT training CLI."""
    parser = argparse.ArgumentParser(
        description="KoineFormer DAPT: domain-adaptive pre-training on Koine Greek",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"),
                        help="Root directory with downloaded raw corpora")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dapt"),
                        help="Directory for adapter checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate for AdamW")
    parser.add_argument("--max-steps", type=int, default=20_000,
                        help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear warmup steps")
    parser.add_argument("--val-steps", type=int, default=500,
                        help="Validation interval (steps)")
    parser.add_argument("--save-steps", type=int, default=2_000,
                        help="Checkpoint interval (steps)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum token sequence length")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (auto-detected if not set)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a checkpoint directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 100 steps on CPU to verify the pipeline")
    parser.add_argument("--exclude-books", nargs="*", default=None,
                        metavar="BOOK",
                        help="Canonical book names to hold out of DAPT text "
                             "(e.g. Matthew Mark Luke)")
    parser.add_argument("--no-synoptics", action="store_true",
                        help="Build KoineFormer-NS: exclude Matthew/Mark/Luke and "
                             "default the output dir to outputs/dapt_ns")
    return parser


def main() -> int:
    """Parse arguments and run the DAPT training loop. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.smoke_test:
        _LOG.info("smoke test mode — 100 steps on CPU")
        args.max_steps = 100
        args.val_steps = 50
        args.save_steps = 50
        args.batch_size = 2
        args.lr = 5e-4  # higher LR for quick convergence
        args.device = "cpu"

    # Decontamination: --no-synoptics is shorthand for excluding the three gospels
    # and writing to a distinct output dir so the NS adapters never clobber KoineFormer.
    exclude_books: tuple[str, ...] = tuple(args.exclude_books or ())
    if args.no_synoptics:
        exclude_books = ("Matthew", "Mark", "Luke")
        if args.output_dir == Path("outputs/dapt"):
            args.output_dir = Path("outputs/dapt_ns")

    device = args.device or detect_device()
    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    _LOG.info(
        "KoineFormer DAPT",
        extra={
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "device": device,
            "max_steps": args.max_steps,
            "lr": args.lr,
        },
    )

    # ── Load model ──────────────────────────────────────────────────────
    _LOG.info("loading KoineFormer")
    model = KoineFormer.from_pretrained(device=device)

    if args.resume:
        _LOG.info("resuming from checkpoint", extra={"path": str(args.resume)})
        model.load_adapters(args.resume)

    # ── Load tokenizer ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Configure training ──────────────────────────────────────────────
    config = DAPTConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        val_steps=args.val_steps,
        save_steps=args.save_steps,
        grad_accum_steps=args.grad_accum,
        max_length=args.max_length,
        output_dir=output_dir,
        exclude_books=exclude_books,
    )
    if exclude_books:
        _LOG.info("decontaminated DAPT (KoineFormer-NS)", extra={"excluded": exclude_books})

    # ── Train ───────────────────────────────────────────────────────────
    trainer = DAPTTrainer(model, data_dir, tokenizer, config, device=device)
    history = trainer.run()

    # ── Report ──────────────────────────────────────────────────────────
    final_loss = history["loss"][-1]
    best_loss = min(history["loss"])
    _LOG.info(
        "DAPT complete",
        extra={
            "final_loss": f"{final_loss:.4f}",
            "best_loss": f"{best_loss:.4f}",
            "n_steps": len(history["loss"]),
            "output_dir": str(output_dir),
        },
    )

    if args.smoke_test:
        start_loss = history["loss"][0]
        end_loss = history["loss"][-1]
        pct = (start_loss - end_loss) / start_loss * 100
        if pct > 30:
            _LOG.info(f"✓ SMOKE TEST PASSED — loss reduced {pct:.0f}%")
        else:
            _LOG.error(
                f"✗ SMOKE TEST FAILED — loss only reduced {pct:.0f}% "
                f"({start_loss:.2f} → {end_loss:.2f})"
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
