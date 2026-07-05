"""Phase 3: Train the Direction Scorer.

Detects literary copying direction between parallel Gospel passages.

Usage:
    # Smoke test (100 steps, CPU, quick check):
    python scripts/train_direction.py --smoke-test

    # Full training (5,000 steps, GPU):
    python scripts/train_direction.py

    # Evaluate only (load checkpoint):
    python scripts/train_direction.py --eval-only --checkpoint outputs/direction/final
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # type: ignore[import-untyped]

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.models.direction import DirectionScorer, DirectionScorerConfig  # noqa: E402
from synoptiq.training.direction import (  # noqa: E402
    DirectionDataset,
    DirectionTrainer,
    DirectionTrainingConfig,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SynoptiQ Direction Scorer (Phase 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Root data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/direction"),
                        help="Output for checkpoints and logs")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 100-step CPU check")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate only (no training)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint to load for eval or resume")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (auto-detect if unset)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or _detect_device()

    # ── Load corpus ────────────────────────────────────────────────────
    processed = data_dir / "processed"
    corpus = Corpus.from_parquet(
        processed / "tokens.parquet",
        processed / "pericopes.parquet",
        alignments_path=processed / "alignments.json",
        splits_path=processed / "splits.json",
    )

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ── Build model ────────────────────────────────────────────────────
    from synoptiq.models.koineformer import KoineFormer

    koine = KoineFormer.from_pretrained(device=device)
    # Try loading DAPT adapters if available
    dapt_path = Path("models/koineformer/dapt/final")
    if dapt_path.exists():
        try:
            koine.load_adapters(dapt_path)
        except Exception:
            pass  # Use zero-shot GreTa
    koine.model.resize_token_embeddings(len(tokenizer))

    encoder = koine.model.base_model.encoder
    scorer_config = DirectionScorerConfig()
    scorer = DirectionScorer(encoder, scorer_config)

    if args.checkpoint:
        ckpt = args.checkpoint / "model.pt"
        if ckpt.exists():
            scorer.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    # ── Build datasets ─────────────────────────────────────────────────
    dataset_kwargs = {
        "corpus": corpus,
        "tokenizer": tokenizer,
        "max_length": 512,
        "min_aligned_tokens": 5,
        "use_scribal_noise": not args.eval_only,
    }

    if args.smoke_test:
        ds = DirectionDataset(split="train", **dataset_kwargs)
        print(f"\n  Smoke test dataset: {len(ds)} samples")
        print(f"  Sample keys: {list(ds[0].keys())}")
        print(f"  input_ids_a shape: {ds[0]['input_ids_a'].shape}")
        print(f"  direction_label: {ds[0]['direction_label']}")

        # Quick forward pass
        scorer = scorer.to(device)
        scorer.train()
        batch = {
            "input_ids_a": ds[0]["input_ids_a"].unsqueeze(0).to(device),
            "attention_mask_a": ds[0]["attention_mask_a"].unsqueeze(0).to(device),
            "input_ids_b": ds[0]["input_ids_b"].unsqueeze(0).to(device),
            "attention_mask_b": ds[0]["attention_mask_b"].unsqueeze(0).to(device),
        }
        output = scorer(**batch)
        print(f"  direction_logits: {output['direction_logits'].shape}")
        print(f"  author_logits_a: {output['author_logits_a'].shape}")
        print(f"  asymmetry_features: {output['asymmetry_features'].shape}")
        print("\n  ✓ Forward pass OK — smoke test passed")

        # Quick training check
        print("\n  Running 100 training steps (CPU)...")
        train_ds = DirectionDataset(split="train", **dataset_kwargs)
        val_ds = DirectionDataset(split="val", **dataset_kwargs)
        train_cfg = DirectionTrainingConfig(
            max_steps=100, val_steps=50, save_steps=100,
            batch_size=4, output_dir=output_dir,
            use_sliding_windows=False,
        )
        trainer = DirectionTrainer(scorer, train_ds, val_ds, train_cfg, device=device)
        history = trainer.train()
        final_loss = history["train_loss"][-1]
        print(f"  Final training loss: {final_loss:.4f}")
        print("  ✓ Training loop OK")
        return 0

    # ── Full training ──────────────────────────────────────────────────
    train_ds = DirectionDataset(split="train", **dataset_kwargs)
    val_ds = DirectionDataset(split="val", **dataset_kwargs)

    print(f"\n  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")

    train_cfg = DirectionTrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=output_dir,
    )
    trainer = DirectionTrainer(scorer, train_ds, val_ds, train_cfg, device=device)
    history = trainer.train()

    # ── Final evaluation ───────────────────────────────────────────────
    print("\n  === Final Evaluation ===")
    _evaluate(scorer, corpus, tokenizer, device, output_dir)

    return 0


def _evaluate(
    scorer: DirectionScorer,
    corpus: Corpus,
    tokenizer: object,
    device: str,
    output_dir: Path,
) -> None:
    """Evaluate on test set and save results."""
    import json

    scorer.eval()
    test_ds = DirectionDataset(
        corpus, tokenizer, split="test",
        max_length=512, min_aligned_tokens=5,
        use_sliding_windows=False, use_scribal_noise=False,
    )

    correct = 0
    total = 0
    results: list[dict] = []

    for i in range(len(test_ds)):
        sample = test_ds[i]
        batch = {
            "input_ids_a": sample["input_ids_a"].unsqueeze(0).to(device),
            "attention_mask_a": sample["attention_mask_a"].unsqueeze(0).to(device),
            "input_ids_b": sample["input_ids_b"].unsqueeze(0).to(device),
            "attention_mask_b": sample["attention_mask_b"].unsqueeze(0).to(device),
        }
        output = scorer(**batch)
        pred = output["direction_logits"].argmax(dim=1).item()
        label = sample["direction_label"].item()
        correct += int(pred == label)
        total += 1

        results.append({
            "prediction": scorer._IDX_TO_DIRECTION[pred],
            "label": scorer._IDX_TO_DIRECTION[label],
            "correct": pred == label,
        })

    accuracy = correct / max(total, 1)
    print(f"  Test accuracy: {accuracy:.2%} ({correct}/{total})")

    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps({
        "accuracy": accuracy,
        "n_samples": total,
        "per_sample": results,
    }, indent=2, ensure_ascii=False))
    print(f"  Results saved: {results_path}")


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    sys.exit(main())
