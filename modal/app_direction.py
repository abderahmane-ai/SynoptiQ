"""Modal deployment for Phase 3: Direction Scorer training.

Trains the cross-attention asymmetry classifier on triple-tradition
pericopes with known Mark→Matthew/Luke copying direction.

Commands:
    # Upload data (once):
    modal run modal/app_direction.py::upload_data

    # Train direction scorer (auto-resumes):
    modal run modal/app_direction.py::start_training

    # Monitor:
    modal app logs synoptiq-direction

    # Smoke test (100 steps, quick check):
    modal run modal/app_direction.py::smoke_test
"""

from __future__ import annotations

from pathlib import Path

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

GPU_TYPE = "t4"  # T4 is sufficient — no 220M param updates, just cross-attn heads
TIMEOUT_SECONDS = 3600  # 1 hour max
DATA_VOLUME = "synoptiq-data"
OUTPUT_VOLUME = "synoptiq-outputs"
APP_NAME = "synoptiq-direction"


def _build_image() -> object:
    """Build Modal container image with SynoptiQ + dependencies."""
    if modal is None:
        raise RuntimeError("modal not installed")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch>=2.6",
            "transformers>=4.51",
            "peft>=0.15",
            "datasets",
            "biopython",
        )
        .pip_install("huggingface_hub")
    )

    # Copy project files into container
    image = image.add_local_dir("synoptiq", "/app/synoptiq")
    image = image.add_local_file("pyproject.toml", "/app/pyproject.toml")
    image = image.add_local_file("README.md", "/app/README.md")

    # Install synoptiq package
    image = image.run_commands(
        "cd /app && pip install -e . --no-deps",
        "pip install sentencepiece",
    )

    return image


# ── Step 1: Upload data (local → Modal volumes) ────────────────────────────────


if modal is not None:

    @modal.local_entrypoint()  # type: ignore[misc]
    def upload_data() -> None:
        """Upload raw + processed data to Modal volumes."""
        import subprocess

        for path, vol in [
            ("data/raw/", DATA_VOLUME),
            ("data/processed/", DATA_VOLUME),
        ]:
            if Path(path).exists():
                subprocess.run(
                    ["modal", "volume", "put", "--force", vol, path, "/data/" + path.split("/")[-2] + "/"],
                    check=True,
                )
                print(f"Uploaded {path} → {vol}:/data/{Path(path).name}")
            else:
                print(f"Skipping {path} — not found")

        # Also upload DAPT adapters if available
        dapt_dir = Path("models/koineformer/dapt/final")
        if dapt_dir.exists():
            subprocess.run(
                ["modal", "volume", "put", "--force", OUTPUT_VOLUME,
                 str(dapt_dir), "/outputs/dapt/final/"],
                check=True,
            )
            print(f"Uploaded DAPT adapters → {OUTPUT_VOLUME}:/outputs/dapt/final/")


# ── Step 2: Train direction scorer ─────────────────────────────────────────────


if modal is not None:

    @modal.function(  # type: ignore[misc]
        gpu=GPU_TYPE,
        image=_build_image(),
        volumes={
            "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
            "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
        },
        timeout=TIMEOUT_SECONDS,
        container_idle_timeout=300,
    )
    def start_training(
        max_steps: int = 5_000,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
    ) -> None:
        """Train the direction scorer on Modal GPU.

        Auto-resumes from latest checkpoint if available.
        """
        import json
        import sys

        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from transformers import AutoTokenizer

        from synoptiq.data.corpus import Corpus
        from synoptiq.models.direction import DirectionScorer, DirectionScorerConfig
        from synoptiq.training.direction import (
            DirectionDataset,
            DirectionTrainer,
            DirectionTrainingConfig,
        )
        from synoptiq.models.koineformer import KoineFormer

        output_dir = Path("/outputs/direction")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Load corpus ──────────────────────────────────────────────────
        processed = Path("/data/processed")
        corpus = Corpus.from_parquet(
            processed / "tokens.parquet",
            processed / "pericopes.parquet",
            alignments_path=processed / "alignments.json",
            splits_path=processed / "splits.json",
        )

        tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # ── Build model ──────────────────────────────────────────────────
        print("Loading KoineFormer encoder...")
        koine = KoineFormer.from_pretrained(device="cuda")

        # Load DAPT adapters from volume
        dapt_path = Path("/outputs/dapt/final")
        if dapt_path.exists():
            try:
                koine.load_adapters(dapt_path)
                print("Loaded DAPT adapters")
            except Exception as e:
                print(f"DAPT adapters not available ({e}) — using zero-shot GreTa")

        koine.model.resize_token_embeddings(len(tokenizer))
        encoder = koine.model.base_model.encoder
        scorer_config = DirectionScorerConfig()
        scorer = DirectionScorer(encoder, scorer_config)

        # ── Training config ──────────────────────────────────────────────
        train_cfg = DirectionTrainingConfig(
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            grl_warmup_steps=1_000,
            grl_lambda_max=1.0,
            val_steps=250,
            save_steps=1_000,
        )

        # ── Datasets ─────────────────────────────────────────────────────
        dataset_kwargs = {
            "corpus": corpus,
            "tokenizer": tokenizer,
            "max_length": 512,
            "min_aligned_tokens": 5,
            "use_scribal_noise": True,
        }

        train_ds = DirectionDataset(split="train", **dataset_kwargs)
        val_ds = DirectionDataset(split="val", **dataset_kwargs)

        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")

        # ── Train ────────────────────────────────────────────────────────
        trainer = DirectionTrainer(scorer, train_ds, val_ds, train_cfg, device="cuda")
        history = trainer.train()

        # ── Commit to volume ─────────────────────────────────────────────
        try:
            import subprocess
            subprocess.run(["modal", "volume", "commit", OUTPUT_VOLUME], check=True)
        except Exception:
            pass

        # ── Final eval on test set ────────────────────────────────────────
        print("\n=== Final Test Evaluation ===")
        scorer.eval()
        test_ds = DirectionDataset(
            corpus, tokenizer, split="test",
            max_length=512, min_aligned_tokens=5,
            use_scribal_noise=False,
        )

        correct = 0
        total = 0
        for i in range(len(test_ds)):
            sample = test_ds[i]
            batch = {
                "input_ids_a": sample["input_ids_a"].unsqueeze(0).to("cuda"),
                "attention_mask_a": sample["attention_mask_a"].unsqueeze(0).to("cuda"),
                "input_ids_b": sample["input_ids_b"].unsqueeze(0).to("cuda"),
                "attention_mask_b": sample["attention_mask_b"].unsqueeze(0).to("cuda"),
            }
            with torch.no_grad():
                output = scorer(**batch)
            pred = output["direction_logits"].argmax(dim=1).item()
            label = sample["direction_label"].item()
            correct += int(pred == label)
            total += 1

        accuracy = correct / max(total, 1)
        print(f"Test accuracy: {accuracy:.2%} ({correct}/{total})")

        results = {"accuracy": accuracy, "n_samples": total}
        (output_dir / "test_results.json").write_text(
            json.dumps(results, indent=2),
        )
        print(f"Results saved: {output_dir / 'test_results.json'}")


# ── Step 3: Smoke test ─────────────────────────────────────────────────────────


if modal is not None:

    @modal.function(  # type: ignore[misc]
        gpu="t4",
        image=_build_image(),
        volumes={
            "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        },
        timeout=600,
        container_idle_timeout=120,
    )
    def smoke_test() -> None:
        """Quick 100-step smoke test on GPU."""
        import torch
        from transformers import AutoTokenizer

        from synoptiq.data.corpus import Corpus
        from synoptiq.models.direction import DirectionScorer, DirectionScorerConfig
        from synoptiq.training.direction import (
            DirectionDataset,
            DirectionTrainer,
            DirectionTrainingConfig,
        )
        from synoptiq.models.koineformer import KoineFormer

        processed = Path("/data/processed")
        corpus = Corpus.from_parquet(
            processed / "tokens.parquet",
            processed / "pericopes.parquet",
            alignments_path=processed / "alignments.json",
            splits_path=processed / "splits.json",
        )

        tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        koine = KoineFormer.from_pretrained(device="cuda")
        koine.model.resize_token_embeddings(len(tokenizer))

        encoder = koine.model.base_model.encoder
        scorer = DirectionScorer(encoder, DirectionScorerConfig()).to("cuda")

        dataset_kwargs = {
            "corpus": corpus, "tokenizer": tokenizer,
            "max_length": 512, "min_aligned_tokens": 5,
            "use_scribal_noise": False,
        }
        train_ds = DirectionDataset(split="train", **dataset_kwargs)
        val_ds = DirectionDataset(split="val", **dataset_kwargs)

        cfg = DirectionTrainingConfig(
            max_steps=100, val_steps=50, save_steps=100,
            batch_size=8, output_dir=Path("/outputs/direction"),
        )
        trainer = DirectionTrainer(scorer, train_ds, val_ds, cfg, device="cuda")
        history = trainer.train()
        print(f"✓ Smoke test complete — final loss: {history['train_loss'][-1]:.4f}")
        print(f"  Val accuracy: {history['val_accuracy'][-1] if history['val_accuracy'] else 'N/A'}")
