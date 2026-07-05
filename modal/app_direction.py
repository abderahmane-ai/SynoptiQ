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
    import modal  # type: ignore[import-untyped]
except ImportError:
    modal = None  # type: ignore[assignment]

GPU_TYPE = "t4"  # T4 is sufficient — no 220M param updates, just cross-attn heads
TIMEOUT_SECONDS = 3600
DATA_VOLUME = "synoptiq-data"
OUTPUT_VOLUME = "synoptiq-outputs"

app = modal.App("synoptiq-direction") if modal is not None else None


def _build_image() -> object:
    """Build Modal container image with SynoptiQ + dependencies."""
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch>=2.6",
            "transformers>=4.51",
            "peft>=0.15",
            "datasets",
            "biopython",
            "huggingface_hub",
            "sentencepiece",
        )
    )
    # Copy project files into container
    image = image.add_local_dir("synoptiq", "/app/synoptiq", copy=True)
    image = image.add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
    image = image.add_local_file("README.md", "/app/README.md", copy=True)
    image = image.run_commands("cd /app && pip install -e . --no-deps")
    return image


# ── Step 1: Upload data (local → Modal volumes) ────────────────────────────────


@app.local_entrypoint()  # type: ignore[misc]
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
        print("Uploaded DAPT adapters")


# ── Step 2: Train direction scorer ─────────────────────────────────────────────


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
    },
    timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def start_training(
    max_steps: int = 5_000,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
) -> None:
    """Train the direction scorer on Modal GPU."""
    import json

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

    dapt_path = Path("/outputs/dapt/final")
    if dapt_path.exists():
        try:
            koine.load_adapters(dapt_path)
            print("Loaded DAPT adapters")
        except Exception as e:
            print(f"DAPT adapters not available ({e}) — using zero-shot GreTa")

    koine.model.resize_token_embeddings(len(tokenizer))
    encoder = koine.model.base_model.encoder
    scorer = DirectionScorer(encoder, DirectionScorerConfig())

    # ── Training config ──────────────────────────────────────────────
    train_cfg = DirectionTrainingConfig(
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        val_steps=250,
        save_steps=1_000,
    )

    # ── Datasets ─────────────────────────────────────────────────────
    dataset_kwargs = {
        "corpus": corpus, "tokenizer": tokenizer,
        "max_length": 512, "min_aligned_tokens": 5,
        "use_scribal_noise": True,
    }
    train_ds = DirectionDataset(split="train", **dataset_kwargs)
    val_ds = DirectionDataset(split="val", **dataset_kwargs)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Train ────────────────────────────────────────────────────────
    trainer = DirectionTrainer(scorer, train_ds, val_ds, train_cfg, device="cuda")
    trainer.train()

    # ── Final eval ───────────────────────────────────────────────────
    print("\n=== Final Test Evaluation ===")
    scorer.eval()
    test_ds = DirectionDataset(
        corpus, tokenizer, split="test",
        max_length=512, min_aligned_tokens=5,
        use_scribal_noise=False,
    )
    correct = total = 0
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
        correct += int(pred == sample["direction_label"].item())
        total += 1

    acc = correct / max(total, 1)
    print(f"Test accuracy: {acc:.2%} ({correct}/{total})")
    (output_dir / "test_results.json").write_text(
        json.dumps({"accuracy": acc, "n_samples": total}, indent=2),
    )


# ── Step 3: Smoke test ─────────────────────────────────────────────────────────


@app.function(  # type: ignore[misc]
    gpu="t4",
    image=_build_image(),
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
    },
    timeout=600,
) if modal is not None else None
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

    dapt_path = Path("/outputs/dapt/final")
    if dapt_path.exists():
        koine.load_adapters(dapt_path)
        print("Loaded DAPT adapters")

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
    print(f"Smoke test done — final loss: {history['train_loss'][-1]:.4f}")
