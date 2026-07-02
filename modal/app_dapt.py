"""Modal deployment for KoineFormer DAPT training.

Supports detached cloud training — runs on Modal's A10G GPU, survives
laptop sleep, checkpoints saved to persistent volume.

Usage:
    # Step 1: Upload data to Modal volume (once):
    modal run modal/app_dapt.py::upload_data

    # Step 2: Start training (detached — runs in cloud even if laptop sleeps):
    modal run modal/app_dapt.py::start_training

    # Step 3: Monitor progress (from any machine, any time):
    modal app logs synoptiq-dapt

    # Step 4: Download checkpoints when done:
    modal volume get synoptiq-outputs outputs/dapt/ models/koineformer/dapt/

    # Or do everything in one shot:
    modal run modal/app_dapt.py::upload_data && modal run modal/app_dapt.py::start_training
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import modal  # type: ignore[import-untyped]
except ImportError:
    modal = None  # type: ignore[assignment]

# ── Constants ────────────────────────────────────────────────────────────────

DATA_VOLUME = "synoptiq-data"
OUTPUT_VOLUME = "synoptiq-outputs"
IMAGE_PYTHON = "3.12"
GPU_TYPE = "A10G"
TIMEOUT_SECONDS = 86_400  # 24 hours

_REQUIREMENTS = [
    "torch>=2.6.0",
    "transformers>=4.51.0",
    "peft>=0.14.0",
    "safetensors>=0.4.0",
    "sentencepiece>=0.2.0",
]

# ── Image ───────────────────────────────────────────────────────────────────


def _build_image() -> Any:  # noqa: F821
    """Build the Modal container image with SynoptiQ installed."""
    if modal is None:
        msg = "Modal not installed — run with `modal run`"
        raise RuntimeError(msg)

    image = modal.Image.debian_slim(python_version=IMAGE_PYTHON)
    for req in _REQUIREMENTS:
        image = image.pip_install(req)
    # Also install core SynoptiQ deps that aren't in the GPU list.
    image = image.pip_install("pandas", "pyarrow", "biopython", "pyyaml", "tqdm", "numpy")
    # Copy the project source into the image and install it.
    image = image.add_local_dir("synoptiq", "/app/synoptiq", copy=True)
    image = image.add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
    image = image.add_local_file("README.md", "/app/README.md", copy=True)
    image = image.run_commands("pip install /app/")
    return image


# ── App ─────────────────────────────────────────────────────────────────────

app = modal.App("synoptiq-dapt") if modal is not None else None


def _get_volumes() -> tuple[Any, Any]:  # noqa: F821
    """Get or create Modal volumes."""
    data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
    return data_vol, output_vol


# ── Step 1: Upload data (runs locally) ──────────────────────────────────────


@app.local_entrypoint()  # type: ignore[misc]
def upload_data() -> None:
    """Upload local data/raw/ to the Modal synoptiq-data volume.

    Runs on YOUR machine — copies files from the local filesystem into
    the cloud volume.  Run once before training.  Skips files already
    present.
    """
    local_raw = Path("data/raw")
    if not local_raw.exists():
        print(f"ERROR: {local_raw} not found. Run prepare_data.py first.")
        sys.exit(1)

    # Count local files
    local_files = sum(1 for _ in local_raw.rglob("*") if _.is_file())
    print(f"Uploading {local_files} files from {local_raw} ...")

    # Use Modal CLI to batch-upload files into the volume.
    # `modal volume put` handles large directories efficiently.
    import subprocess
    result = subprocess.run(
        [
            "modal", "volume", "put", DATA_VOLUME,
            str(local_raw), "/raw",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Upload failed: {result.stderr}")
        sys.exit(1)

    print(f"Upload complete. Volume: {DATA_VOLUME} (mounted at /data/raw)")


# ── Step 2: Train (detached) ───────────────────────────────────────────────


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
    },
    timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def start_training() -> None:
    """Run the full DAPT training loop on Modal GPU.

    Logs progress to stdout (streamable via `modal app logs synoptiq-dapt`).
    Checkpoints saved to the synoptiq-outputs volume.
    Training survives laptop sleep — runs entirely in the cloud.
    """
    from synoptiq.models.koineformer import KoineFormer
    from synoptiq.training.dapt import DAPTConfig, DAPTTrainer
    from transformers import AutoTokenizer

    data_dir = Path("/data/raw")
    output_dir = Path("/outputs/dapt")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    print(f"Loading KoineFormer on {GPU_TYPE}...")
    model = KoineFormer.from_pretrained(device=device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Check for existing checkpoint
    ckpt_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
    )
    if ckpt_dirs:
        latest = ckpt_dirs[-1]
        step = latest.name.split("-")[1]
        print(f"Found checkpoint at step {step} — will auto-resume")

    config = DAPTConfig(
        batch_size=8,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=20_000,
        val_steps=500,
        save_steps=2_000,
        grad_accum_steps=1,
        max_length=512,
        use_amp=True,
        output_dir=output_dir,
    )

    # Get the output volume for crash-safe commits
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME) if modal is not None else None

    print(f"Starting DAPT: {config.max_steps} steps, batch={config.batch_size}, "
          f"seq_len={config.max_length}, AMP={config.use_amp}")
    print(f"Data: /data/raw  |  Checkpoints: /outputs/dapt")
    print(f"Auto-resume: yes  |  Volume commits: yes")
    print(f"Monitor: modal app logs synoptiq-dapt")
    print(f"{'='*50}")

    trainer = DAPTTrainer(model, data_dir, tokenizer, config, device=device)
    history = trainer.run(
        resume=True,
        commit_volume=True,
        volume=output_vol,
    )

    final_loss = history["loss"][-1]
    best_loss = min(history["loss"])
    print(f"\n{'='*50}")
    print(f"DAPT COMPLETE")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Checkpoints: /outputs/dapt/")
    print(f"  Download: modal volume get {OUTPUT_VOLUME} outputs/dapt/ models/")
    print(f"{'='*50}")


# ── Step 3: Ablation (LoRA vs full fine-tune) ──────────────────────────────


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
    },
    timeout=3600,
) if modal is not None else None
def run_ablation(n_steps: int = 2_000) -> None:
    """Run LoRA vs full fine-tune ablation on Modal GPU.

    Trains both variants for *n_steps* and saves loss curves to the output
    volume.  Full FT typically collapses after 1-2K steps on 1M-token data.
    """
    import json

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    from synoptiq.models.koineformer import KoineFormer
    from synoptiq.training.dapt import DAPTIterableDataset

    data_dir = Path("/data/raw")
    output_dir = Path("/outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _train(model_obj, label, steps):
        ds = DAPTIterableDataset(data_dir, tokenizer, max_length=512)
        it = iter(ds)
        opt = AdamW(model_obj.parameters(), lr=1e-4)
        losses = []
        model_obj.train()
        for s in range(1, steps + 1):
            sample = next(it)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            mask = torch.ones_like(input_ids)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model_obj(input_ids=input_ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())
            if s % 200 == 0:
                print(f"  [{label}] step {s}/{steps}: loss={out.loss.item():.4f}")
        return losses

    # Variant 1: LoRA
    print("=== LoRA DAPT ===")
    lora_model = KoineFormer.from_pretrained(device=device)
    lora_model.enable_dapt()
    peft_m = lora_model.model
    peft_m.resize_token_embeddings(len(tokenizer))
    lora_losses = _train(peft_m, "LoRA", n_steps)

    # Variant 2: Full fine-tune (raw GreTa)
    print("=== Full Fine-Tune ===")
    raw = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa", torch_dtype=torch.float32).to(device)
    if hasattr(raw.config, "tie_word_embeddings"):
        raw.config.tie_word_embeddings = False
    raw.resize_token_embeddings(len(tokenizer))
    ft_losses = _train(raw, "Full FT", n_steps)

    # Report
    lora_end = lora_losses[-1]
    ft_end = ft_losses[-1]
    print(f"\n{'='*50}")
    print(f"  LoRA final loss: {lora_end:.4f}")
    print(f"  Full FT final loss: {ft_end:.4f}")
    print(f"  {'✓ LoRA wins' if lora_end < ft_end else 'Full FT lower loss'}")

    results = {
        "steps": n_steps,
        "lora_losses": lora_losses,
        "lora_final": lora_end,
        "fullft_losses": ft_losses,
        "fullft_final": ft_end,
        "verdict": "lora_wins" if lora_end < ft_end else "fullft_lower",
    }
    (output_dir / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results: /outputs/ablation/ablation_results.json")


# ── Local entrypoint ───────────────────────────────────────────────────────


@app.local_entrypoint()  # type: ignore[misc]
def main() -> None:
    """Local entry point — prints usage instructions."""
    print("""
    SynoptiQ DAPT — Modal GPU Training
    ==================================

    Step 1 — Upload data (once, idempotent):
      modal run modal/app_dapt.py::upload_data

    Step 2 — Start training (auto-resumes from checkpoint):
      modal run modal/app_dapt.py::start_training

    Step 3 — Monitor progress (live logs):
      modal app logs synoptiq-dapt

    Step 4 — Download trained model:
      modal volume get synoptiq-outputs outputs/dapt/ models/koineformer/dapt/
    """)


if __name__ == "__main__":
    main()
