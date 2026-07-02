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
    image = image.run_commands("pip install /app/")
    return image


# ── App ─────────────────────────────────────────────────────────────────────

app = modal.App("synoptiq-dapt") if modal is not None else None


def _get_volumes() -> tuple[Any, Any]:  # noqa: F821
    """Get or create Modal volumes."""
    data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
    return data_vol, output_vol


# ── Step 1: Upload data ────────────────────────────────────────────────────


@app.function(  # type: ignore[misc]
    image=_build_image(),
    volumes={"/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)},
    timeout=600,
) if modal is not None else None
def upload_data() -> None:
    """Upload local data/raw/ to the Modal synoptiq-data volume.

    Run once before training.  Skips files already present.
    """
    import os
    import subprocess

    local_raw = Path("data/raw")
    if not local_raw.exists():
        print(f"ERROR: {local_raw} not found. Run prepare_data.py first.")
        sys.exit(1)

    # Count local files
    local_files = sum(1 for _ in local_raw.rglob("*") if _.is_file())
    print(f"Uploading {local_files} files from {local_raw} to /data/raw ...")

    # Use os.walk + open() to copy files into the volume.
    # Modal volumes are writable at the mount path.
    dest_root = Path("/data/raw")
    dest_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for filepath in local_raw.rglob("*"):
        if filepath.is_file():
            rel = filepath.relative_to(local_raw)
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(filepath.read_bytes())
                copied += 1

    print(f"Upload complete: {copied} new files copied to Modal volume.")
    print(f"Volume: {DATA_VOLUME} mounted at /data")


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
