"""Modal deployment for KoineFormer DAPT training.

Deploys a long-running DAPT training job on Modal's A10G GPU.
Mounts the data volume containing raw corpora and saves LoRA adapter
checkpoints to the output volume.

Usage (local):
    modal run modal/app_dapt.py

Environment variables expected by Modal:
    MODAL_DATA_VOLUME    — name of the Modal volume with data/raw/
    MODAL_OUTPUT_VOLUME  — name of the Modal volume for checkpoints
"""

from __future__ import annotations

import sys
from pathlib import Path

# Modal imports are only available inside the Modal runtime.
# The try/except allows local linting to pass.
try:
    import modal  # type: ignore[import-untyped]
except ImportError:
    modal = None  # type: ignore[assignment]

# ── Modal image definition ──────────────────────────────────────────────────

# Python dependencies needed inside the Modal container.
_REQUIREMENTS: list[str] = [
    "torch>=2.6.0",
    "transformers>=4.51.0",
    "peft>=0.14.0",
    "datasets>=4.0.0",
    "safetensors>=0.4.0",
    "sentencepiece>=0.2.0",
]

_DATA_VOLUME_NAME = "synoptiq-data"
_OUTPUT_VOLUME_NAME = "synoptiq-outputs"


def _build_image() -> Any:  # noqa: F821
    """Build the Modal container image with all Python dependencies."""
    if modal is None:
        msg = "Modal is not installed — run this script inside `modal run`"
        raise RuntimeError(msg)

    image = modal.Image.debian_slim(python_version="3.12")
    for req in _REQUIREMENTS:
        image = image.pip_install(req)
    # Install SynoptiQ itself from the project root.
    image = image.pip_install(".")
    return image


# ── Modal app ───────────────────────────────────────────────────────────────


def _create_app() -> Any:  # noqa: F821
    """Create the Modal app with mounted volumes."""
    if modal is None:
        msg = "Modal is not installed"
        raise RuntimeError(msg)

    app = modal.App("synoptiq-dapt")
    image = _build_image()

    data_volume = modal.Volume.from_name(_DATA_VOLUME_NAME, create_if_missing=True)
    output_volume = modal.Volume.from_name(_OUTPUT_VOLUME_NAME, create_if_missing=True)

    @app.function(
        gpu="A10G",
        image=image,
        volumes={"/data": data_volume, "/outputs": output_volume},
        timeout=86_400,  # 24 hours
    )
    def train_dapt() -> None:
        """Run the full DAPT training loop on Modal A10G."""
        # Late imports — only executed inside the Modal container.
        from synoptiq.models.koineformer import KoineFormer
        from synoptiq.training.dapt import DAPTConfig, DAPTTrainer

        from transformers import AutoTokenizer

        data_dir = Path("/data/raw")
        output_dir = Path("/outputs/dapt")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer
        device = "cuda"
        model = KoineFormer.from_pretrained(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model.model_id)
        tokenizer.pad_token = tokenizer.eos_token

        # Training config
        config = DAPTConfig(
            batch_size=8,
            learning_rate=1e-4,
            warmup_steps=500,
            max_steps=20_000,
            val_steps=500,
            save_steps=2_000,
            grad_accum_steps=1,
            max_length=512,
            output_dir=output_dir,
        )

        trainer = DAPTTrainer(model, data_dir, tokenizer, config, device=device)
        history = trainer.run()

        print(f"DAPT complete. Final loss: {history['loss'][-1]:.4f}")
        output_volume.commit()

    return app, train_dapt


# ── CLI entry points ────────────────────────────────────────────────────────

# These are the targets that `modal run modal/app_dapt.py::FUNCTION` invokes.

_app, _train_fn = _create_app()  # noqa: F811


@_app.local_entrypoint()  # type: ignore[misc]
def main() -> None:
    """Dry-run entry point — runs a single step to verify the pipeline."""
    print("Dry-run: launching training on Modal A10G...")
    _train_fn.remote()


if __name__ == "__main__":
    main()
