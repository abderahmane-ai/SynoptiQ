"""Modal deployment for Phase-5 M2 training (redaction operators + Fusion-in-Decoder).

Runs the tested training scripts on an A10G GPU against the volume-resident corpus and
the decontaminated KoineFormer-NS adapters, writing results to the outputs volume.

Prerequisites (already done in earlier steps):
    modal run modal/app_dapt.py::upload_data          # corpus on /data
    modal run modal/app_dapt.py::start_training_ns    # NS adapters on /outputs/dapt_ns

Usage:
    modal run modal/app_fid.py::train_redactors       # R_Lk / R_Mt / G_Mt / G_Lk, fold 0
    modal run modal/app_fid.py::train_fid             # Track A: Mt+Lk -> Mark, fold 0
    modal app logs synoptiq-fid
    modal volume get synoptiq-outputs study/ outputs/study/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import modal  # type: ignore[import-untyped]
except ImportError:
    modal = None  # type: ignore[assignment]

DATA_VOLUME = "synoptiq-data"
OUTPUT_VOLUME = "synoptiq-outputs"
GPU_TYPE = "A10G"
TIMEOUT_SECONDS = 86_400

_REQUIREMENTS = [
    "torch>=2.6.0", "transformers>=4.51.0", "peft>=0.14.0",
    "safetensors>=0.4.0", "sentencepiece>=0.2.0",
]


def _build_image() -> Any:
    if modal is None:
        msg = "Modal not installed — run with `modal run`"
        raise RuntimeError(msg)
    image = modal.Image.debian_slim(python_version="3.12")
    for req in _REQUIREMENTS:
        image = image.pip_install(req)
    image = image.pip_install("pandas", "pyarrow", "biopython", "pyyaml", "tqdm", "numpy")
    image = image.add_local_dir("synoptiq", "/app/synoptiq", copy=True)
    image = image.add_local_dir("scripts", "/app/scripts", copy=True)
    image = image.add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
    image = image.add_local_file("README.md", "/app/README.md", copy=True)
    image = image.run_commands("pip install /app/")
    return image


app = modal.App("synoptiq-fid") if modal is not None else None

_VOLUMES = {
    "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
} if modal is not None else {}

_COMMON = dict(  # noqa: C408
    tokens="/data/processed/tokens.parquet",
    pericopes="/data/processed/pericopes.parquet",
    ns_adapters="/outputs/dapt_ns/final",
)


def _commit() -> None:
    if modal is not None:
        modal.Volume.from_name(OUTPUT_VOLUME).commit()


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE, image=_build_image(), volumes=_VOLUMES, timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def train_redactors(fold: int = 0, epochs: int = 3) -> None:
    """Train the four redaction operators on the given held-out fold."""
    import sys

    sys.path.insert(0, "/app")
    from scripts.train_redactors import main

    main([
        "--tokens", _COMMON["tokens"], "--pericopes", _COMMON["pericopes"],
        "--init-adapters", _COMMON["ns_adapters"], "--fold", str(fold),
        "--epochs", str(epochs), "--device", "cuda", "--out", "/outputs/study/redactors",
    ])
    _commit()
    print("Download: modal volume get synoptiq-outputs study/redactors/ outputs/study/redactors/")


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE, image=_build_image(), volumes=_VOLUMES, timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def train_fid(fold: int = 0, epochs: int = 3) -> None:
    """Train the Fusion-in-Decoder (Track A: Matthew+Luke -> Mark) on the fold."""
    import sys

    sys.path.insert(0, "/app")
    from scripts.train_fid import main

    main([
        "--tokens", _COMMON["tokens"], "--pericopes", _COMMON["pericopes"],
        "--init-adapters", _COMMON["ns_adapters"], "--fold", str(fold),
        "--epochs", str(epochs), "--device", "cuda", "--out", "/outputs/study/fid",
    ])
    _commit()
    print("Download: modal volume get synoptiq-outputs study/fid/ outputs/study/fid/")


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE, image=_build_image(), volumes=_VOLUMES, timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def train_fid_mai(fold: int = 0, epochs: int = 8) -> None:
    """Train the E2 FiD (source-dropout, (Mark,Matthew) -> Luke) on the fold (M4)."""
    import sys

    sys.path.insert(0, "/app")
    from scripts.train_fid import main

    main([
        "--tokens", _COMMON["tokens"], "--pericopes", _COMMON["pericopes"],
        "--init-adapters", _COMMON["ns_adapters"], "--fold", str(fold),
        "--epochs", str(epochs), "--device", "cuda", "--out", "/outputs/study/fid_mai",
        "--witnesses", "Mark", "Matthew", "--target", "Luke",
    ])
    _commit()
    print("Then: modal run modal/app_fid.py::run_mai --fold", fold)


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE, image=_build_image(), volumes=_VOLUMES, timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def run_mai(fold: int = 0) -> None:
    """Run the E2 minor-agreement verdict for the fold (needs train_fid_mai first)."""
    import sys

    sys.path.insert(0, "/app")
    from scripts.run_mai_test import main

    main([
        "--tokens", _COMMON["tokens"], "--pericopes", _COMMON["pericopes"],
        "--fid-adapters", f"/outputs/study/fid_mai/fold{fold}", "--fold", str(fold),
        "--device", "cuda", "--out", "/outputs/study/mai",
    ])
    _commit()
    print("Download: modal volume get synoptiq-outputs study/mai/ outputs/study/mai/")


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE, image=_build_image(), volumes=_VOLUMES, timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def mai_cv(epochs: int = 8) -> None:
    """Full E2 cross-validation: train + score all 5 folds, then pool the verdict (M4)."""
    import sys

    sys.path.insert(0, "/app")
    from scripts.pool_mai import main as pool_main
    from scripts.run_mai_test import main as mai_main
    from scripts.train_fid import main as train_main

    for fold in range(5):
        print(f"\n===== FOLD {fold} =====")
        train_main([
            "--tokens", _COMMON["tokens"], "--pericopes", _COMMON["pericopes"],
            "--init-adapters", _COMMON["ns_adapters"], "--fold", str(fold),
            "--epochs", str(epochs), "--device", "cuda", "--out", "/outputs/study/fid_mai",
            "--witnesses", "Mark", "Matthew", "--target", "Luke",
        ])
        mai_main([
            "--tokens", _COMMON["tokens"], "--pericopes", _COMMON["pericopes"],
            "--fid-adapters", f"/outputs/study/fid_mai/fold{fold}", "--fold", str(fold),
            "--device", "cuda", "--out", "/outputs/study/mai",
        ])
        _commit()
    pool_main(["--glob", "/outputs/study/mai/mai_fold*.json",
               "--out", "/outputs/study/mai/mai_pooled.json"])
    _commit()
    print("Download: modal volume get synoptiq-outputs study/mai/ outputs/study/mai/")


if __name__ == "__main__":
    _ = Path  # keep Path import used for local linting parity
