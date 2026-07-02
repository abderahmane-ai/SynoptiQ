"""CLI utilities shared across all SynoptiQ scripts.

Provides argument parsing, YAML config loading, device detection,
and directory setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from typing import Any

import yaml

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is malformed.
    """
    p = Path(path)
    if not p.exists():
        msg = f"Config file not found: {p}"
        raise FileNotFoundError(msg)
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def detect_device() -> str:
    """Detect the best available compute device.

    Returns:
        "cuda" if CUDA is available, "mps" if on Apple Silicon with MPS,
        or "cpu" as fallback.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def make_base_parser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with common flags.

    All SynoptiQ scripts inherit these flags:
      --data-dir      Root data directory (default: data/)
      --config-dir    Config directory (default: configs/)
      --output-dir    Output directory (default: outputs/)
      --device        Compute device (default: auto-detect)
      --verbose / -v  Enable debug logging

    Args:
        description: Script description string.

    Returns:
        ArgumentParser with common flags pre-added.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for results and checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda/mps/cpu — auto-detected if not specified)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser
