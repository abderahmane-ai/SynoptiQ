"""Safe file I/O helpers for SynoptiQ.

All writes use atomic rename (write to .tmp, then os.rename) to prevent
partial-write corruption on interruption. Parquet helpers include
schema validation for the token and pericope dataframes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

# ── Directory helpers ─────────────────────────────────────────────────────────


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.

    Raises:
        NotADirectoryError: If path exists but is a file.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── JSON helpers ──────────────────────────────────────────────────────────────


def load_json(path: Path | str) -> Any:
    """Load a JSON file and return the parsed object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path | str, *, indent: int = 2) -> None:
    """Atomically write an object to a JSON file.

    Uses write-to-temp + rename to prevent partial writes on interruption.

    Args:
        obj: JSON-serializable object.
        path: Destination file path.
        indent: JSON indentation (default 2).
    """
    p = Path(path)
    ensure_dir(p.parent)
    content = json.dumps(obj, ensure_ascii=False, indent=indent, default=str)

    # Atomic write: write to sibling .tmp file then rename
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, p)  # Atomic on POSIX; near-atomic on Windows
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Parquet helpers ───────────────────────────────────────────────────────────

# Expected columns for the token DataFrame
TOKEN_DF_COLUMNS: frozenset[str] = frozenset(
    {
        "token_id",
        "book",
        "chapter",
        "verse",
        "position",
        "text",
        "normalized",
        "lemma",
        "pos",
        "morph",
        "pericope_id",
        "is_punctuation",
    }
)

# Expected columns for the pericope DataFrame
PERICOPE_DF_COLUMNS: frozenset[str] = frozenset(
    {
        "pericope_id",
        "tradition",
        "genre",
        "books",
    }
)


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        msg = f"Parquet file not found: {p}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(p)


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Atomically write a DataFrame to a Parquet file.

    Args:
        df: DataFrame to save.
        path: Destination file path.
    """
    p = Path(path)
    ensure_dir(p.parent)

    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp.parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def validate_token_df(df: pd.DataFrame) -> None:
    """Validate that a DataFrame has the expected token schema.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = TOKEN_DF_COLUMNS - set(df.columns)
    if missing:
        msg = f"Token DataFrame missing columns: {sorted(missing)}"
        raise ValueError(msg)


def validate_pericope_df(df: pd.DataFrame) -> None:
    """Validate that a DataFrame has the expected pericope schema.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = PERICOPE_DF_COLUMNS - set(df.columns)
    if missing:
        msg = f"Pericope DataFrame missing columns: {sorted(missing)}"
        raise ValueError(msg)
