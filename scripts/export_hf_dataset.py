"""Export the SynoptiQ corpus as a HuggingFace-compatible dataset.

Produces a directory structure ready for `push_to_hub`:

    datasets/synoptiq-corpus/
    ├── README.md              # Dataset card (already written)
    ├── data/
    │   ├── train-00000.parquet
    │   ├── validation-00000.parquet
    │   └── test-00000.parquet
    ├── pericopes.parquet       # Pericope-level metadata table
    └── alignments.json         # Token-level Needleman-Wunsch alignment pairs

The main dataset (data/*.parquet) is token-level with pericope metadata
denormalised onto each row so users can filter by tradition, genre, or
pericope without joining tables.

Usage:
    python scripts/export_hf_dataset.py              # Full export
    python scripts/export_hf_dataset.py --push       # Export + push to Hub
    python scripts/export_hf_dataset.py --push --repo synoptiq/synoptiq-corpus
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.utils.io_ import ensure_dir  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# ── Column schema for the exported token Parquet ──────────────────────────
# These are the columns written to the split Parquet files, in order.
# Every column has a description; types are inferred by Pandas/PyArrow.
TOKEN_COLUMNS = [
    "token_id",         # str   — stable identifier: "Matthew.1.1.0"
    "book",             # str   — canonical gospel name: Matthew, Mark, Luke
    "chapter",          # int   — chapter number (1-indexed)
    "verse",            # int   — verse number (1-indexed)
    "position",         # int   — token position within verse (0-indexed)
    "text",             # str   — surface form as written in SBLGNT (e.g. "γενέσεως")
    "normalized",       # str   — NFD-normalised, accent-stripped (e.g. "γενεσεως")
    "lemma",            # str   — dictionary headword from MorphGNT (e.g. "γένεσις")
    "pos",              # str   — part-of-speech tag (e.g. "N-", "V-", "RA")
    "morph",            # str   — 8-char CCAT morphological parsing (e.g. "----GSF-")
    "pericope_id",      # str   — Aland pericope number (e.g. "018"); "" if unassigned
    "tradition",        # str   — triple | double | matthean_unique | lukan_unique | mark_unique
    "genre",            # str   — narrative | discourse | wisdom | passion | other
    "books_in_pericope",# str   — JSON list: '["Matthew","Mark","Luke"]'
    "is_punctuation",   # bool  — True if the token is punctuation (always False in SBLGNT)
]

# ── Column schema for the pericope metadata Parquet ───────────────────────
PERICOPE_COLUMNS = [
    "pericope_id",      # str   — Aland pericope number
    "tradition",        # str   — triple | double | matthean_unique | lukan_unique | mark_unique
    "genre",            # str   — narrative | discourse | wisdom | passion | other
    "books",            # str   — JSON list of canonical gospel names present
    "n_tokens",         # int   — total token count across all books in this pericope
    "n_alignment_pairs",# int   — total alignment pairs across all book pairs; 0 for single-source
    "split",            # str   — train | val | test
]


def _enrich_tokens(
    token_df: pd.DataFrame,
    pericope_df: pd.DataFrame,
    split_assignment: dict[str, str],
) -> pd.DataFrame:
    """Join pericope metadata onto the token table and add split labels.

    Args:
        token_df: Flat token DataFrame from the Corpus.
        pericope_df: Pericope metadata DataFrame.
        split_assignment: Dict pericope_id → train/val/test.

    Returns:
        Enriched token DataFrame with tradition, genre, books_in_pericope, split.
    """
    # Build lookup: pericope_id → (tradition, genre, books)
    pericope_lookup: dict[str, tuple[str, str, str]] = {}
    for _, row in pericope_df.iterrows():
        pid = row["pericope_id"]
        pericope_lookup[pid] = (row["tradition"], row["genre"], row["books"])

    # Apply enrichment
    traditions: list[str] = []
    genres: list[str] = []
    books_in_pericope: list[str] = []
    splits: list[str] = []

    for _, token in token_df.iterrows():
        pid = token["pericope_id"]
        if pid and pid in pericope_lookup:
            trad, genre, books = pericope_lookup[pid]
            traditions.append(trad)
            genres.append(genre)
            books_in_pericope.append(books)
        else:
            traditions.append("")
            genres.append("")
            books_in_pericope.append("[]")
        splits.append(split_assignment.get(pid, ""))

    result = token_df.copy()
    result["tradition"] = traditions
    result["genre"] = genres
    result["books_in_pericope"] = books_in_pericope
    result["split"] = splits

    # Keep split column for the caller to partition; it is dropped when
    # writing individual split files (it is redundant with file location).
    return result[[*TOKEN_COLUMNS, "split"]]


def _build_pericope_table(
    pericope_df: pd.DataFrame,
    token_df: pd.DataFrame,
    alignments: dict,
    split_assignment: dict[str, str],
) -> pd.DataFrame:
    """Build the pericope-level metadata table for auxiliary export.

    Counts tokens and alignment pairs per pericope, adds split labels.
    """
    # Count tokens per pericope
    token_counts = token_df.groupby("pericope_id").size().to_dict()

    # Count alignment pairs per pericope
    alignment_counts: dict[str, int] = defaultdict(int)
    for (pid, _ba, _bb), pairs in alignments.items():
        alignment_counts[pid] += len(pairs)

    records = []
    for _, row in pericope_df.iterrows():
        pid = row["pericope_id"]
        records.append({
            "pericope_id": pid,
            "tradition": row["tradition"],
            "genre": row["genre"],
            "books": row["books"],
            "n_tokens": token_counts.get(pid, 0),
            "n_alignment_pairs": alignment_counts.get(pid, 0),
            "split": split_assignment.get(pid, ""),
        })

    return pd.DataFrame(records)[PERICOPE_COLUMNS]


def _validate_export(
    output_dir: Path,
    corpus: Corpus,
    enriched: pd.DataFrame,
) -> bool:
    """Run sanity checks on the exported dataset.

    Returns True if all checks pass.
    """
    ok = True

    # Check 1: Token count preserved
    if len(enriched) != corpus.n_tokens:
        _LOG.error(f"token count mismatch: {len(enriched)} vs {corpus.n_tokens}")
        ok = False

    # Check 2: All splits present
    splits_present = set(enriched["split"].unique())
    expected_splits = {"train", "val", "test"}
    if not expected_splits.issubset(splits_present):
        _LOG.error(f"missing splits: {expected_splits - splits_present}")
        ok = False

    # Check 3: Pericope IDs preserved
    original_pids = set(corpus._pericope_df["pericope_id"])
    enriched_pids = set(enriched.loc[enriched["pericope_id"] != "", "pericope_id"])
    missing = original_pids - enriched_pids
    if missing:
        _LOG.warning(f"{len(missing)} pericopes have no tokens: {sorted(missing)[:10]}...")

    # Check 4: Each split file is readable
    for split_name in ["train", "validation", "test"]:
        parquet_path = output_dir / "data" / f"{split_name}-00000.parquet"
        if not parquet_path.exists():
            _LOG.error(f"missing split file: {parquet_path}")
            ok = False
            continue
        loaded = pd.read_parquet(parquet_path)
        if len(loaded) == 0:
            _LOG.error(f"empty split file: {parquet_path}")
            ok = False

    # Check 5: Alignment JSON is valid and round-trips
    alignments_path = output_dir / "alignments.json"
    if alignments_path.exists():
        raw = json.loads(alignments_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            _LOG.error("alignments.json is not a dict")
            ok = False

    # Check 6: Pericope parquet is valid
    pericopes_path = output_dir / "pericopes.parquet"
    if pericopes_path.exists():
        pericope_df = pd.read_parquet(pericopes_path)
        if len(pericope_df) != corpus.n_pericopes:
            _LOG.error(f"pericope count mismatch: {len(pericope_df)} vs {corpus.n_pericopes}")
            ok = False

    if ok:
        _LOG.info("All export validation checks passed")
    return ok


def export(
    data_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Export the corpus as a HuggingFace-compatible dataset directory.

    Args:
        data_dir: Root data directory containing processed/ and raw/.
        output_dir: Destination directory for the HF dataset.
        force: If True, overwrite existing output.

    Returns:
        The output directory path.
    """
    processed_dir = data_dir / "processed"
    tokens_path = processed_dir / "tokens.parquet"
    pericopes_path = processed_dir / "pericopes.parquet"
    alignments_path = processed_dir / "alignments.json"
    splits_path = processed_dir / "splits.json"

    for p in [tokens_path, pericopes_path, alignments_path, splits_path]:
        if not p.exists():
            _LOG.error(f"missing input file: {p}")
            _LOG.error("Run 'python scripts/prepare_data.py' first to build the corpus.")
            sys.exit(1)

    # Load the corpus
    _LOG.info("loading corpus from processed Parquet cache")
    corpus = Corpus.from_parquet(
        tokens_path,
        pericopes_path,
        alignments_path=alignments_path,
        splits_path=splits_path,
    )

    # Load raw data frames
    token_df = corpus._token_df
    pericope_df = corpus._pericope_df
    alignments = corpus._alignments
    split_assignment = corpus._split_assignment

    # ── Enrich tokens with pericope metadata ─────────────────────────────
    _LOG.info("enriching tokens with pericope metadata and split labels")
    enriched = _enrich_tokens(token_df, pericope_df, split_assignment)

    # ── Build pericope metadata table ────────────────────────────────────
    _LOG.info("building pericope metadata table")
    pericope_table = _build_pericope_table(pericope_df, token_df, alignments, split_assignment)

    # ── Write output ─────────────────────────────────────────────────────
    if output_dir.exists():
        if force:
            # Remove only generated files, preserve README.md and other
            # hand-authored content in the dataset directory.
            for p in output_dir.iterdir():
                if p.name == "README.md":
                    continue
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        else:
            _LOG.error(f"output directory already exists: {output_dir}")
            _LOG.error("Use --force to overwrite.")
            sys.exit(1)

    data_subdir = output_dir / "data"
    ensure_dir(data_subdir)

    # Write split Parquet files (HF naming: train, validation, test)
    split_file_map = {
        "train": data_subdir / "train-00000.parquet",
        "val": data_subdir / "validation-00000.parquet",
        "test": data_subdir / "test-00000.parquet",
    }

    for split_key, filepath in split_file_map.items():
        split_data = enriched[enriched["split"] == split_key].drop(columns=["split"])
        split_data.to_parquet(filepath, index=False, engine="pyarrow")
        _LOG.info(
            f"wrote {split_key} split",
            extra={"path": str(filepath), "n_rows": len(split_data)},
        )

    # Write pericope metadata
    pericope_out = output_dir / "pericopes.parquet"
    pericope_table.to_parquet(pericope_out, index=False, engine="pyarrow")
    _LOG.info("wrote pericope metadata", extra={"n_rows": len(pericope_table)})

    # Write alignments JSON
    alignments_out = output_dir / "alignments.json"
    alignments_json = {
        f"{pid}|{ba}|{bb}": [[a, b] for a, b in pairs]
        for (pid, ba, bb), pairs in alignments.items()
    }
    alignments_out.write_text(
        json.dumps(alignments_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _LOG.info("wrote alignments", extra={"n_pairs": len(alignments)})

    # ── Validate ─────────────────────────────────────────────────────────
    ok = _validate_export(output_dir, corpus, enriched)
    if not ok:
        _LOG.error("Export validation failed — dataset may be incomplete.")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────
    _LOG.info("=" * 60)
    _LOG.info("Export complete: " + str(output_dir))
    _LOG.info(f"  Train tokens:      {len(enriched[enriched['split'] == 'train']):,}")
    _LOG.info(f"  Validation tokens: {len(enriched[enriched['split'] == 'val']):,}")
    _LOG.info(f"  Test tokens:       {len(enriched[enriched['split'] == 'test']):,}")
    _LOG.info(f"  Pericopes:         {len(pericope_table)}")
    _LOG.info(f"  Alignment pairs:   {len(alignments)}")
    _LOG.info("=" * 60)
    _LOG.info("Ready to push: huggingface-cli upload synoptiq/synoptiq-corpus " + str(output_dir))

    return output_dir


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SynoptiQ: Export HuggingFace dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "data",
        help="Root data directory containing processed/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "datasets" / "synoptiq-corpus",
        help="Destination directory for the HF dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace Hub after export",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="ainouche-abderahmane/synoptiq-corpus",
        help="HuggingFace Hub repository ID",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create as a private repository",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    export(data_dir, output_dir, force=args.force)

    if args.push:
        from huggingface_hub import HfApi, create_repo

        _LOG.info("pushing to HuggingFace Hub", extra={"repo": args.repo})

        try:
            create_repo(args.repo, private=args.private, exist_ok=True)
        except Exception as exc:
            _LOG.error("failed to create repo", extra={"error": str(exc)})
            _LOG.error("Run 'huggingface-cli login' first if you haven't.")
            return 1

        api = HfApi()
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.repo,
            repo_type="dataset",
        )
        _LOG.info(f"pushed to https://huggingface.co/datasets/{args.repo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
