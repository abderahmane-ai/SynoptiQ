"""Phase 1 data preparation pipeline entry point.

Orchestrates the full Phase 1 pipeline:
  1. Download all corpora (git clone --depth 1)
  2. Parse SBLGNT XML → token records
  3. Parse MorphGNT TSV → morphological annotations
  4. Merge SBLGNT + MorphGNT
  5. Assign pericope IDs from Aland table
  6. Compute Needleman-Wunsch token alignments
  7. Build pericope metadata and stratified splits
  8. Cache to Parquet in data/processed/

Usage:
    python scripts/prepare_data.py                     # Full pipeline
    python scripts/prepare_data.py --download-only     # Download only
    python scripts/prepare_data.py --no-download       # Skip download (re-parse)
    python scripts/prepare_data.py --force-rebuild     # Ignore cache
    python scripts/prepare_data.py --validate          # Just validate existing corpus

Go/No-Go validation:
  - 130,000 ≤ n_tokens ≤ 145,000 (SBLGNT synoptic word count)
  - n_pericopes > 100 (at minimum core pericopes assigned)
  - Alignment lemma match rate ≥ 80% on a random sample
  - make check passes (lint + typecheck + unit tests)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure scripts/ dir is on path when running from project root
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._cli_utils import make_base_parser  # noqa: E402

from synoptiq.utils.io_ import ensure_dir  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = make_base_parser("SynoptiQ Phase 1: Data preparation pipeline")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download corpora only; do not parse or process",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip corpus download; use existing raw data",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore existing Parquet cache and rebuild from scratch",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing processed corpus and exit",
    )
    parser.add_argument(
        "--books",
        nargs="+",
        default=["Matthew", "Mark", "Luke"],
        choices=["Matthew", "Mark", "Luke", "John"],
        help="Books to include in the corpus",
    )
    parser.add_argument(
        "--force-reclone",
        action="store_true",
        help="Delete and re-clone all corpora (use if download is corrupted)",
    )
    return parser


def validate_corpus(corpus: Corpus) -> bool:  # noqa: F821
    """Run go/no-go validation checks on the built corpus.

    Returns True if all checks pass, False otherwise.
    Logs detailed failure messages for any check that fails.
    """
    passed = True

    # Check 1: Token count
    min_tokens, max_tokens = 130_000, 145_000
    if not (min_tokens <= corpus.n_tokens <= max_tokens):
        _LOG.error(
            "token count out of expected range",
            extra={
                "n_tokens": corpus.n_tokens,
                "expected": f"{min_tokens:,}–{max_tokens:,}",
            },
        )
        passed = False
    else:
        _LOG.info("✓ token count OK", extra={"n_tokens": corpus.n_tokens})

    # Check 2: Pericope count
    min_pericopes = 100
    if corpus.n_pericopes < min_pericopes:
        _LOG.error(
            "too few pericopes",
            extra={"n_pericopes": corpus.n_pericopes, "minimum": min_pericopes},
        )
        passed = False
    else:
        _LOG.info("✓ pericope count OK", extra={"n_pericopes": corpus.n_pericopes})

    # Check 3: Triple tradition pericopes exist
    triple_pericopes = list(corpus.iter_pericopes(tradition="triple"))
    if not triple_pericopes:
        _LOG.error("no triple tradition pericopes found — alignment or pericope assignment failed")
        passed = False
    else:
        _LOG.info("✓ triple tradition pericopes found", extra={"count": len(triple_pericopes)})

    # Check 4: Double tradition pericopes exist
    double_pericopes = list(corpus.iter_pericopes(tradition="double"))
    if not double_pericopes:
        _LOG.error("no double tradition pericopes found")
        passed = False
    else:
        _LOG.info("✓ double tradition pericopes found", extra={"count": len(double_pericopes)})

    # Check 5: Direction pairs can be iterated
    n_pairs = sum(1 for _ in corpus.direction_pairs(tradition="triple"))
    if n_pairs < 50:
        _LOG.error(
            "too few direction pairs",
            extra={"n_pairs": n_pairs, "minimum": 50},
        )
        passed = False
    else:
        _LOG.info("✓ direction pairs OK", extra={"n_pairs": n_pairs})

    # Check 6: Alignment quality spot-check (random 10 pericopes)
    import random

    from synoptiq.data.alignment import alignment_score

    sample = random.sample(triple_pericopes, min(10, len(triple_pericopes)))
    lemma_rates: list[float] = []
    for pericope in sample:
        books = pericope["books"]
        if len(books) >= 2:
            book_a, book_b = books[0], books[1]
            tokens_a = pericope["tokens"].get(book_a, [])
            tokens_b = pericope["tokens"].get(book_b, [])
            pairs = pericope["alignment"].get((book_a, book_b), [])
            if tokens_a and tokens_b and pairs:
                scores = alignment_score(tokens_a, tokens_b, pairs)
                lemma_rates.append(scores["lemma_match_rate"])

    if lemma_rates:
        avg_lemma_rate = sum(lemma_rates) / len(lemma_rates)
        if avg_lemma_rate < 0.70:
            _LOG.error(
                "alignment quality too low",
                extra={"avg_lemma_match_rate": f"{avg_lemma_rate:.1%}", "threshold": "70%"},
            )
            passed = False
        else:
            _LOG.info(
                "✓ alignment quality OK",
                extra={"avg_lemma_match_rate": f"{avg_lemma_rate:.1%}"},
            )

    if passed:
        _LOG.info("✓ ALL VALIDATION CHECKS PASSED — corpus is ready for Phase 2")
    else:
        _LOG.error("✗ VALIDATION FAILED — see errors above")

    return passed


def main() -> int:
    """Main entry point for prepare_data.py."""
    parser = _build_parser()
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    ensure_dir(data_dir)

    _LOG.info(
        "SynoptiQ Phase 1 — Data Preparation",
        extra={
            "data_dir": str(data_dir),
            "books": args.books,
            "download": not args.no_download,
            "force_rebuild": args.force_rebuild,
        },
    )

    # Validate only
    if args.validate:
        from synoptiq.data.corpus import Corpus

        processed_dir = data_dir / "processed"
        corpus = Corpus.from_parquet(
            processed_dir / "tokens.parquet",
            processed_dir / "pericopes.parquet",
            alignments_path=processed_dir / "alignments.json",
            splits_path=processed_dir / "splits.json",
        )
        ok = validate_corpus(corpus)
        return 0 if ok else 1

    # Step 1: Download corpora
    if not args.no_download:
        _LOG.info("Step 1/5: Downloading corpora...")
        from synoptiq.data._download import download_all

        download_all(
            data_dir / "raw",
            force_reclone=args.force_reclone,
        )

    if args.download_only:
        _LOG.info("--download-only specified; stopping after download")
        return 0

    # Steps 2–7: Build corpus
    _LOG.info("Steps 2–7: Building corpus from raw data...")
    from synoptiq.data.corpus import Corpus

    corpus = Corpus.from_raw(
        data_dir,
        books=args.books,
        use_cache=not args.force_rebuild,
    )

    # Step 8: Validate
    _LOG.info("Step 8: Validating corpus...")
    ok = validate_corpus(corpus)

    _LOG.info(repr(corpus))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
