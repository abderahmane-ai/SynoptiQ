"""The central Corpus class for SynoptiQ.

``Corpus`` is the single data entry point for all downstream models.
It provides:
  - Lazy loading from Parquet cache (avoids re-parsing on every run)
  - Iteration over pericopes filtered by tradition, book, genre
  - Direction pair generation (the primary input to the direction scorer)
  - Token and verse-level access

Usage:
    # Build from raw data (Phase 1 prepare_data.py):
    corpus = Corpus.from_raw(Path("data/"))

    # Load from Parquet cache (subsequent runs):
    corpus = Corpus.from_parquet(
        Path("data/processed/tokens.parquet"),
        Path("data/processed/pericopes.parquet"),
    )

    # Iterate pericopes:
    for pericope in corpus.iter_pericopes(tradition="triple"):
        matthew_tokens = pericope["tokens"]["Matthew"]
        mark_tokens = pericope["tokens"]["Mark"]
        ...

    # Iterate direction pairs:
    for book_a, tokens_a, book_b, tokens_b, alignment_pairs in corpus.direction_pairs():
        ...
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from itertools import combinations
import json
from pathlib import Path

import pandas as pd

from synoptiq.utils.io_ import (
    ensure_dir,
    load_parquet,
    save_parquet,
    validate_pericope_df,
    validate_token_df,
)
from synoptiq.utils.logging_ import get_logger
from synoptiq.utils.types_ import Book, Genre, PericopeAlignment, TokenRecord, Tradition

_LOG = get_logger(__name__)


class Corpus:
    """Aligned Synoptic Gospel corpus with lazy Parquet loading.

    The corpus is the central data object of SynoptiQ. It holds:
    - A flat DataFrame of all annotated tokens (``_token_df``)
    - A DataFrame of pericope metadata (``_pericope_df``)
    - Pre-computed alignment pairs for each pericope×book_pair
    - Split assignments (train/val/test)

    Attributes:
        n_tokens: Total token count across all books.
        n_pericopes: Number of unique Aland pericopes in the corpus.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        token_df: pd.DataFrame,
        pericope_df: pd.DataFrame,
        alignments: dict[tuple[str, str, str], list[tuple[int | None, int | None]]],
        split_assignment: dict[str, str] | None = None,
    ) -> None:
        """Direct constructor — prefer from_parquet() or from_raw() factories.

        Args:
            token_df: DataFrame of all token records.
            pericope_df: DataFrame of pericope metadata.
            alignments: Dict (pericope_id, book_a, book_b) → token pair list.
            split_assignment: Dict pericope_id → "train"/"val"/"test".
        """
        validate_token_df(token_df)
        validate_pericope_df(pericope_df)

        self._token_df = token_df
        self._pericope_df = pericope_df
        self._alignments = alignments
        self._split_assignment = split_assignment or {}

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(
        cls,
        tokens_path: Path | str,
        pericopes_path: Path | str,
        *,
        alignments_path: Path | str | None = None,
        splits_path: Path | str | None = None,
    ) -> Corpus:
        """Load corpus from Parquet cache files.

        Args:
            tokens_path: Path to tokens Parquet file.
            pericopes_path: Path to pericopes Parquet file.
            alignments_path: Path to alignments JSON file (optional).
            splits_path: Path to split assignment JSON file (optional).

        Returns:
            Corpus instance with lazy-loaded DataFrames.
        """
        _LOG.info("loading corpus from Parquet", extra={"tokens": str(tokens_path)})
        token_df = load_parquet(Path(tokens_path))
        pericope_df = load_parquet(Path(pericopes_path))

        alignments: dict[tuple[str, str, str], list[tuple[int | None, int | None]]] = {}
        if alignments_path and Path(alignments_path).exists():
            raw = json.loads(Path(alignments_path).read_text(encoding="utf-8"))
            for key_str, pairs in raw.items():
                pid, ba, bb = key_str.split("|")
                alignments[(pid, ba, bb)] = [(a, b) for a, b in pairs]

        splits: dict[str, str] = {}
        if splits_path and Path(splits_path).exists():
            splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))

        _LOG.info(
            "corpus loaded",
            extra={
                "n_tokens": len(token_df),
                "n_pericopes": len(pericope_df),
                "n_alignments": len(alignments),
            },
        )
        return cls(token_df, pericope_df, alignments, splits)

    @classmethod
    def from_raw(
        cls,
        data_dir: Path,
        *,
        books: list[Book] | None = None,
        use_cache: bool = True,
    ) -> Corpus:
        """Build corpus from raw downloaded data, running the full parse pipeline.

        This is the heavy factory method called by ``prepare_data.py``.
        On subsequent runs, ``from_parquet()`` is much faster.

        Args:
            data_dir: Root data directory (contains ``raw/`` and ``processed/``).
            books: Books to include. Defaults to Matthew, Mark, Luke.
            use_cache: If True and processed/ files exist, load from cache.

        Returns:
            Built Corpus instance.
        """
        processed_dir = data_dir / "processed"
        tokens_path = processed_dir / "tokens.parquet"
        pericopes_path = processed_dir / "pericopes.parquet"

        if use_cache and tokens_path.exists() and pericopes_path.exists():
            _LOG.info("cache found — loading from Parquet")
            return cls.from_parquet(
                tokens_path,
                pericopes_path,
                alignments_path=processed_dir / "alignments.json",
                splits_path=processed_dir / "splits.json",
            )

        _LOG.info("building corpus from raw data")
        target_books: list[Book] = books or ["Matthew", "Mark", "Luke"]
        raw_dir = data_dir / "raw"

        # ── Parse SBLGNT ──────────────────────────────────────────────────
        from synoptiq.data._parse_sblgnt import parse_sblgnt

        sblgnt_tokens_by_book = parse_sblgnt(raw_dir / "sblgnt", books=target_books)

        # ── Parse MorphGNT + merge ────────────────────────────────────────
        from synoptiq.data._parse_morphgnt import merge_sblgnt_with_morphgnt, parse_morphgnt

        morphgnt_lookup = parse_morphgnt(raw_dir / "morphgnt", books=target_books)

        all_tokens: list[dict[str, object]] = []
        for book in target_books:
            book_tokens = sblgnt_tokens_by_book.get(book, [])
            merged = merge_sblgnt_with_morphgnt(book_tokens, morphgnt_lookup)
            all_tokens.extend(merged)

        # ── Assign pericope IDs from Aland table ─────────────────────────
        from synoptiq.data._parse_n1904 import assign_pericope_ids

        all_tokens = assign_pericope_ids(all_tokens)

        # ── Build token DataFrame ─────────────────────────────────────────
        token_df = pd.DataFrame(all_tokens)
        # Ensure is_punctuation is bool (may be None from parser)
        token_df["is_punctuation"] = token_df["is_punctuation"].fillna(False).astype(bool)
        token_df["pericope_id"] = token_df["pericope_id"].astype(str)
        token_df.loc[token_df["pericope_id"] == "None", "pericope_id"] = ""

        # ── Group tokens by pericope ──────────────────────────────────────
        from synoptiq.data.pericope import build_pericope_alignments, group_tokens_by_pericope

        token_records: list[TokenRecord] = token_df.to_dict(orient="records")  # type: ignore[assignment]
        grouped = group_tokens_by_pericope(token_records)

        # ── Compute token-level alignments ────────────────────────────────
        from synoptiq.data.alignment import align_tokens

        alignment_dict: dict[tuple[str, str, str], list[tuple[int | None, int | None]]] = {}

        for pericope_id, book_token_dict in grouped.items():
            if pericope_id == "__unassigned__":
                continue
            books_present = list(book_token_dict.keys())
            for book_a, book_b in combinations(books_present, 2):
                tokens_a = book_token_dict[book_a]
                tokens_b = book_token_dict[book_b]
                if tokens_a and tokens_b:
                    pairs = align_tokens(tokens_a, tokens_b)
                    alignment_dict[(pericope_id, book_a, book_b)] = pairs

        _LOG.info("alignments computed", extra={"n_pairs": len(alignment_dict)})

        # ── Build pericope metadata DataFrame ─────────────────────────────
        pericope_alignments = build_pericope_alignments(grouped, alignment_dict)
        pericope_records = [
            {
                "pericope_id": a["pericope_id"],
                "tradition": a["tradition"],
                "genre": a["genre"],
                "books": json.dumps(a["books"]),  # Serialize list for Parquet
            }
            for a in pericope_alignments
        ]
        pericope_df = pd.DataFrame(pericope_records)

        # ── Compute splits ────────────────────────────────────────────────
        from synoptiq.data.splits import split_pericopes

        split_result = split_pericopes(pericope_alignments)
        split_assignment: dict[str, str] = {}
        for pid in split_result["train_ids"]:
            split_assignment[pid] = "train"
        for pid in split_result["val_ids"]:
            split_assignment[pid] = "val"
        for pid in split_result["test_ids"]:
            split_assignment[pid] = "test"

        # ── Cache to Parquet ──────────────────────────────────────────────
        ensure_dir(processed_dir)
        save_parquet(token_df, tokens_path)
        save_parquet(pericope_df, pericopes_path)

        # Save alignments as JSON (tuples with None values can't go in Parquet)
        alignments_json = {
            f"{pid}|{ba}|{bb}": [[a, b] for a, b in pairs]
            for (pid, ba, bb), pairs in alignment_dict.items()
        }
        (processed_dir / "alignments.json").write_text(
            json.dumps(alignments_json, allow_nan=False), encoding="utf-8"
        )
        (processed_dir / "splits.json").write_text(
            json.dumps(split_assignment, indent=2), encoding="utf-8"
        )

        _LOG.info(
            "corpus built and cached",
            extra={
                "n_tokens": len(token_df),
                "n_pericopes": len(pericope_df),
            },
        )
        return cls(token_df, pericope_df, alignment_dict, split_assignment)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_tokens(self) -> int:
        """Total token count across all books."""
        return len(self._token_df)

    @property
    def n_pericopes(self) -> int:
        """Number of unique Aland pericopes in the corpus."""
        return len(self._pericope_df)

    # ── Iteration ─────────────────────────────────────────────────────────────

    def iter_pericopes(
        self,
        *,
        tradition: Tradition | None = None,
        books: tuple[Book, ...] | None = None,
        genre: Genre | None = None,
        split: str | None = None,
    ) -> Iterator[PericopeAlignment]:
        """Iterate over pericopes with optional filtering.

        Args:
            tradition: Filter by tradition type (e.g., "triple").
            books: Filter to pericopes containing ALL specified books.
            genre: Filter by genre (e.g., "narrative").
            split: Filter by split assignment ("train", "val", "test").

        Yields:
            PericopeAlignment dicts.
        """
        pericope_df = self._pericope_df.copy()

        if tradition is not None:
            pericope_df = pericope_df[pericope_df["tradition"] == tradition]

        if genre is not None:
            pericope_df = pericope_df[pericope_df["genre"] == genre]

        if split is not None and self._split_assignment:
            valid_pids = {pid for pid, s in self._split_assignment.items() if s == split}
            pericope_df = pericope_df[pericope_df["pericope_id"].isin(valid_pids)]

        for _, row in pericope_df.iterrows():
            pid = row["pericope_id"]
            pericope_books: list[Book] = json.loads(row["books"])  # type: ignore[assignment]

            if books is not None and not all(b in pericope_books for b in books):
                continue

            # Reconstruct PericopeAlignment from token DataFrame
            token_mask = self._token_df["pericope_id"] == pid
            pericope_tokens = self._token_df[token_mask]

            book_tokens: dict[Book, list[TokenRecord]] = defaultdict(list)
            for _, token_row in pericope_tokens.sort_values(
                ["chapter", "verse", "position"]
            ).iterrows():
                book: Book = token_row["book"]  # type: ignore[assignment]
                book_tokens[book].append(token_row.to_dict())  # type: ignore[arg-type]

            # Retrieve pre-computed alignments
            pericope_alignments: dict[tuple[Book, Book], list[tuple[int | None, int | None]]] = {}
            for book_a, book_b in combinations(pericope_books, 2):
                key = (pid, book_a, book_b)
                if key in self._alignments:
                    pericope_alignments[(book_a, book_b)] = self._alignments[key]

            yield PericopeAlignment(
                pericope_id=pid,
                tradition=row["tradition"],  # type: ignore[arg-type]
                genre=row["genre"],  # type: ignore[arg-type]
                books=pericope_books,
                tokens=dict(book_tokens),
                alignment=pericope_alignments,
            )

    def iter_direction_pairs(
        self,
        *,
        tradition: Tradition | None = None,
        split: str | None = None,
    ) -> Iterator[
        tuple[
            str, Book, list[TokenRecord], Book, list[TokenRecord],
            list[tuple[int | None, int | None]],
        ]
    ]:
        """Like :meth:`direction_pairs` but also yields the source ``pericope_id``.

        The pericope_id is needed to group samples for pericope-level evaluation
        (e.g. bootstrap CIs), where multiple book pairs and their swap-augmented
        copies all originate from a single pericope and are therefore not
        statistically independent.

        Yields:
            Tuples of (pericope_id, book_a, tokens_a, book_b, tokens_b, alignment_pairs).
        """
        for pericope in self.iter_pericopes(tradition=tradition, split=split):
            pericope_id = pericope["pericope_id"]
            books = pericope["books"]
            for book_a, book_b in combinations(books, 2):
                tokens_a = pericope["tokens"].get(book_a, [])
                tokens_b = pericope["tokens"].get(book_b, [])
                alignment_pairs = pericope["alignment"].get(
                    (book_a, book_b),
                    pericope["alignment"].get((book_b, book_a), []),
                )
                if tokens_a and tokens_b:
                    yield pericope_id, book_a, tokens_a, book_b, tokens_b, alignment_pairs

    def direction_pairs(
        self,
        *,
        tradition: Tradition | None = None,
        split: str | None = None,
    ) -> Iterator[
        tuple[Book, list[TokenRecord], Book, list[TokenRecord], list[tuple[int | None, int | None]]]
    ]:
        """Iterate over book pairs for direction detection training.

        For triple tradition, yields 3 pairs: (Matt, Mark), (Matt, Luke), (Mark, Luke).
        For double tradition, yields 1 pair: (Matthew, Luke).

        Args:
            tradition: Filter to specific tradition type.
            split: Filter by split assignment.

        Yields:
            Tuples of (book_a, tokens_a, book_b, tokens_b, alignment_pairs).
        """
        for _pid, book_a, tokens_a, book_b, tokens_b, alignment_pairs in (
            self.iter_direction_pairs(tradition=tradition, split=split)
        ):
            yield book_a, tokens_a, book_b, tokens_b, alignment_pairs

    def get_tokens(
        self,
        book: Book | None = None,
        *,
        pericope_id: str | None = None,
        split: str | None = None,
        exclude_punctuation: bool = True,
    ) -> list[TokenRecord]:
        """Retrieve tokens with optional filtering.

        Args:
            book: Filter to a specific book. None = all books.
            pericope_id: Filter to a specific pericope.
            split: Filter to a specific split partition.
            exclude_punctuation: If True, exclude punctuation tokens.

        Returns:
            List of TokenRecord dicts.
        """
        mask = pd.Series([True] * len(self._token_df), index=self._token_df.index)

        if book is not None:
            mask &= self._token_df["book"] == book
        if pericope_id is not None:
            mask &= self._token_df["pericope_id"] == pericope_id
        if exclude_punctuation:
            mask &= ~self._token_df["is_punctuation"].fillna(False)
        if split is not None and self._split_assignment:
            valid_pids = {pid for pid, s in self._split_assignment.items() if s == split}
            mask &= self._token_df["pericope_id"].isin(valid_pids)

        return self._token_df[mask].to_dict(orient="records")  # type: ignore[return-value]

    def get_verse(
        self,
        book: Book,
        chapter: int,
        verse: int,
    ) -> list[TokenRecord]:
        """Retrieve all tokens for a specific verse.

        Args:
            book: Canonical book name.
            chapter: Chapter number.
            verse: Verse number.

        Returns:
            List of TokenRecords for the verse, in position order.
        """
        mask = (
            (self._token_df["book"] == book)
            & (self._token_df["chapter"] == chapter)
            & (self._token_df["verse"] == verse)
        )
        return (
            self._token_df[mask]
            .sort_values(["chapter", "verse", "position"])
            .to_dict(orient="records")  # type: ignore[return-value]
        )

    # ── I/O ───────────────────────────────────────────────────────────────────

    def to_parquet(
        self,
        tokens_path: Path | str,
        pericopes_path: Path | str,
    ) -> None:
        """Save the corpus to Parquet files.

        Args:
            tokens_path: Destination for token DataFrame.
            pericopes_path: Destination for pericope DataFrame.
        """
        save_parquet(self._token_df, Path(tokens_path))
        save_parquet(self._pericope_df, Path(pericopes_path))
        _LOG.info(
            "corpus saved",
            extra={"tokens": str(tokens_path), "pericopes": str(pericopes_path)},
        )

    def __repr__(self) -> str:
        splits = (
            dict.fromkeys(set(self._split_assignment.values()))
            if self._split_assignment
            else "none"
        )
        return (
            f"Corpus(n_tokens={self.n_tokens:,}, n_pericopes={self.n_pericopes}, "
            f"splits={splits})"
        )
