"""Compact serialised gold index — ship the reader without the raw Text-Fabric tree.

The Space bundles a ~2 MB gzipped JSON (built by ``scripts/build_reader_index.py``)
instead of the multi-file Text-Fabric dataset. :class:`IndexReader` loads it and, via
:class:`~synoptiq.reader.gold._ReaderBase`, exposes the exact same read API as
:class:`~synoptiq.reader.gold.GoldReader` — so the app is agnostic to the source.

Serialisation ships only the surfaced fields (surface, lemma, pos, morphology, gloss,
Strong's, trailing text); the syntax tree and any restricted columns are dropped.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from synoptiq.reader.gold import GoldReader, WordAnalysis, _ReaderBase
from synoptiq.utils.greek import strip_accents

SCHEMA_VERSION = 1


def serialize(reader: GoldReader) -> dict[str, Any]:
    """Serialise a gold reader into a compact JSON-able document.

    Word rows are positional lists ``[surface, lemma, pos, features, gloss, strong,
    after]`` to keep the artifact small.
    """
    books: dict[str, Any] = {}
    for b in reader.books():
        chapters: dict[str, Any] = {}
        for c in reader.chapters(b):
            chapters[str(c)] = {
                str(v): [
                    [w.surface, w.lemma, w.pos, w.features, w.gloss, w.strong, w.after]
                    for w in reader.verse(b, c, v)
                ]
                for v in reader.verses(b, c)
            }
        books[b] = chapters
    return {
        "schema": SCHEMA_VERSION,
        "name": reader.name,
        "book_order": reader.books(),
        "books": books,
    }


def save_index(reader: GoldReader, path: Path | str) -> Path:
    """Serialise ``reader`` and write it as gzipped JSON to ``path``. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(serialize(reader), ensure_ascii=False).encode("utf-8")
    path.write_bytes(gzip.compress(blob, 9))
    return path


def load_index(path: Path | str) -> dict[str, Any]:
    """Load a (optionally gzipped) index document written by :func:`save_index`."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)
    doc: dict[str, Any] = json.loads(raw)
    return doc


class IndexReader(_ReaderBase):
    """Read API backed by a serialised index document (see :func:`serialize`)."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self.name = doc.get("name", "")
        self._data: dict[str, Any] = doc["books"]
        self._book_order = list(doc.get("book_order") or self._data.keys())
        self._lexicon = None

        self._chapters = {}
        self._verses = {}
        self._book_cv = {}
        for b in self._book_order:
            cints = sorted(int(c) for c in self._data[b])
            self._chapters[b] = cints
            cv: list[tuple[int, int]] = []
            for c in cints:
                vints = sorted(int(v) for v in self._data[b][str(c)])
                self._verses[(b, c)] = vints
                cv.extend((c, v) for v in vints)
            self._book_cv[b] = cv
        self._book_lookup = {strip_accents(b).lower(): b for b in self._book_order}

    @classmethod
    def from_file(cls, path: Path | str) -> IndexReader:
        """Load an :class:`IndexReader` from a gzipped-JSON index file."""
        return cls(load_index(path))

    def verse(self, book: str, chapter: int, verse: int) -> list[WordAnalysis]:
        """Return the analysed words of a single verse (empty if not found)."""
        b = self._canon_book(book)
        ref = f"{b} {chapter}:{verse}"
        rows = self._data.get(b, {}).get(str(chapter), {}).get(str(verse), [])
        return [
            WordAnalysis(
                surface=r[0], lemma=r[1], pos=r[2], features=r[3],
                gloss=r[4], strong=r[5], after=r[6], ref=ref,
            )
            for r in rows
        ]
