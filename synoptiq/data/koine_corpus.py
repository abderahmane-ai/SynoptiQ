"""Unified Koine corpus builder for Koine-T5-Hexapla.

Assembles large-scale raw Koine prose into passage-level training units for the
generation-focused multitask model. It adds two capabilities the current pipeline
lacked (see ``docs/GENERATION_PLAN.md``):

  1. A minimal, dependency-free **Text-Fabric reader** for the on-disk Rahlfs-1935
     LXX (``data/raw/lxx/tf/1935``, 623,693 words). The section features
     (``book``/``chapter``/``verse``) are stored dense per word-slot and line-aligned
     with ``word.tf``, so grouping into verses is a single ``zip`` — no ``oslots``/
     ``otype`` parsing needed. This recovers the single largest body of Koine Greek, which
     ``_extract_text_from_dir`` cannot read because it does not parse TF slot files
     (the documented "LXX = 0 chunks"). The on-disk LXX is the *eliranwong* Text-Fabric
     edition, not the *biblicalhumanities* plaintext one that ``_parse_lxx`` expects —
     hence the mismatch.

  2. **Passage-window chunking** + **continuation (prefix-LM) example** construction —
     the two generative signals the current denoise-only diet lacks.

The pure helpers (TF reader, chunking, continuation, dedup/decontamination) are
import-light and unit-tested; only :func:`build_raw_passages` lazily imports the
torch-backed TEI/txt extractor from ``synoptiq.training.dapt``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# ── Passage record ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Passage:
    """A coherent unit of Greek prose with provenance.

    ``register`` ("koine" / "classical") drives corpus census + replay balancing;
    ``ref`` is a best-effort human-readable locus (e.g. "Gen 1:1"), empty when the
    source (TEI fragments) carries no reliable reference.
    """

    text: str
    source: str
    register: str
    ref: str = ""


# ── Minimal Text-Fabric reader ──────────────────────────────────────────────


def read_tf_column(path: Path | str) -> list[str]:
    """Read a Text-Fabric ``@node`` feature file into a dense list of per-slot values.

    TF node files are a leading run of ``@``-prefixed header lines, one blank separator
    line, then one value per node in slot order (see ``data/raw/lxx/tf/1935/word.tf``).
    Returns the data values in slot order (index 0 == slot 1).
    """
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("@"):
        i += 1
    if i < len(lines) and lines[i] == "":  # single blank header/data separator
        i += 1
    data = lines[i:]
    if data and data[-1] == "":  # trailing-newline artefact
        data.pop()
    return data


def iter_lxx_verses(tf_dir: Path | str) -> Iterator[Passage]:
    """Yield one :class:`Passage` per LXX verse from a Rahlfs-1935 TF dataset.

    Groups consecutive word slots sharing ``(book, chapter, verse)``. Validated against
    the on-disk data: 623,693 words, Gen 1:1 == "ἐν ἀρχῇ ἐποίησεν ὁ θεὸς …".
    """
    tf_dir = Path(tf_dir)
    words = read_tf_column(tf_dir / "word.tf")
    n = len(words)

    # Section features (book/chapter/verse) are dense per-node: the first ``n`` values are
    # the per-word labels (words are the lowest-numbered nodes), followed by each section
    # node's own label (57 books, 1192 chapters, 30371 verses). Keep only the word slice.
    cols: dict[str, list[str]] = {}
    for name in ("book", "chapter", "verse"):
        col = read_tf_column(tf_dir / f"{name}.tf")
        if len(col) < n:
            raise ValueError(f"LXX TF column {name!r} shorter than word slots: {len(col)} < {n}")
        cols[name] = col[:n]
    books, chaps, verses = cols["book"], cols["chapter"], cols["verse"]

    cur: tuple[str, str, str] | None = None
    buf: list[str] = []
    for w, b, c, v in zip(words, books, chaps, verses, strict=True):
        key = (b, c, v)
        if key != cur:
            if buf and cur is not None:
                yield Passage(" ".join(buf), "lxx", "koine", f"{cur[0]} {cur[1]}:{cur[2]}")
            cur, buf = key, []
        if w:
            buf.append(w)
    if buf and cur is not None:
        yield Passage(" ".join(buf), "lxx", "koine", f"{cur[0]} {cur[1]}:{cur[2]}")


# ── Passage-window chunking ─────────────────────────────────────────────────


def chunk_passages(
    passages: Iterable[Passage],
    *,
    target_words: int = 150,
    max_words: int = 300,
) -> Iterator[Passage]:
    """Merge consecutive same-source passages into ~``target_words`` windows.

    Verses / TEI fragments are individually too short to teach discourse-level
    coherence, so consecutive units from the SAME source are concatenated in reading
    order until a window reaches ``target_words`` (never exceeding ``max_words``).
    Windows never cross a source boundary. A window's ``ref`` spans "<first>–<last>".
    """
    buf: list[str] = []
    buf_words = 0
    first_ref = last_ref = ""
    cur_source: str | None = None
    cur_register = ""

    def _flush() -> Passage | None:
        if not buf:
            return None
        ref = first_ref if first_ref == last_ref else f"{first_ref}–{last_ref}"
        return Passage(" ".join(buf), cur_source or "", cur_register, ref)

    for p in passages:
        n = len(p.text.split())
        if n == 0:
            continue
        if cur_source is not None and (p.source != cur_source or buf_words + n > max_words):
            flushed = _flush()
            if flushed is not None:
                yield flushed
            buf, buf_words, first_ref = [], 0, ""
        if not buf:
            cur_source, cur_register, first_ref = p.source, p.register, p.ref
        buf.append(p.text)
        buf_words += n
        last_ref = p.ref
        if buf_words >= target_words:
            flushed = _flush()
            if flushed is not None:
                yield flushed
            buf, buf_words, first_ref = [], 0, ""

    flushed = _flush()
    if flushed is not None:
        yield flushed


# ── Continuation (prefix-LM) examples ───────────────────────────────────────


def build_continuation_examples(
    windows: Iterable[Passage],
    *,
    min_words: int = 20,
    prefix_frac: float = 0.5,
    task_prefix: str = "continue: ",
) -> Iterator[dict]:
    """Turn passage windows into prefix-LM (continuation) instruction examples.

    Splits each window at a word boundary near ``prefix_frac`` so the model learns to
    continue coherent Koine prose — the autoregressive-fluency signal missing from the
    span-infill denoise diet. Windows shorter than ``min_words`` are skipped.
    """
    for w in windows:
        toks = w.text.split()
        if len(toks) < min_words:
            continue
        cut = max(1, min(len(toks) - 1, round(len(toks) * prefix_frac)))
        prefix = " ".join(toks[:cut])
        cont = " ".join(toks[cut:])
        if not cont:
            continue
        yield {
            "task": "continuation",
            "input_text": f"{task_prefix}{prefix}",
            "target_text": cont,
            "source": w.source,
            "ref": w.ref,
        }


# ── Dedup + decontamination ─────────────────────────────────────────────────


def _norm_key(text: str) -> str:
    """Accent-stripped, casefolded, whitespace-collapsed key for dedup/decontamination."""
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch)
    )
    return " ".join(stripped.casefold().split())


def shingles(text: str, n: int = 8) -> set[str]:
    """Word ``n``-gram shingles of the normalized text (for overlap detection).

    A passage shorter than ``n`` words contributes a single whole-passage shingle so it
    can still be screened.
    """
    words = _norm_key(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_contamination_index(forbidden_texts: Iterable[str], n: int = 8) -> set[str]:
    """Build a shingle set from held-out text (e.g. the Gospel test split) to screen against."""
    idx: set[str] = set()
    for t in forbidden_texts:
        idx |= shingles(t, n)
    return idx


def is_contaminated(text: str, index: set[str], n: int = 8) -> bool:
    """True if *text* shares any ``n``-gram shingle with the forbidden ``index``."""
    if not index:
        return False
    return bool(shingles(text, n) & index)


def dedup_passages(
    passages: Iterable[Passage],
    *,
    contamination_index: set[str] | None = None,
    n: int = 8,
) -> Iterator[Passage]:
    """Drop exact/near-duplicate windows and any window contaminated by held-out text."""
    seen: set[str] = set()
    for p in passages:
        key = _norm_key(p.text)
        if not key or key in seen:
            continue
        if contamination_index is not None and is_contaminated(p.text, contamination_index, n):
            continue
        seen.add(key)
        yield p


# ── Raw-prose sources ───────────────────────────────────────────────────────

# (subdir, register). The synoptic gospels are held out of SBLGNT so the held-out
# Gospel test set never leaks into the generative pools (decontamination).
RAW_TEI_SOURCES: tuple[tuple[str, str], ...] = (
    ("first1k", "classical"),
    ("apostolic", "koine"),
    ("sblgnt", "koine"),
)
SYNOPTIC_BOOKS: tuple[str, ...] = ("Matthew", "Mark", "Luke")


def build_raw_passages(data_raw: Path | str) -> Iterator[Passage]:
    """Stream :class:`Passage` units from every raw Koine/Classical source on disk.

    LXX via the TF reader (verse units); first1k/apostolic/sblgnt via the shared TEI/txt
    extractor (``synoptiq.training.dapt._extract_text_from_dir``), with the synoptic
    gospels held out of SBLGNT. Torch is imported lazily so the pure helpers above stay
    import-light for testing.
    """
    from synoptiq.training.dapt import _extract_text_from_dir, sblgnt_stems_for_books

    data_raw = Path(data_raw)

    lxx_tf = data_raw / "lxx" / "tf" / "1935"
    if (lxx_tf / "word.tf").exists():
        yield from iter_lxx_verses(lxx_tf)

    synoptic_stems = sblgnt_stems_for_books(SYNOPTIC_BOOKS)
    for subdir, register in RAW_TEI_SOURCES:
        excl = synoptic_stems if subdir == "sblgnt" else frozenset()
        for chunk in _extract_text_from_dir(data_raw, subdir, exclude_stems=excl):
            yield Passage(chunk, subdir, register, "")
