"""Gold reading engine — reference lookup over the GNT/LXX Text-Fabric data.

:class:`GoldReader` indexes a Text-Fabric dataset by ``(book, chapter, verse)`` and
returns, per word, a :class:`WordAnalysis`: surface form, lemma, part of speech,
full morphology, English gloss, and Strong's number — all *gold* (human-curated),
so there is no model error and no inference cost.

The picker / reference-resolution / lexicon logic lives in :class:`_ReaderBase`, shared
with the artifact-backed :class:`~synoptiq.reader.index.IndexReader`; subclasses only
supply :meth:`verse`. Two on-disk schemas are supported — Nestle-1904 GNT
(:data:`N1904_SCHEMA`) and Rahlfs LXX (:data:`LXX_SCHEMA`) — auto-detected by
:meth:`GoldReader.from_dir`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from synoptiq.reader.morphology import describe_morphology, tidy_pos
from synoptiq.reader.textfabric import TFDataset
from synoptiq.utils.greek import strip_accents

# ── Per-word record ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WordAnalysis:
    """A single analysed word, from either the gold or the neural engine.

    ``after`` is the trailing whitespace/punctuation that renders flowing reading
    text (``surface + after``); ``predicted`` is True only for neural-engine output.
    """

    surface: str
    lemma: str
    pos: str
    features: dict[str, str] = field(default_factory=dict)
    gloss: str = ""
    strong: str = ""
    after: str = " "
    ref: str = ""
    predicted: bool = False

    @property
    def morphology(self) -> str:
        """Readable morphology string (e.g. ``"genitive · masculine · singular"``)."""
        return describe_morphology(self.features)


@dataclass(frozen=True)
class ReadResult:
    """A resolved passage: the reference label plus its analysed words."""

    ref: str
    words: list[WordAnalysis]

    @property
    def text(self) -> str:
        """The flowing Greek reading text (surface forms + trailing material)."""
        return "".join(w.surface + w.after for w in self.words).strip()

    def __bool__(self) -> bool:
        return bool(self.words)


# ── Reference parsing ─────────────────────────────────────────────────────────

# "Matt 5:3", "1 John 1:1-3", "Mark 1", "John 1:1"
_REF_RE = re.compile(
    r"^\s*(?P<book>[1-3]?\s?[A-Za-zΑ-Ωα-ω.]+)\s+"
    r"(?P<ch>\d+)"
    r"(?::(?P<v1>\d+)(?:-(?P<v2>\d+))?)?\s*$"
)

# Common abbreviations → the full English book names used by the N1904 edition.
_BOOK_ALIASES: dict[str, str] = {
    "matt": "Matthew", "mt": "Matthew", "mk": "Mark", "mrk": "Mark",
    "lk": "Luke", "luk": "Luke", "jn": "John", "jhn": "John",
    "rom": "Romans", "gal": "Galatians", "eph": "Ephesians", "php": "Philippians",
    "phil": "Philippians", "col": "Colossians", "heb": "Hebrews", "jas": "James",
    "rev": "Revelation", "apoc": "Revelation",
}


def parse_reference(ref: str) -> tuple[str, int, int | None, int | None]:
    """Parse a reference into ``(book, chapter, start_verse, end_verse)``.

    ``start_verse``/``end_verse`` are None for a whole-chapter reference; ``end_verse``
    is None for a single verse.

    Raises:
        ValueError: If ``ref`` cannot be parsed.
    """
    m = _REF_RE.match(ref)
    if not m:
        msg = f"Cannot parse reference: {ref!r}"
        raise ValueError(msg)
    book = m.group("book").strip()
    chapter = int(m.group("ch"))
    v1 = int(m.group("v1")) if m.group("v1") else None
    v2 = int(m.group("v2")) if m.group("v2") else None
    return book, chapter, v1, v2


# ── Shared reader base ────────────────────────────────────────────────────────


class _ReaderBase:
    """Reference resolution, pickers, and lexicon shared by gold/index readers.

    Subclasses populate the index maps (``_book_order``, ``_chapters``, ``_verses``,
    ``_book_cv``, ``_book_lookup``) and implement :meth:`verse`.
    """

    _book_order: list[str]
    _chapters: dict[str, list[int]]
    _verses: dict[tuple[str, int], list[int]]
    _book_cv: dict[str, list[tuple[int, int]]]
    _book_lookup: dict[str, str]
    _lexicon: dict[str, tuple[str, str]] | None = None

    # ── pickers ────────────────────────────────────────────────────────────────

    def books(self) -> list[str]:
        """Book names in canonical (corpus) order."""
        return list(self._book_order)

    def chapters(self, book: str) -> list[int]:
        """Chapter numbers present in ``book``."""
        return self._chapters.get(self._canon_book(book), [])

    def verses(self, book: str, chapter: int) -> list[int]:
        """Verse numbers present in ``book`` chapter ``chapter``."""
        return self._verses.get((self._canon_book(book), chapter), [])

    def _canon_book(self, book: str) -> str:
        key = strip_accents(book).lower().strip()
        if key in self._book_lookup:
            return self._book_lookup[key]
        alias = _BOOK_ALIASES.get(key)
        if alias and strip_accents(alias).lower() in self._book_lookup:
            return alias
        return book  # unknown; downstream lookups yield empty

    # ── lookups ────────────────────────────────────────────────────────────────

    def verse(self, book: str, chapter: int, verse: int) -> list[WordAnalysis]:
        """Return the analysed words of a single verse (subclass-provided)."""
        raise NotImplementedError

    def passage(
        self,
        book: str,
        start: tuple[int, int],
        end: tuple[int, int] | None = None,
    ) -> ReadResult:
        """Return an analysed passage spanning ``start``..``end`` ``(chapter, verse)``.

        Ranges may cross chapter boundaries. ``end`` defaults to ``start`` (one verse).
        """
        b = self._canon_book(book)
        end = end or start
        words: list[WordAnalysis] = []
        for cv in self._book_cv.get(b, []):
            if start <= cv <= end:
                words.extend(self.verse(b, cv[0], cv[1]))
        return ReadResult(self._range_label(b, start, end), words)

    def read(self, ref: str) -> ReadResult:
        """Resolve a reference string (``"John 1:1"``, ``"Mark 1:1-3"``, ``"Matt 5"``).

        A bare ``"Book chapter"`` returns the whole chapter.

        Raises:
            ValueError: If ``ref`` cannot be parsed.
        """
        book_raw, chapter, v1, v2 = parse_reference(ref)
        book = self._canon_book(book_raw)
        if v1 is None:  # whole chapter
            vs = self.verses(book, chapter)
            if not vs:
                return ReadResult(f"{book} {chapter}", [])
            return self.passage(book, (chapter, vs[0]), (chapter, vs[-1]))
        return self.passage(book, (chapter, v1), (chapter, v2 or v1))

    @staticmethod
    def _range_label(book: str, start: tuple[int, int], end: tuple[int, int]) -> str:
        (c1, v1), (c2, v2) = start, end
        if start == end:
            return f"{book} {c1}:{v1}"
        if c1 == c2:
            return f"{book} {c1}:{v1}-{v2}"
        return f"{book} {c1}:{v1}-{c2}:{v2}"

    # ── lexicon (powers neural-mode glossing) ──────────────────────────────────

    def lexicon(self) -> dict[str, tuple[str, str]]:
        """Harvest an accent-stripped ``lemma → (gloss, strong)`` map from all words.

        The first non-empty gloss seen for a lemma wins. Keys are accent-stripped and
        lower-cased so neural-mode predicted lemmas match regardless of accentuation.
        Computed once and cached on the instance.
        """
        if self._lexicon is not None:
            return self._lexicon
        lex: dict[str, tuple[str, str]] = {}
        for b in self._book_order:
            for c, v in self._book_cv.get(b, []):
                for w in self.verse(b, c, v):
                    if not w.lemma:
                        continue
                    key = strip_accents(w.lemma).lower()
                    if key in lex and lex[key][0]:
                        continue
                    lex[key] = (w.gloss, w.strong)
        self._lexicon = lex
        return lex

    def gloss_for(self, lemma: str) -> tuple[str, str]:
        """Look up ``(gloss, strong)`` for a (possibly accented) lemma; ``("","")`` if absent."""
        return self.lexicon().get(strip_accents(lemma).lower(), ("", ""))


# ── Dataset schema ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TFSchema:
    """Maps canonical reader fields onto edition-specific Text-Fabric feature names.

    ``lemma``/``gloss``/``strong``/``after`` are tuples of candidate feature names,
    tried in order (the first that exists in the dataset wins). ``feature_map`` maps
    canonical morphology keys onto this edition's feature names.
    """

    surface: str
    lemma: tuple[str, ...]
    gloss: tuple[str, ...]
    strong: tuple[str, ...]
    after: tuple[str, ...]
    pos: str
    feature_map: dict[str, str]


N1904_SCHEMA = TFSchema(
    surface="text",
    lemma=("lemma",),
    gloss=("gloss",),
    strong=("strong",),
    after=("trailer", "after"),
    pos="sp",
    feature_map={
        "person": "person", "tense": "tense", "voice": "voice", "mood": "mood",
        "case": "case", "gender": "gender", "number": "number", "degree": "degree",
    },
)

LXX_SCHEMA = TFSchema(
    surface="word",
    lemma=("lex", "bol_lexeme_dict", "freq_lemma"),
    gloss=("gloss", "bol_gloss"),
    strong=("strongs", "strong"),
    after=("trailer", "after"),
    pos="sp",
    feature_map={
        "person": "ps", "tense": "tense", "voice": "voice", "mood": "mood",
        "case": "case", "gender": "gn", "number": "nu", "degree": "degree",
    },
)


# ── Gold (Text-Fabric-backed) reader ──────────────────────────────────────────


class GoldReader(_ReaderBase):
    """Reference-indexed access to a gold GNT/LXX Text-Fabric dataset.

    Build with :meth:`from_dir`. Look words up with :meth:`read` (a reference string),
    :meth:`verse`, or :meth:`passage`; drive UI pickers with :meth:`books`,
    :meth:`chapters`, and :meth:`verses`.
    """

    def __init__(self, dataset: TFDataset, schema: TFSchema, name: str = "") -> None:
        self.ds = dataset
        self.schema = schema
        self.name = name
        self._lemma_feat = self._first_present(schema.lemma)
        self._gloss_feat = self._first_present(schema.gloss)
        self._strong_feat = self._first_present(schema.strong)
        self._after_feat = self._first_present(schema.after)
        self._lexicon = None
        self._index()

    @classmethod
    def from_dir(
        cls,
        tf_dir: Path | str,
        *,
        schema: TFSchema | None = None,
        name: str = "",
    ) -> GoldReader:
        """Load a reader from a TF directory, auto-detecting the edition schema.

        Args:
            tf_dir: Directory of ``.tf`` files (e.g. ``data/raw/n1904/tf/1.0.0``).
            schema: Override the auto-detected schema.
            name: Corpus label; defaults to ``"n1904"`` or ``"lxx"``.
        """
        ds = TFDataset(tf_dir)
        if schema is None:
            schema = N1904_SCHEMA if ds.has("text") else LXX_SCHEMA
        if not name:
            name = "n1904" if schema is N1904_SCHEMA else "lxx"
        return cls(ds, schema, name)

    def _first_present(self, candidates: tuple[str, ...]) -> str:
        for feat in candidates:
            if self.ds.has(feat):
                return feat
        return candidates[0] if candidates else ""

    def _index(self) -> None:
        """Scan word slots once, building book/chapter/verse → slot lookups."""
        book = self.ds.feature("book")
        chap = self.ds.feature("chapter")
        vrs = self.ds.feature("verse")

        self._verse_slots: dict[tuple[str, int, int], list[int]] = {}
        self._book_order = []
        seen_book: set[str] = set()
        chapters: dict[str, set[int]] = {}
        verses: dict[tuple[str, int], set[int]] = {}

        for slot in range(1, self.ds.n_slots + 1):
            b = book.get(slot)
            c_raw = chap.get(slot)
            v_raw = vrs.get(slot)
            if not b or not c_raw or not v_raw:
                continue
            c, v = int(c_raw), int(v_raw)
            self._verse_slots.setdefault((b, c, v), []).append(slot)
            if b not in seen_book:
                seen_book.add(b)
                self._book_order.append(b)
            chapters.setdefault(b, set()).add(c)
            verses.setdefault((b, c), set()).add(v)

        self._chapters = {b: sorted(cs) for b, cs in chapters.items()}
        self._verses = {bc: sorted(vs) for bc, vs in verses.items()}
        self._book_cv = {}
        for (b, c, v) in self._verse_slots:
            self._book_cv.setdefault(b, []).append((c, v))
        for b in self._book_cv:
            self._book_cv[b].sort()
        self._book_lookup = {strip_accents(b).lower(): b for b in self._book_order}

    def _word(self, slot: int, ref: str) -> WordAnalysis:
        ds, sc = self.ds, self.schema
        features = {
            canon: ds.value(feat, slot)
            for canon, feat in sc.feature_map.items()
            if ds.value(feat, slot)
        }
        after = ds.value(self._after_feat, slot) if self._after_feat else ""
        return WordAnalysis(
            surface=ds.value(sc.surface, slot),
            lemma=ds.value(self._lemma_feat, slot),
            pos=tidy_pos(ds.value(sc.pos, slot)),
            features=features,
            gloss=ds.value(self._gloss_feat, slot),
            strong=ds.value(self._strong_feat, slot),
            after=after or " ",
            ref=ref,
        )

    def verse(self, book: str, chapter: int, verse: int) -> list[WordAnalysis]:
        """Return the analysed words of a single verse (empty if not found)."""
        b = self._canon_book(book)
        ref = f"{b} {chapter}:{verse}"
        return [self._word(s, ref) for s in self._verse_slots.get((b, chapter, verse), [])]
