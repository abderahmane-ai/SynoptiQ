"""Tests for the gold reading engine.

Most tests build a hermetic 5-word synthetic Text-Fabric dataset in ``tmp_path`` so
they run in CI without the git-ignored ``data/`` tree. A synthetic corpus also lets
us assert the compact-encoding alignment through the full stack: word 1 (``Ἐν``, a
preposition) has no case, so a naive reader would shift every following case by one.

Two verses (John 1:1-2):
    Ἐν ἀρχῇ λόγος. / ἦν λόγος.
with deliberately sparse morphology (only nominals get case; only the verb gets
person/tense/voice/mood).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from synoptiq.reader.gold import GoldReader


def _nfc(text: str) -> str:
    """NFC-normalise so comparisons are robust to the data's mixed accent encoding."""
    return unicodedata.normalize("NFC", text)

# feature name → data section (compact TF encoding where sparse).
_MINI: dict[str, str] = {
    "otype": "1-5\tword\n6\tsentence",
    "text": "Ἐν\nἀρχῇ\nλόγος\nἦν\nλόγος",
    "sp": "prep\nsubs\nsubs\nverb\nsubs",
    "lemma": "ἐν\nἀρχή\nλόγος\nεἰμί\nλόγος",
    "gloss": "in\nbeginning\nword, speech\nam, exist\nword, speech",
    "strong": "1722\n746\n3056\n1510\n3056",
    "trailer": " \n \n. \n \n. ",
    "book": "John\nJohn\nJohn\nJohn\nJohn",
    "chapter": "1\n1\n1\n1\n1",
    "verse": "1\n1\n1\n2\n2",
    "case": "2\tdative\nnominative\n5\tnominative",
    "gender": "2\tfeminine\nmasculine\n5\tmasculine",
    "number": "2\tsingular\nsingular\nsingular\nsingular",
    "person": "4\tp3",
    "tense": "4\timperfect",
    "voice": "4\tactive",
    "mood": "4\tindicative",
}


def _mini(tmp_path: Path) -> Path:
    d = tmp_path / "tf"
    d.mkdir()
    for name, data in _MINI.items():
        (d / f"{name}.tf").write_text(f"@node\n@valueType=str\n\n{data}", encoding="utf-8")
    return d


def test_pickers(tmp_path: Path) -> None:
    r = GoldReader.from_dir(_mini(tmp_path))
    assert r.books() == ["John"]
    assert r.chapters("John") == [1]
    assert r.verses("John", 1) == [1, 2]


def test_read_verse_text_and_words(tmp_path: Path) -> None:
    r = GoldReader.from_dir(_mini(tmp_path))
    res = r.read("John 1:1")
    assert res.ref == "John 1:1"
    assert [w.surface for w in res.words] == ["Ἐν", "ἀρχῇ", "λόγος"]
    assert res.text == "Ἐν ἀρχῇ λόγος."


def test_compact_morphology_alignment(tmp_path: Path) -> None:
    # The crux: word 1 has no case, so misalignment would give ἀρχῇ 'nominative'.
    r = GoldReader.from_dir(_mini(tmp_path))
    words = r.read("John 1:1").words
    assert words[0].pos == "preposition" and words[0].morphology == ""
    assert words[1].surface == "ἀρχῇ"
    assert words[1].pos == "noun"
    assert words[1].morphology == "dative · feminine · singular"
    assert words[2].morphology == "nominative · masculine · singular"


def test_verb_full_parse_and_gloss(tmp_path: Path) -> None:
    r = GoldReader.from_dir(_mini(tmp_path))
    verb = r.read("John 1:2").words[0]
    assert verb.surface == "ἦν"
    assert verb.pos == "verb"
    assert verb.morphology == "3rd person · imperfect · active · indicative · singular"
    assert verb.lemma == "εἰμί"
    assert verb.gloss == "am, exist"
    assert verb.strong == "1510"


def test_abbreviation_and_multiverse_range(tmp_path: Path) -> None:
    r = GoldReader.from_dir(_mini(tmp_path))
    res = r.read("Jn 1:1-2")  # abbreviation + range
    assert res.ref == "John 1:1-2"
    assert len(res.words) == 5


def test_whole_chapter(tmp_path: Path) -> None:
    r = GoldReader.from_dir(_mini(tmp_path))
    res = r.read("John 1")
    assert len(res.words) == 5


def test_lexicon_is_accent_insensitive(tmp_path: Path) -> None:
    r = GoldReader.from_dir(_mini(tmp_path))
    assert r.gloss_for("λόγος") == ("word, speech", "3056")
    assert r.gloss_for("λογος") == ("word, speech", "3056")  # de-accented lookup
    assert r.gloss_for("nonexistent") == ("", "")


def test_bad_reference_raises(tmp_path: Path) -> None:
    r = GoldReader.from_dir(_mini(tmp_path))
    with pytest.raises(ValueError, match="Cannot parse reference"):
        r.read("not a reference")


def test_index_reader_matches_gold(tmp_path: Path) -> None:
    from synoptiq.reader.index import IndexReader, save_index, serialize

    gold = GoldReader.from_dir(_mini(tmp_path))
    index = IndexReader(serialize(gold))
    # same pickers, same resolved text + morphology + gloss
    assert index.books() == gold.books()
    assert index.chapters("John") == gold.chapters("John")
    g_res, i_res = gold.read("John 1:1"), index.read("John 1:1")
    assert i_res.ref == g_res.ref
    assert i_res.text == g_res.text
    assert [w.morphology for w in i_res.words] == [w.morphology for w in g_res.words]
    assert index.gloss_for("λόγος") == gold.gloss_for("λόγος")
    # round-trip through a gzipped file
    path = save_index(gold, tmp_path / "idx.json.gz")
    reloaded = IndexReader.from_file(path)
    assert reloaded.read("Jn 1:2").text == gold.read("Jn 1:2").text


# ── Optional checks against the real Nestle-1904 data, when present ──────────────

_N1904 = Path("data/raw/n1904/tf/1.0.0")


@pytest.mark.skipif(not _N1904.exists(), reason="N1904 gold data not present")
def test_real_n1904_john_1_1() -> None:
    r = GoldReader.from_dir(_N1904)
    assert len(r.books()) == 27
    res = r.read("John 1:1")
    assert _nfc(res.text).startswith(_nfc("Ἐν ἀρχῇ ἦν ὁ Λόγος"))
    first = res.words[0]
    assert _nfc(first.surface) == _nfc("Ἐν") and first.pos == "preposition"


@pytest.mark.skipif(not _N1904.exists(), reason="N1904 gold data not present")
def test_real_n1904_lexicon_size() -> None:
    r = GoldReader.from_dir(_N1904)
    assert len(r.lexicon()) > 4000  # ~5.4k distinct lemmas in the GNT
