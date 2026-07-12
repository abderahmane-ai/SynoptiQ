"""Tests for the Koine-T5-Hexapla corpus builder (synoptiq/data/koine_corpus.py).

These exercise the pure helpers with synthetic inputs — no torch, no GPU, no dependence
on the multi-hundred-MB on-disk corpora — so they run in the standard suite.
"""

from __future__ import annotations

from pathlib import Path

from synoptiq.data.koine_corpus import (
    Passage,
    build_continuation_examples,
    build_contamination_index,
    chunk_passages,
    dedup_passages,
    is_contaminated,
    iter_lxx_verses,
    read_tf_column,
    shingles,
)


def _write_tf(path: Path, values: list[str]) -> None:
    """Write a minimal Text-Fabric @node feature file: header, blank line, then values."""
    header = "@node\n@valueType=str\n@writtenBy=test\n"
    path.write_text(header + "\n" + "\n".join(values) + "\n", encoding="utf-8")


def test_read_tf_column_skips_header_and_blank(tmp_path: Path) -> None:
    p = tmp_path / "word.tf"
    _write_tf(p, ["ἐν", "ἀρχῇ", "ἐποίησεν"])
    assert read_tf_column(p) == ["ἐν", "ἀρχῇ", "ἐποίησεν"]


def test_iter_lxx_verses_groups_and_truncates_node_labels(tmp_path: Path) -> None:
    # 5 word slots across two verses; the section columns carry per-word labels plus
    # trailing section-node labels, which must be truncated to stay aligned with the words.
    _write_tf(tmp_path / "word.tf", ["w1", "w2", "w3", "w4", "w5"])
    _write_tf(tmp_path / "book.tf", ["Gen", "Gen", "Gen", "Gen", "Gen", "Gen"])  # +1 book node
    _write_tf(tmp_path / "chapter.tf", ["1", "1", "1", "1", "1", "1"])  # +1 chapter node
    _write_tf(tmp_path / "verse.tf", ["1", "1", "2", "2", "2", "1", "2"])  # +2 verse nodes

    verses = list(iter_lxx_verses(tmp_path))
    assert [v.ref for v in verses] == ["Gen 1:1", "Gen 1:2"]
    assert [v.text for v in verses] == ["w1 w2", "w3 w4 w5"]
    assert all(v.source == "lxx" and v.register == "koine" for v in verses)


def test_chunk_passages_respects_target_max_and_source_boundary() -> None:
    # Two koine verses of 3 words each, then a classical fragment.
    passages = [
        Passage("a b c", "lxx", "koine", "V1"),
        Passage("d e f", "lxx", "koine", "V2"),
        Passage("x y z", "first1k", "classical", ""),
    ]
    windows = list(chunk_passages(passages, target_words=6, max_words=6))
    # First two merge to a 6-word window (hits target); classical never merges with koine.
    assert windows[0].text == "a b c d e f"
    assert windows[0].ref == "V1–V2"
    assert windows[0].source == "lxx"
    assert windows[1].text == "x y z"
    assert windows[1].source == "first1k"


def test_chunk_passages_max_words_forces_flush() -> None:
    passages = [Passage("a b c d", "lxx", "koine", f"V{i}") for i in range(3)]
    windows = list(chunk_passages(passages, target_words=100, max_words=6))
    # 4 + 4 = 8 > 6, so each 4-word verse flushes on its own.
    assert [w.text for w in windows] == ["a b c d", "a b c d", "a b c d"]


def test_build_continuation_examples_split_and_min_words() -> None:
    win = Passage(" ".join(str(i) for i in range(10)), "lxx", "koine", "V1")
    short = Passage("only three words", "lxx", "koine", "V2")
    examples = list(build_continuation_examples([win, short], min_words=5, prefix_frac=0.5))
    assert len(examples) == 1  # the 3-word passage is skipped (< min_words)
    ex = examples[0]
    assert ex["task"] == "continuation"
    assert ex["input_text"] == "continue: 0 1 2 3 4"
    assert ex["target_text"] == "5 6 7 8 9"


def test_shingles_and_contamination_detection() -> None:
    forbidden = "μακαριοι οι πτωχοι τω πνευματι οτι αυτων εστιν η βασιλεια"
    index = build_contamination_index([forbidden], n=5)
    # Accent/case variant of an overlapping span is still caught (normalized shingles).
    assert is_contaminated("Μακάριοι οἱ πτωχοὶ τῷ πνεύματι ὅτι αὐτῶν", index, n=5)
    assert not is_contaminated("ἐν ἀρχῇ ἐποίησεν ὁ θεὸς τὸν οὐρανόν", index, n=5)


def test_shingles_short_passage_is_single_shingle() -> None:
    assert shingles("alpha beta", n=8) == {"alpha beta"}
    assert shingles("", n=8) == set()


def test_dedup_passages_drops_duplicates_and_contaminated() -> None:
    index = build_contamination_index(["forbidden secret phrase here now today"], n=4)
    passages = [
        Passage("ἐν ἀρχῇ ἦν ὁ λόγος", "lxx", "koine", "A"),
        Passage("ἐν ἀρχῇ ἦν ὁ λόγος", "lxx", "koine", "B"),  # exact dup
        Passage("forbidden secret phrase here now today", "first1k", "classical", "C"),
    ]
    kept = list(dedup_passages(passages, contamination_index=index, n=4))
    assert len(kept) == 1
    assert kept[0].ref == "A"
