"""Tests for the dependency-free Text-Fabric feature reader.

The compact-encoding expansion is the load-bearing part: sparse morphology features
(``case``/``tense``/…) skip nodes that lack the feature, so a naive reader misaligns
every token after the first gap. These tests pin the three line forms and slot count.
"""

from __future__ import annotations

from pathlib import Path

from synoptiq.reader.textfabric import TFDataset, read_tf_feature, slot_count


def _write(tmp_path: Path, name: str, data: str) -> Path:
    p = tmp_path / f"{name}.tf"
    p.write_text(f"@node\n@valueType=str\n\n{data}", encoding="utf-8")
    return p


def test_bare_values_increment(tmp_path: Path) -> None:
    p = _write(tmp_path, "f", "a\nb\nc")
    assert read_tf_feature(p) == {1: "a", 2: "b", 3: "c"}


def test_explicit_node(tmp_path: Path) -> None:
    p = _write(tmp_path, "f", "5\tx")
    assert read_tf_feature(p) == {5: "x"}


def test_inclusive_range(tmp_path: Path) -> None:
    p = _write(tmp_path, "f", "2-4\ty")
    assert read_tf_feature(p) == {2: "y", 3: "y", 4: "y"}


def test_mixed_compact_alignment(tmp_path: Path) -> None:
    # slot 1 has no value; slot 2 explicit; slot 3 bare (=prev+1); slot 5 explicit.
    # This is exactly the pattern that trips the naive reader (a caseless word 1).
    p = _write(tmp_path, "f", "2\tdative\nnominative\n5\tnominative")
    assert read_tf_feature(p) == {2: "dative", 3: "nominative", 5: "nominative"}


def test_empty_value_line_still_advances_node(tmp_path: Path) -> None:
    # A bare empty line is a node with no value: node advances, nothing is stored.
    p = _write(tmp_path, "f", "a\n\nc")
    assert read_tf_feature(p) == {1: "a", 3: "c"}


def test_header_lines_skipped(tmp_path: Path) -> None:
    p = tmp_path / "f.tf"
    p.write_text("@node\n@author=x\n@description=y\n@valueType=str\n\nval", encoding="utf-8")
    assert read_tf_feature(p) == {1: "val"}


def test_slot_count_from_otype(tmp_path: Path) -> None:
    _write(tmp_path, "otype", "1-4\tword\n5\tsentence")
    assert slot_count(tmp_path) == 4


def test_tfdataset_value_and_has(tmp_path: Path) -> None:
    _write(tmp_path, "otype", "1-3\tword")
    _write(tmp_path, "text", "α\nβ\nγ")
    ds = TFDataset(tmp_path)
    assert ds.n_slots == 3
    assert ds.has("text") and not ds.has("gloss")
    assert ds.value("text", 2) == "β"
    assert ds.value("gloss", 2) == ""  # absent feature → empty


def test_tfdataset_requires_otype(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        TFDataset(tmp_path)
