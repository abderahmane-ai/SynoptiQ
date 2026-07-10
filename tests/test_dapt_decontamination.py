"""Tests for DAPT text decontamination (holding the synoptic gospels out)."""

from __future__ import annotations

from pathlib import Path

from synoptiq.training.dapt import (
    DAPTConfig,
    _extract_text_from_dir,
    sblgnt_stems_for_books,
)

def _greek_with_marker(marker: str) -> str:
    # One sentence (single trailing period) so the chunker keeps it whole, with a
    # distinctive Latin marker embedded mid-sentence alongside real Greek so it
    # survives both the length filter and the Greek-content filter.
    return f"ἐν ἀρχῇ ἦν ὁ λόγος {marker} καὶ ὁ λόγος ἦν πρὸς τὸν θεόν."


def _make_sblgnt(tmp_path: Path) -> Path:
    src = tmp_path / "sblgnt"
    src.mkdir()
    for stem in ("Matt", "Mark", "Luke", "John", "Rom", "1Cor"):
        (src / f"{stem}.txt").write_text(_greek_with_marker(f"bk{stem}"), encoding="utf-8")
    return tmp_path


def test_stems_map_gospels_to_sblgnt_names() -> None:
    assert sblgnt_stems_for_books(("Matthew", "Mark", "Luke")) == {"Matt", "Mark", "Luke"}
    # unknown book names pass through unchanged
    assert sblgnt_stems_for_books(("Rom",)) == {"Rom"}


def test_extract_without_exclusion_reads_all_books(tmp_path: Path) -> None:
    data_dir = _make_sblgnt(tmp_path)
    joined = " ".join(_extract_text_from_dir(data_dir, "sblgnt"))
    for stem in ("Matt", "Mark", "Luke", "John", "Rom", "1Cor"):
        assert f"bk{stem}" in joined


def test_extract_excludes_synoptic_gospels(tmp_path: Path) -> None:
    data_dir = _make_sblgnt(tmp_path)
    stems = sblgnt_stems_for_books(("Matthew", "Mark", "Luke"))
    joined = " ".join(_extract_text_from_dir(data_dir, "sblgnt", exclude_stems=stems))
    # gospels gone
    for stem in ("Matt", "Mark", "Luke"):
        assert f"bk{stem}" not in joined
    # everything else (including John) retained
    for stem in ("John", "Rom", "1Cor"):
        assert f"bk{stem}" in joined


def test_excludes_apparatus_copy_by_stem(tmp_path: Path) -> None:
    # SBLGNT ships a second Matt/Mark/Luke under sblgntapp/; the stem filter must
    # catch it too, regardless of directory.
    data_dir = _make_sblgnt(tmp_path)
    app = data_dir / "sblgnt" / "sblgntapp"
    app.mkdir()
    (app / "Matt.txt").write_text(_greek_with_marker("bkMattApp"), encoding="utf-8")
    stems = sblgnt_stems_for_books(("Matthew", "Mark", "Luke"))
    joined = " ".join(_extract_text_from_dir(data_dir, "sblgnt", exclude_stems=stems))
    assert "bkMattApp" not in joined


def test_dapt_config_defaults_to_no_exclusion() -> None:
    assert DAPTConfig().exclude_books == ()
    assert DAPTConfig(exclude_books=("Matthew", "Mark", "Luke")).exclude_books == (
        "Matthew", "Mark", "Luke",
    )
