"""Tests for synoptic-parallel lookup over the Aland pericope table.

Uses the real :data:`ALAND_PERICOPES` constant (no large data needed). Pericope 008
(Baptism of Jesus) is the anchor: Matt 3:13-17 · Mark 1:9-11 · Luke 3:21-22.
"""

from __future__ import annotations

from synoptiq.reader.parallels import find_pericope, format_range, parallel_ranges


def test_find_pericope_contains_verse() -> None:
    # Aland 010 = Baptism of Jesus (Matt 3:13-17 · Mark 1:9-11 · Luke 3:21-22).
    assert find_pericope("Mark", 1, 9) == "010"
    assert find_pericope("Matthew", 3, 15) == "010"
    assert find_pericope("Luke", 3, 22) == "010"


def test_find_pericope_none_for_unknown_verse() -> None:
    assert find_pericope("Mark", 99, 99) is None


def test_parallel_ranges_excludes_self_and_absent() -> None:
    par = parallel_ranges("Mark", 1, 9)
    assert set(par) == {"Matthew", "Luke"}
    assert par["Matthew"] == ((3, 13), (3, 17))
    assert par["Luke"] == ((3, 21), (3, 22))
    assert "Mark" not in par
    assert "John" not in par  # Baptism absent from John → omitted


def test_parallel_ranges_empty_when_no_pericope() -> None:
    assert parallel_ranges("Mark", 99, 99) == {}


def test_format_range() -> None:
    assert format_range(((3, 23), (3, 38))) == "3:23-38"
    assert format_range(((1, 1), (1, 1))) == "1:1"
    assert format_range(((26, 69), (27, 2))) == "26:69-27:2"
