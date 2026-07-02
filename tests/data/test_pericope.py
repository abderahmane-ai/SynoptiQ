"""Tests for synoptiq/data/pericope.py — pericope classification."""

from __future__ import annotations

from synoptiq.data.pericope import classify_tradition, get_genre, get_pericope_books


class TestClassifyTradition:
    def test_triple(self) -> None:
        assert classify_tradition(frozenset({"Matthew", "Mark", "Luke"})) == "triple"

    def test_double(self) -> None:
        assert classify_tradition(frozenset({"Matthew", "Luke"})) == "double"

    def test_mark_unique(self) -> None:
        assert classify_tradition(frozenset({"Mark"})) == "mark_unique"

    def test_matthean_unique(self) -> None:
        assert classify_tradition(frozenset({"Matthew"})) == "matthean_unique"

    def test_lukan_unique(self) -> None:
        assert classify_tradition(frozenset({"Luke"})) == "lukan_unique"

    def test_triple_with_john(self) -> None:
        assert classify_tradition(frozenset({"Matthew", "Mark", "Luke", "John"})) == "triple"

    def test_mark_matthew_partial(self) -> None:
        assert classify_tradition(frozenset({"Matthew", "Mark"})) == "triple"


class TestGetPericopeBooks:
    def test_020_triple(self) -> None:
        assert get_pericope_books("020") == frozenset({"Matthew", "Mark", "Luke"})

    def test_044_mark_only(self) -> None:
        assert get_pericope_books("044") == frozenset({"Mark"})

    def test_088_double(self) -> None:
        assert get_pericope_books("088") == frozenset({"Matthew", "Luke"})

    def test_unknown_raises_keyerror(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            get_pericope_books("99999")


class TestGetGenre:
    def test_narrative(self) -> None:
        assert get_genre("020") == "narrative"

    def test_discourse(self) -> None:
        assert get_genre("088") == "discourse"

    def test_passion(self) -> None:
        assert get_genre("175") == "passion"

    def test_wisdom(self) -> None:
        assert get_genre("044") == "wisdom"

    def test_unknown_defaults_to_other(self) -> None:
        assert get_genre("99999") == "other"
