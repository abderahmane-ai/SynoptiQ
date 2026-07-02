"""Tests for synoptiq/utils/greek.py — normalization and parsing utilities."""

from __future__ import annotations

from synoptiq.utils.greek import (
    is_greek,
    normalize_greek,
    strip_punctuation,
)


class TestNormalizeGreek:
    """Tests for normalize_greek()."""

    def test_lowercase(self) -> None:
        assert normalize_greek("Λόγος") == normalize_greek("λόγος")

    def test_strips_diacritics(self) -> None:
        # "ά" (alpha with acute) should normalize to "α"
        result = normalize_greek("ά")
        assert result == "α"

    def test_full_word_normalization(self) -> None:
        # "Βίβλος" → "βιβλοσ" (lowercased, de-accented, final sigma → medial sigma)
        result = normalize_greek("Βίβλος")
        assert result == "βιβλοσ"

    def test_sigma_normalization(self) -> None:
        # Final sigma ς → regular sigma σ
        result = normalize_greek("λόγος")
        assert result.endswith("σ")
        assert "ς" not in result

    def test_empty_string(self) -> None:
        assert normalize_greek("") == ""

    def test_idempotent(self) -> None:
        """Applying normalize_greek twice should give the same result."""
        word = "θεός"
        assert normalize_greek(normalize_greek(word)) == normalize_greek(word)

    def test_known_vocabulary(self) -> None:
        """Test key theological terms normalize correctly."""
        cases = {
            "κύριος": "κυριοσ",
            "θεός": "θεοσ",
            "Ἰησοῦς": "ιησουσ",
            "χριστός": "χριστοσ",
        }
        for word, expected in cases.items():
            assert normalize_greek(word) == expected, (
                f"normalize_greek({word!r}) = {normalize_greek(word)!r}, expected {expected!r}"
            )


class TestIsGreek:
    """Tests for is_greek()."""

    def test_greek_word_is_greek(self) -> None:
        assert is_greek("λόγος")

    def test_normalized_greek_is_greek(self) -> None:
        assert is_greek("λογοσ")

    def test_latin_is_not_greek(self) -> None:
        assert not is_greek("logos")

    def test_empty_is_not_greek(self) -> None:
        assert not is_greek("")

    def test_numeric_is_not_greek(self) -> None:
        assert not is_greek("123")

    def test_mixed_with_greek_is_greek(self) -> None:
        # A string with mostly Greek should pass; function tests Unicode block
        assert is_greek("αβγ")


class TestStripPunctuation:
    """Tests for strip_punctuation()."""

    def test_removes_leading_comma(self) -> None:
        assert strip_punctuation(",λόγος") == "λόγος"

    def test_removes_trailing_period(self) -> None:
        assert strip_punctuation("λόγος.") == "λόγος"

    def test_removes_greek_question_mark(self) -> None:
        # Greek question mark is U+037E (;)
        assert strip_punctuation("λόγος;") == "λόγος"

    def test_preserves_pure_greek(self) -> None:
        assert strip_punctuation("λόγος") == "λόγος"

    def test_empty_string(self) -> None:
        assert strip_punctuation("") == ""
