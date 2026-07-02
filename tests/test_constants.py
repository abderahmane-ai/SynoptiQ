"""Tests for constants.py — especially the Aland pericope table.

The Aland table is the most critical constant in the system.
These tests validate spot-checks against the print Synopsis.
"""

from __future__ import annotations

from synoptiq.utils.constants import (
    ALAND_PERICOPES,
    GOODACRE_FATIGUE_PERICOPES,
    PERICOPE_GENRES,
    parse_morph_tag,
)


class TestAlandPericopes:
    """Spot-check against Aland Synopsis Quattuor Evangeliorum print edition."""

    def test_pericope_001_genealogy(self) -> None:
        """001: Genealogy of Jesus — Matthew only + Luke parallel."""
        entry = ALAND_PERICOPES["001"]
        assert entry["Matthew"] == ((1, 1), (1, 17))
        assert entry["Mark"] is None
        assert entry["Luke"] == ((3, 23), (3, 38))
        assert entry["John"] is None

    def test_pericope_010_baptism(self) -> None:
        """010: Baptism of Jesus — triple tradition."""
        entry = ALAND_PERICOPES["010"]
        assert entry["Matthew"] == ((3, 13), (3, 17))
        assert entry["Mark"] == ((1, 9), (1, 11))
        assert entry["Luke"] == ((3, 21), (3, 22))
        assert entry["John"] is None

    def test_pericope_020_leper(self) -> None:
        """020: Cleansing of a Leper — triple tradition."""
        entry = ALAND_PERICOPES["020"]
        assert entry["Matthew"] == ((8, 2), (8, 4))
        assert entry["Mark"] == ((1, 40), (1, 45))
        assert entry["Luke"] == ((5, 12), (5, 16))
        assert entry["John"] is None

    def test_pericope_058_feeding_5000(self) -> None:
        """058: Feeding of the Five Thousand — quadruple tradition."""
        entry = ALAND_PERICOPES["058"]
        assert entry["Matthew"] == ((14, 13), (14, 21))
        assert entry["Mark"] == ((6, 30), (6, 44))
        assert entry["Luke"] == ((9, 10), (9, 17))
        assert entry["John"] == ((6, 1), (6, 15))

    def test_pericope_127_third_passion_prediction(self) -> None:
        """127: Third Passion Prediction — triple tradition."""
        entry = ALAND_PERICOPES["127"]
        assert entry["Matthew"] == ((20, 17), (20, 19))
        assert entry["Mark"] == ((10, 32), (10, 34))
        assert entry["Luke"] == ((18, 31), (18, 34))
        assert entry["John"] is None

    def test_pericope_147_talents_pounds(self) -> None:
        """147: Parable of the Talents / Pounds — Matthean & Lukan versions."""
        entry = ALAND_PERICOPES["147"]
        assert entry["Matthew"] == ((25, 14), (25, 30))
        # Luke has the Pounds variant at 19:11-27
        assert entry["Luke"] == ((19, 11), (19, 27))
        assert entry["Mark"] is None

    def test_pericope_183_trial_denial(self) -> None:
        """183: Trial before High Priest / Peter's Denial — quadruple tradition."""
        entry = ALAND_PERICOPES["183"]
        assert entry["Matthew"] == ((26, 57), (26, 75))
        assert entry["Mark"] == ((14, 53), (14, 72))
        assert entry["Luke"] == ((22, 54), (22, 71))
        assert entry["John"] == ((18, 12), (18, 27))

    def test_pericope_280_empty_tomb(self) -> None:
        """280: Empty Tomb — all four gospels."""
        entry = ALAND_PERICOPES["280"]
        assert entry["Matthew"] is not None
        assert entry["Mark"] is not None
        assert entry["Luke"] is not None
        assert entry["John"] is not None

    def test_all_pericopes_have_required_book_keys(self) -> None:
        """Every pericope must define all four Gospel keys (value may be None)."""
        required_keys = {"Matthew", "Mark", "Luke", "John"}
        for pid, entry in ALAND_PERICOPES.items():
            assert set(entry.keys()) == required_keys, (
                f"Pericope {pid} missing keys: {required_keys - set(entry.keys())}"
            )

    def test_verse_ranges_are_valid(self) -> None:
        """Verse ranges must have start ≤ end (chapter-by-chapter)."""
        for pid, entry in ALAND_PERICOPES.items():
            for book, verse_range in entry.items():
                if verse_range is None:
                    continue
                (start_ch, start_vs), (end_ch, end_vs) = verse_range
                assert start_ch <= end_ch, (
                    f"Pericope {pid}/{book}: start chapter {start_ch} > end chapter {end_ch}"
                )
                if start_ch == end_ch:
                    assert start_vs <= end_vs, (
                        f"Pericope {pid}/{book}: start verse {start_vs} > end verse {end_vs}"
                    )

    def test_minimum_pericope_count(self) -> None:
        """There must be at least 50 pericopes defined."""
        assert len(ALAND_PERICOPES) >= 50, f"Only {len(ALAND_PERICOPES)} pericopes defined"

    def test_pericope_044_mark_only(self) -> None:
        """044: Seed Growing Secretly — Mark only (Markan Sondergut)."""
        entry = ALAND_PERICOPES["044"]
        assert entry["Mark"] == ((4, 26), (4, 29))
        assert entry["Matthew"] is None
        assert entry["Luke"] is None

    def test_pericope_088_lords_prayer(self) -> None:
        """088: Lord's Prayer — Matthew + Luke (double tradition, potential Q)."""
        entry = ALAND_PERICOPES["088"]
        assert entry["Matthew"] == ((6, 9), (6, 13))
        assert entry["Luke"] == ((11, 1), (11, 4))
        assert entry["Mark"] is None


class TestGoodacreFatiguePericopes:
    """Validate Goodacre fatigue test-case pericope IDs are correct."""

    def test_cleansing_leper_id(self) -> None:
        pericope_id = GOODACRE_FATIGUE_PERICOPES["cleansing_leper"]["pericope_id"]
        assert pericope_id == "020"
        assert "020" in ALAND_PERICOPES

    def test_feeding_5000_id(self) -> None:
        pericope_id = GOODACRE_FATIGUE_PERICOPES["feeding_5000"]["pericope_id"]
        assert pericope_id == "058"
        assert "058" in ALAND_PERICOPES

    def test_parable_pounds_talents_id(self) -> None:
        pericope_id = GOODACRE_FATIGUE_PERICOPES["parable_pounds_talents"]["pericope_id"]
        assert pericope_id == "147"
        assert "147" in ALAND_PERICOPES


class TestMorphTagParser:
    """Tests for parse_morph_tag()."""

    def test_parse_aorist_active_indicative(self) -> None:
        result = parse_morph_tag("3AAINS--")
        assert result.get("tense") == "aorist"
        assert result.get("voice") == "active"
        assert result.get("mood") == "indicative"
        assert result.get("number") == "singular"

    def test_parse_nominative_masculine_singular_noun(self) -> None:
        # MorphGNT CCAT tag: person-tense-voice-mood-case-number-gender-degree
        # N=nominative, S=singular, M=masculine → "----NSM-"
        result = parse_morph_tag("----NSM-")
        assert result.get("case") == "nominative"
        assert result.get("number") == "singular"
        assert result.get("gender") == "masculine"

    def test_empty_features_omitted(self) -> None:
        """Unspecified features (dashes) should not appear in output."""
        result = parse_morph_tag("--------")
        assert result == {}

    def test_short_tag_handled(self) -> None:
        """Tags shorter than 8 chars should be padded with dashes without error."""
        # "3P" → padded to "3P------": person=3 (third), tense=P (present)
        result = parse_morph_tag("3P")
        assert result.get("person") == "third"
        assert result.get("tense") == "present"


class TestPericopeGenres:
    """Validate genre classification coverage."""

    def test_passion_pericopes_classified(self) -> None:
        """Key passion narrative pericopes must be labeled 'passion'."""
        for pid in ["175", "176", "177", "178", "179", "180", "181", "182", "183"]:
            assert PERICOPE_GENRES.get(pid) == "passion", (
                f"Pericope {pid} should be 'passion', got {PERICOPE_GENRES.get(pid)!r}"
            )

    def test_wisdom_parables_classified(self) -> None:
        """Key parable pericopes must be labeled 'wisdom'."""
        for pid in ["044", "049", "112"]:
            assert PERICOPE_GENRES.get(pid) == "wisdom", (
                f"Pericope {pid} should be 'wisdom', got {PERICOPE_GENRES.get(pid)!r}"
            )
