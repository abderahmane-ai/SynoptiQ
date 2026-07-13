"""Tests for human-readable morphology formatting."""

from __future__ import annotations

from synoptiq.reader.morphology import describe_morphology, tidy_pos, tidy_value


def test_tidy_pos_maps_known_and_passes_unknown() -> None:
    assert tidy_pos("subs") == "noun"
    assert tidy_pos("art") == "article"
    assert tidy_pos("verb") == "verb"
    assert tidy_pos("weird") == "weird"
    assert tidy_pos("") == ""


def test_person_values_normalised() -> None:
    assert tidy_value("person", "p3") == "3rd person"
    assert tidy_value("person", "third") == "3rd person"
    assert tidy_value("person", "1") == "1st person"
    # non-person features pass through untouched
    assert tidy_value("case", "nominative") == "nominative"


def test_describe_orders_nominal_features() -> None:
    features = {"number": "singular", "case": "nominative", "gender": "masculine"}
    assert describe_morphology(features) == "nominative · masculine · singular"


def test_describe_orders_verbal_features() -> None:
    features = {
        "mood": "indicative",
        "voice": "active",
        "tense": "imperfect",
        "person": "p3",
        "number": "singular",
    }
    assert describe_morphology(features) == (
        "3rd person · imperfect · active · indicative · singular"
    )


def test_describe_empty() -> None:
    assert describe_morphology({}) == ""
    assert describe_morphology({"case": ""}) == ""
