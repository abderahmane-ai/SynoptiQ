"""Human-readable formatting of gold morphology features.

The Nestle-1904 Text-Fabric edition ships morphology *pre-split* into separate
features (``sp``/``case``/``gender``/``number``/``person``/``tense``/``voice``/
``mood``/``degree``) with spelled-out values (``nominative``, ``feminine``, …), so
no morph-code decoding is required — only light tidying (``subs`` → ``noun``,
``p3`` → ``3rd person``) and assembly into a readable string. The Rahlfs LXX
edition reuses the same value vocabulary under different feature *names*
(``gn``/``nu``/``ps``); the schema layer in :mod:`synoptiq.reader.gold` maps those
onto the canonical keys used here.
"""

from __future__ import annotations

# ── Part-of-speech labels (MACULA/lowfat ``sp`` abbreviations → friendly nouns) ──

_POS_LABELS: dict[str, str] = {
    "subs": "noun",
    "noun": "noun",
    "verb": "verb",
    "art": "article",
    "adj": "adjective",
    "adv": "adverb",
    "prep": "preposition",
    "conj": "conjunction",
    "pron": "pronoun",
    "ptcl": "particle",
    "prt": "particle",
    "intj": "interjection",
    "num": "numeral",
    "det": "determiner",
    "advb": "adverb",
    "x": "other",
}

# Person values differ between editions (``p1`` vs ``first`` vs ``1``); normalise all.
_PERSON_LABELS: dict[str, str] = {
    "p1": "1st person",
    "p2": "2nd person",
    "p3": "3rd person",
    "first": "1st person",
    "second": "2nd person",
    "third": "3rd person",
    "1": "1st person",
    "2": "2nd person",
    "3": "3rd person",
}

# Order in which present features are rendered (verbal features first, then nominal).
CANONICAL_FEATURES: tuple[str, ...] = (
    "person",
    "tense",
    "voice",
    "mood",
    "case",
    "gender",
    "number",
    "degree",
)


def tidy_pos(sp: str) -> str:
    """Map a gold part-of-speech code to a friendly label (unknown codes pass through).

    Example:
        >>> tidy_pos("subs")
        'noun'
        >>> tidy_pos("verb")
        'verb'
    """
    return _POS_LABELS.get(sp.lower(), sp) if sp else sp


def tidy_value(feature: str, value: str) -> str:
    """Normalise a single morphology value for display (person codes → ordinals)."""
    if feature == "person":
        return _PERSON_LABELS.get(value.lower(), value)
    return value


def describe_morphology(features: dict[str, str]) -> str:
    """Assemble present morphology features into a ``·``-separated readable string.

    Args:
        features: Canonical feature → value (missing/empty features are skipped).

    Returns:
        A string like ``"3rd person · imperfect · active · indicative · singular"``
        or ``"genitive · masculine · singular"``; empty when nothing is set.

    Example:
        >>> describe_morphology({"case": "nominative", "gender": "feminine",
        ...                      "number": "singular"})
        'nominative · feminine · singular'
    """
    parts = [
        tidy_value(feat, features[feat])
        for feat in CANONICAL_FEATURES
        if features.get(feat)
    ]
    return " · ".join(parts)
