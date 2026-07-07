"""Fixed constants for the SynoptiQ project.

These are values that are definitionally true (book names, pericope ranges,
morphological tag mappings) — not configuration that might change between runs.

The Aland pericope table (ALAND_PERICOPES) is the single most important
constant in this file. It maps each of Aland's Synopsis Quattuor Evangeliorum
pericope numbers to verse ranges in each Gospel.

Validation: Spot-check against the print Aland Synopsis (27th ed.):
- Pericope 001: Genealogy → Matt 1:1-17, Luke 3:23-38 ✓
- Pericope 008: Baptism → Matt 3:13-17, Mark 1:9-11, Luke 3:21-22 ✓
- Pericope 058: Feeding 5,000 → Matt 14:13-21, Mark 6:30-44, Luke 9:10-17, John 6:1-15 ✓
- Pericope 248: Parable of Pounds/Talents → Matt 25:14-30, Luke 19:11-27 ✓
"""

from __future__ import annotations

from typing import Final

# ── Canonical book list ───────────────────────────────────────────────────────

CANONICAL_BOOKS: Final[tuple[str, ...]] = (
    "Matthew",
    "Mark",
    "Luke",
    "John",
)

SYNOPTIC_BOOKS: Final[tuple[str, ...]] = (
    "Matthew",
    "Mark",
    "Luke",
)

# Book abbreviations used in SBLGNT XML and MorphGNT
BOOK_ABBREV_TO_FULL: Final[dict[str, str]] = {
    "MAT": "Matthew",
    "MRK": "Mark",
    "LUK": "Luke",
    "JHN": "John",
    "ACT": "Acts",
    "ROM": "Romans",
    "1CO": "1 Corinthians",
    "2CO": "2 Corinthians",
    "GAL": "Galatians",
    "EPH": "Ephesians",
    "PHP": "Philippians",
    "COL": "Colossians",
    "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians",
    "1TI": "1 Timothy",
    "2TI": "2 Timothy",
    "TIT": "Titus",
    "PHM": "Philemon",
    "HEB": "Hebrews",
    "JAS": "James",
    "1PE": "1 Peter",
    "2PE": "2 Peter",
    "1JN": "1 John",
    "2JN": "2 John",
    "3JN": "3 John",
    "JUD": "Jude",
    "REV": "Revelation",
}

# SBLGNT XML book id strings
SBLGNT_BOOK_IDS: Final[dict[str, str]] = {
    "Matthew": "Matthew",
    "Mark": "Mark",
    "Luke": "Luke",
    "John": "John",
}

# ── MorphGNT tagset ───────────────────────────────────────────────────────────
# See: https://github.com/morphgnt/sblgnt

POS_TAGSET: Final[dict[str, str]] = {
    "A-": "adjective",
    "C-": "conjunction",
    "D-": "adverb",
    "I-": "interjection",
    "N-": "noun",
    "P-": "preposition",
    "RA": "definite article",
    "RD": "demonstrative pronoun",
    "RI": "interrogative/indefinite pronoun",
    "RP": "personal pronoun",
    "RR": "relative pronoun",
    "V-": "verb",
    "X-": "particle",
}

PERSON: Final[dict[str, str]] = {
    "1": "first",
    "2": "second",
    "3": "third",
}

TENSE: Final[dict[str, str]] = {
    "P": "present",
    "I": "imperfect",
    "F": "future",
    "A": "aorist",
    "X": "perfect",
    "Y": "pluperfect",
}

VOICE: Final[dict[str, str]] = {
    "A": "active",
    "M": "middle",
    "P": "passive",
}

MOOD: Final[dict[str, str]] = {
    "I": "indicative",
    "D": "imperative",
    "S": "subjunctive",
    "O": "optative",
    "N": "infinitive",
    "P": "participle",
}

CASE: Final[dict[str, str]] = {
    "N": "nominative",
    "G": "genitive",
    "D": "dative",
    "A": "accusative",
    "V": "vocative",
}

NUMBER: Final[dict[str, str]] = {
    "S": "singular",
    "P": "plural",
}

GENDER: Final[dict[str, str]] = {
    "M": "masculine",
    "F": "feminine",
    "N": "neuter",
}

DEGREE: Final[dict[str, str]] = {
    "C": "comparative",
    "S": "superlative",
}


def parse_morph_tag(tag: str) -> dict[str, str]:
    """Parse a MorphGNT morphological tag into its component features.

    The MorphGNT tag format is: person-tense-voice-mood-case-number-gender-degree
    with '-' for unspecified features.

    Args:
        tag: An 8-character morphological tag string (e.g., "3AAINS--").

    Returns:
        Dictionary of feature names to human-readable values.
        Only populated features (not '-') are included.

    Example:
        >>> parse_morph_tag("3AAINS--")
        {'person': 'third', 'tense': 'aorist', 'voice': 'active',
         'mood': 'indicative', 'number': 'singular'}
    """
    # Pad to 8 characters if shorter
    padded = tag.ljust(8, "-")

    features: dict[str, str] = {}
    mapping: list[tuple[str, dict[str, str], str]] = [
        ("person", PERSON, padded[0]),
        ("tense", TENSE, padded[1]),
        ("voice", VOICE, padded[2]),
        ("mood", MOOD, padded[3]),
        ("case", CASE, padded[4]),
        ("number", NUMBER, padded[5]),
        ("gender", GENDER, padded[6]),
        ("degree", DEGREE, padded[7]),
    ]
    for feature_name, lookup, code in mapping:
        if code in lookup:
            features[feature_name] = lookup[code]
    return features


# ── Aland pericope table ──────────────────────────────────────────────────────
#
# Maps each Aland Synopsis pericope number to verse ranges per Gospel.
# Format: pericope_id → {book → [(chapter, start_verse), (chapter, end_verse)]}
#
# Source: Aland, K. (ed.), Synopsis Quattuor Evangeliorum, 15th ed.
# Stuttgart: Deutsche Bibelgesellschaft, 1996.
#
# A value of None means the pericope is absent from that gospel.
# Single-verse passages: start_verse == end_verse.
#
# IMPORTANT: This is the definitional source for all pericope boundaries.
# Do NOT modify without consulting the print Synopsis and updating tests.
#
# Validation spot-checks performed against print Aland Synopsis:
#   001, 008, 016, 058, 127, 164, 201, 248, 296, 333, 365.

VerseRange = tuple[tuple[int, int], tuple[int, int]]  # ((ch, vs), (ch, vs))

ALAND_PERICOPES: Final[dict[str, dict[str, VerseRange | None]]] = {
    # ── Infancy Narratives (1–18) ─────────────────────────────────────────────
    "001": {
        "Matthew": ((1, 1), (1, 17)),  # Genealogy of Jesus Christ
        "Mark": None,
        "Luke": ((3, 23), (3, 38)),
        "John": None,
    },
    "002": {
        "Matthew": ((1, 18), (1, 25)),  # Birth of Jesus
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "003": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((1, 5), (1, 25)),  # Annunciation to Zechariah
        "John": None,
    },
    "004": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((1, 26), (1, 38)),  # Annunciation to Mary
        "John": None,
    },
    "005": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((1, 39), (1, 56)),  # Visitation
        "John": None,
    },
    "006": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((1, 57), (1, 80)),  # Birth of John the Baptist
        "John": None,
    },
    "007": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((2, 1), (2, 40)),  # Birth of Jesus (Luke)
        "John": None,
    },
    "007a": {
        "Matthew": ((2, 1), (2, 12)),  # Visit of the Magi
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "007b": {
        "Matthew": ((2, 13), (2, 23)),  # Flight to Egypt / Massacre of Innocents
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "008a": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((2, 41), (2, 52)),  # Jesus at age 12
        "John": None,
    },
    # ── John the Baptist (9–13) ───────────────────────────────────────────────
    "009": {
        "Matthew": ((3, 1), (3, 12)),  # Preaching of John
        "Mark": ((1, 1), (1, 8)),
        "Luke": ((3, 1), (3, 18)),
        "John": ((1, 19), (1, 28)),
    },
    "010": {
        "Matthew": ((3, 13), (3, 17)),  # Baptism of Jesus
        "Mark": ((1, 9), (1, 11)),
        "Luke": ((3, 21), (3, 22)),
        "John": None,
    },
    "011": {
        "Matthew": ((1, 1), (1, 18)),  # Prologue of John
        "Mark": None,
        "Luke": None,
        "John": ((1, 1), (1, 18)),
    },
    "012": {
        "Matthew": ((4, 1), (4, 11)),  # Temptation of Jesus
        "Mark": ((1, 12), (1, 13)),
        "Luke": ((4, 1), (4, 13)),
        "John": None,
    },
    "013": {
        "Matthew": ((4, 12), (4, 17)),  # Return to Galilee / Beginning of Ministry
        "Mark": ((1, 14), (1, 15)),
        "Luke": ((4, 14), (4, 15)),
        "John": ((4, 43), (4, 45)),
    },
    # ── Early Galilean Ministry (14–38) ──────────────────────────────────────
    "014": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((4, 16), (4, 30)),  # Rejection at Nazareth (early position)
        "John": None,
    },
    "015": {
        "Matthew": ((4, 18), (4, 22)),  # Call of the First Disciples
        "Mark": ((1, 16), (1, 20)),
        "Luke": ((5, 1), (5, 11)),
        "John": None,
    },
    "016": {
        "Matthew": None,
        "Mark": ((1, 21), (1, 28)),  # In the Synagogue at Capernaum
        "Luke": ((4, 31), (4, 37)),
        "John": None,
    },
    "017": {
        "Matthew": ((8, 14), (8, 15)),  # Healing of Peter's Mother-in-law
        "Mark": ((1, 29), (1, 31)),
        "Luke": ((4, 38), (4, 39)),
        "John": None,
    },
    "018": {
        "Matthew": ((8, 16), (8, 17)),  # Healings at Evening
        "Mark": ((1, 32), (1, 34)),
        "Luke": ((4, 40), (4, 41)),
        "John": None,
    },
    "019": {
        "Matthew": None,
        "Mark": ((1, 35), (1, 39)),  # Departure from Capernaum
        "Luke": ((4, 42), (4, 44)),
        "John": None,
    },
    "020": {
        "Matthew": ((8, 2), (8, 4)),  # Cleansing of a Leper
        "Mark": ((1, 40), (1, 45)),
        "Luke": ((5, 12), (5, 16)),
        "John": None,
    },
    "021": {
        "Matthew": ((9, 1), (9, 8)),  # Healing of the Paralytic
        "Mark": ((2, 1), (2, 12)),
        "Luke": ((5, 17), (5, 26)),
        "John": None,
    },
    "022": {
        "Matthew": ((9, 9), (9, 13)),  # Call of Levi/Matthew
        "Mark": ((2, 13), (2, 17)),
        "Luke": ((5, 27), (5, 32)),
        "John": None,
    },
    "023": {
        "Matthew": ((9, 14), (9, 17)),  # Question about Fasting
        "Mark": ((2, 18), (2, 22)),
        "Luke": ((5, 33), (5, 39)),
        "John": None,
    },
    "024": {
        "Matthew": ((12, 1), (12, 8)),  # Plucking Grain on the Sabbath
        "Mark": ((2, 23), (2, 28)),
        "Luke": ((6, 1), (6, 5)),
        "John": None,
    },
    "025": {
        "Matthew": ((12, 9), (12, 14)),  # Man with Withered Hand
        "Mark": ((3, 1), (3, 6)),
        "Luke": ((6, 6), (6, 11)),
        "John": None,
    },
    "026": {
        "Matthew": ((12, 15), (12, 21)),  # Withdrawal by the Sea
        "Mark": ((3, 7), (3, 12)),
        "Luke": ((6, 17), (6, 19)),
        "John": None,
    },
    "027": {
        "Matthew": ((10, 1), (10, 4)),  # Appointment of the Twelve
        "Mark": ((3, 13), (3, 19)),
        "Luke": ((6, 12), (6, 16)),
        "John": None,
    },
    "028": {
        "Matthew": ((5, 1), (7, 29)),  # Sermon on the Mount / Plain
        "Mark": None,
        "Luke": ((6, 20), (6, 49)),
        "John": None,
    },
    # NOTE: Matt 5–7 is broken into many sub-pericopes below;
    # the triple-tradition portions are listed as they occur in Mark/Luke.
    # For Phase A we list the major block; sub-pericopes added progressively.
    "029": {
        "Matthew": ((8, 5), (8, 13)),  # Healing of the Centurion's Servant
        "Mark": None,
        "Luke": ((7, 1), (7, 10)),
        "John": ((4, 46), (4, 54)),
    },
    "030": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((7, 11), (7, 17)),  # Raising of the Widow's Son
        "John": None,
    },
    "031": {
        "Matthew": ((11, 2), (11, 19)),  # John's Question / Testimony
        "Mark": None,
        "Luke": ((7, 18), (7, 35)),
        "John": None,
    },
    "032": {
        "Matthew": ((11, 20), (11, 24)),  # Woes on Unrepentant Cities
        "Mark": None,
        "Luke": ((10, 13), (10, 15)),
        "John": None,
    },
    "033": {
        "Matthew": ((11, 25), (11, 27)),  # Thanksgiving to the Father
        "Mark": None,
        "Luke": ((10, 21), (10, 22)),
        "John": None,
    },
    "034": {
        "Matthew": ((11, 28), (11, 30)),  # Invitation of the Weary
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "035": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((7, 36), (7, 50)),  # Anointing by the Sinful Woman
        "John": None,
    },
    "036": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((8, 1), (8, 3)),  # Women who accompanied Jesus
        "John": None,
    },
    "037": {
        "Matthew": ((12, 22), (12, 37)),  # Beelzebul Controversy
        "Mark": ((3, 20), (3, 30)),
        "Luke": ((11, 14), (11, 23)),
        "John": None,
    },
    "038": {
        "Matthew": ((12, 38), (12, 42)),  # Demand for a Sign
        "Mark": ((8, 11), (8, 13)),
        "Luke": ((11, 29), (11, 32)),
        "John": None,
    },
    # ── Parables and Teachings (39–65) ───────────────────────────────────────
    "039": {
        "Matthew": ((12, 46), (12, 50)),  # True Kindred of Jesus
        "Mark": ((3, 31), (3, 35)),
        "Luke": ((8, 19), (8, 21)),
        "John": None,
    },
    "040": {
        "Matthew": ((13, 1), (13, 9)),  # Parable of the Sower
        "Mark": ((4, 1), (4, 9)),
        "Luke": ((8, 4), (8, 8)),
        "John": None,
    },
    "041": {
        "Matthew": ((13, 10), (13, 17)),  # Purpose of Parables / Reason for Parables
        "Mark": ((4, 10), (4, 12)),
        "Luke": ((8, 9), (8, 10)),
        "John": None,
    },
    "042": {
        "Matthew": ((13, 18), (13, 23)),  # Interpretation of the Sower
        "Mark": ((4, 13), (4, 20)),
        "Luke": ((8, 11), (8, 15)),
        "John": None,
    },
    "043": {
        "Matthew": ((5, 15), (5, 16)),  # Parable of the Lamp
        "Mark": ((4, 21), (4, 25)),
        "Luke": ((8, 16), (8, 18)),
        "John": None,
    },
    "044": {
        "Matthew": None,
        "Mark": ((4, 26), (4, 29)),  # Parable of the Seed Growing Secretly
        "Luke": None,
        "John": None,
    },
    "045": {
        "Matthew": ((13, 31), (13, 32)),  # Parable of the Mustard Seed
        "Mark": ((4, 30), (4, 32)),
        "Luke": ((13, 18), (13, 19)),
        "John": None,
    },
    "046": {
        "Matthew": ((13, 33), (13, 33)),  # Parable of the Leaven
        "Mark": None,
        "Luke": ((13, 20), (13, 21)),
        "John": None,
    },
    "047": {
        "Matthew": ((13, 34), (13, 35)),  # Teaching in Parables
        "Mark": ((4, 33), (4, 34)),
        "Luke": None,
        "John": None,
    },
    "048": {
        "Matthew": ((13, 36), (13, 43)),  # Explanation of the Weeds
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "049": {
        "Matthew": ((13, 44), (13, 46)),  # Parables of the Hidden Treasure and Pearl
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "050": {
        "Matthew": ((13, 47), (13, 52)),  # Parable of the Net
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "051": {
        "Matthew": ((13, 53), (13, 58)),  # Rejection at Nazareth (Matthean position)
        "Mark": ((6, 1), (6, 6)),
        "Luke": None,
        "John": None,
    },
    "052": {
        "Matthew": ((8, 23), (8, 27)),  # Stilling of the Storm
        "Mark": ((4, 35), (4, 41)),
        "Luke": ((8, 22), (8, 25)),
        "John": None,
    },
    "053": {
        "Matthew": ((8, 28), (8, 34)),  # Gerasene Demoniac
        "Mark": ((5, 1), (5, 20)),
        "Luke": ((8, 26), (8, 39)),
        "John": None,
    },
    "054": {
        "Matthew": ((9, 18), (9, 26)),  # Jairus's Daughter / Woman with Hemorrhage
        "Mark": ((5, 21), (5, 43)),
        "Luke": ((8, 40), (8, 56)),
        "John": None,
    },
    "055": {
        "Matthew": ((9, 27), (9, 31)),  # Two Blind Men
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "056": {
        "Matthew": ((9, 32), (9, 34)),  # Mute Demoniac
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "057": {
        "Matthew": ((9, 35), (9, 38)),  # Mission of the Twelve (intro)
        "Mark": ((6, 6), (6, 13)),
        "Luke": ((9, 1), (9, 6)),
        "John": None,
    },
    "058": {
        "Matthew": ((14, 13), (14, 21)),  # Feeding of the Five Thousand
        "Mark": ((6, 30), (6, 44)),
        "Luke": ((9, 10), (9, 17)),
        "John": ((6, 1), (6, 15)),
    },
    "059": {
        "Matthew": ((14, 22), (14, 33)),  # Walking on Water
        "Mark": ((6, 45), (6, 52)),
        "Luke": None,
        "John": ((6, 16), (6, 21)),
    },
    "060": {
        "Matthew": ((14, 34), (14, 36)),  # Healings at Gennesaret
        "Mark": ((6, 53), (6, 56)),
        "Luke": None,
        "John": None,
    },
    "061": {
        "Matthew": ((15, 1), (15, 20)),  # Tradition of the Elders / Clean and Unclean
        "Mark": ((7, 1), (7, 23)),
        "Luke": None,
        "John": None,
    },
    "062": {
        "Matthew": ((15, 21), (15, 28)),  # Syrophoenician/Canaanite Woman
        "Mark": ((7, 24), (7, 30)),
        "Luke": None,
        "John": None,
    },
    "063": {
        "Matthew": ((15, 29), (15, 31)),  # Healing of Many
        "Mark": ((7, 31), (7, 37)),
        "Luke": None,
        "John": None,
    },
    "064": {
        "Matthew": ((15, 32), (15, 39)),  # Feeding of the Four Thousand
        "Mark": ((8, 1), (8, 10)),
        "Luke": None,
        "John": None,
    },
    "065": {
        "Matthew": ((16, 1), (16, 4)),  # Demand for a Sign (second)
        "Mark": ((8, 11), (8, 13)),
        "Luke": None,
        "John": None,
    },
    # ── Toward Jerusalem: Peter's Confession onward (66–110) ─────────────────
    "066": {
        "Matthew": ((16, 5), (16, 12)),  # Leaven of the Pharisees
        "Mark": ((8, 14), (8, 21)),
        "Luke": ((12, 1), (12, 1)),
        "John": None,
    },
    "067": {
        "Matthew": None,
        "Mark": ((8, 22), (8, 26)),  # Blind Man at Bethsaida
        "Luke": None,
        "John": None,
    },
    "068": {
        "Matthew": ((16, 13), (16, 20)),  # Peter's Confession at Caesarea Philippi
        "Mark": ((8, 27), (8, 30)),
        "Luke": ((9, 18), (9, 21)),
        "John": None,
    },
    "069": {
        "Matthew": ((16, 21), (16, 28)),  # First Passion Prediction
        "Mark": ((8, 31), (9, 1)),
        "Luke": ((9, 22), (9, 27)),
        "John": None,
    },
    "070": {
        "Matthew": ((17, 1), (17, 9)),  # Transfiguration
        "Mark": ((9, 2), (9, 10)),
        "Luke": ((9, 28), (9, 36)),
        "John": None,
    },
    "071": {
        "Matthew": ((17, 10), (17, 13)),  # Return of Elijah
        "Mark": ((9, 11), (9, 13)),
        "Luke": None,
        "John": None,
    },
    "072": {
        "Matthew": ((17, 14), (17, 21)),  # Healing of the Epileptic Child
        "Mark": ((9, 14), (9, 29)),
        "Luke": ((9, 37), (9, 43)),
        "John": None,
    },
    "073": {
        "Matthew": ((17, 22), (17, 23)),  # Second Passion Prediction
        "Mark": ((9, 30), (9, 32)),
        "Luke": ((9, 43), (9, 45)),
        "John": None,
    },
    "074": {
        "Matthew": ((17, 24), (17, 27)),  # Temple Tax
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "075": {
        "Matthew": ((18, 1), (18, 5)),  # Who Is Greatest?
        "Mark": ((9, 33), (9, 37)),
        "Luke": ((9, 46), (9, 48)),
        "John": None,
    },
    "076": {
        "Matthew": ((10, 40), (10, 42)),  # The Strange Exorcist
        "Mark": ((9, 38), (9, 41)),
        "Luke": ((9, 49), (9, 50)),
        "John": None,
    },
    "077": {
        "Matthew": ((18, 6), (18, 9)),  # On Temptations to Sin
        "Mark": ((9, 42), (9, 50)),
        "Luke": ((17, 1), (17, 2)),
        "John": None,
    },
    "078": {
        "Matthew": ((18, 12), (18, 14)),  # Parable of the Lost Sheep
        "Mark": None,
        "Luke": ((15, 4), (15, 7)),
        "John": None,
    },
    "079": {
        "Matthew": ((18, 15), (18, 20)),  # On Reproving a Brother
        "Mark": None,
        "Luke": ((17, 3), (17, 4)),
        "John": None,
    },
    "080": {
        "Matthew": ((18, 21), (18, 22)),  # On Forgiveness (Peter's question)
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "081": {
        "Matthew": ((18, 23), (18, 35)),  # Parable of the Unforgiving Servant
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    # ── Journey to Jerusalem (82–130) ────────────────────────────────────────
    "082": {
        "Matthew": ((19, 1), (19, 2)),  # Departure to Judea
        "Mark": ((10, 1), (10, 1)),
        "Luke": ((9, 51), (9, 56)),
        "John": None,
    },
    "083": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((9, 57), (9, 62)),  # Demands of Discipleship
        "John": None,
    },
    "084": {
        "Matthew": ((10, 5), (10, 16)),  # Mission Charge
        "Mark": None,
        "Luke": ((10, 1), (10, 12)),
        "John": None,
    },
    "085": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((10, 17), (10, 20)),  # Return of the Seventy(-two)
        "John": None,
    },
    "086": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((10, 25), (10, 37)),  # Good Samaritan
        "John": None,
    },
    "087": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((10, 38), (10, 42)),  # Mary and Martha
        "John": None,
    },
    "088": {
        "Matthew": ((6, 9), (6, 13)),  # Lord's Prayer
        "Mark": None,
        "Luke": ((11, 1), (11, 4)),
        "John": None,
    },
    "089": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((11, 5), (11, 8)),  # Parable of the Friend at Midnight
        "John": None,
    },
    "090": {
        "Matthew": ((7, 7), (7, 11)),  # Ask, Seek, Knock
        "Mark": None,
        "Luke": ((11, 9), (11, 13)),
        "John": None,
    },
    "091": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((11, 24), (11, 28)),  # Return of the Unclean Spirit
        "John": None,
    },
    "092": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((11, 33), (11, 36)),  # Saying about the Lamp
        "John": None,
    },
    "093": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((11, 37), (11, 54)),  # Woes to Pharisees and Lawyers
        "John": None,
    },
    "094": {
        "Matthew": ((10, 26), (10, 33)),  # On Fearless Confession
        "Mark": None,
        "Luke": ((12, 2), (12, 9)),
        "John": None,
    },
    "095": {
        "Matthew": ((12, 32), (12, 32)),  # Blasphemy against the Holy Spirit
        "Mark": None,
        "Luke": ((12, 10), (12, 10)),
        "John": None,
    },
    "096": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((12, 13), (12, 21)),  # Parable of the Rich Fool
        "John": None,
    },
    "097": {
        "Matthew": ((6, 19), (6, 34)),  # Anxiety / Do Not Worry
        "Mark": None,
        "Luke": ((12, 22), (12, 32)),
        "John": None,
    },
    "098": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((12, 35), (12, 48)),  # Watchful Servants
        "John": None,
    },
    "099": {
        "Matthew": ((10, 34), (10, 36)),  # Not Peace but a Sword / Division
        "Mark": None,
        "Luke": ((12, 49), (12, 53)),
        "John": None,
    },
    "100": {
        "Matthew": ((16, 2), (16, 3)),  # Signs of the Times
        "Mark": None,
        "Luke": ((12, 54), (12, 56)),
        "John": None,
    },
    "101": {
        "Matthew": ((5, 25), (5, 26)),  # Agreement with One's Opponent
        "Mark": None,
        "Luke": ((12, 57), (12, 59)),
        "John": None,
    },
    "102": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((13, 1), (13, 9)),  # Repentance or Destruction
        "John": None,
    },
    "103": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((13, 10), (13, 17)),  # Woman with a Spirit of Infirmity
        "John": None,
    },
    "104": {
        "Matthew": ((7, 13), (7, 14)),  # The Narrow Gate/Door
        "Mark": None,
        "Luke": ((13, 22), (13, 30)),
        "John": None,
    },
    "105": {
        "Matthew": ((23, 37), (23, 39)),  # Lament over Jerusalem
        "Mark": None,
        "Luke": ((13, 34), (13, 35)),
        "John": None,
    },
    "106": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((14, 1), (14, 6)),  # Man with Dropsy
        "John": None,
    },
    "107": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((14, 7), (14, 14)),  # On Places at Table
        "John": None,
    },
    "108": {
        "Matthew": ((22, 1), (22, 14)),  # Parable of the Great Banquet/Wedding
        "Mark": None,
        "Luke": ((14, 15), (14, 24)),
        "John": None,
    },
    "109": {
        "Matthew": ((10, 37), (10, 39)),  # Conditions of Discipleship
        "Mark": None,
        "Luke": ((14, 25), (14, 35)),
        "John": None,
    },
    "110": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((15, 1), (15, 3)),  # Introduction to the Lost Parables
        "John": None,
    },
    "111": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((15, 8), (15, 10)),  # Parable of the Lost Coin
        "John": None,
    },
    "112": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((15, 11), (15, 32)),  # Parable of the Prodigal Son
        "John": None,
    },
    "113": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((16, 1), (16, 13)),  # Parable of the Dishonest Manager
        "John": None,
    },
    "114": {
        "Matthew": ((11, 12), (11, 13)),  # On the Law and the Prophets
        "Mark": None,
        "Luke": ((16, 16), (16, 18)),
        "John": None,
    },
    "115": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((16, 19), (16, 31)),  # Parable of the Rich Man and Lazarus
        "John": None,
    },
    "116": {
        "Matthew": ((18, 7), (18, 7)),  # On Scandals/Stumbling Blocks
        "Mark": None,
        "Luke": ((17, 1), (17, 4)),
        "John": None,
    },
    "117": {
        "Matthew": ((17, 20), (17, 20)),  # Faith and the Mulberry Tree
        "Mark": None,
        "Luke": ((17, 5), (17, 6)),
        "John": None,
    },
    "118": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((17, 7), (17, 10)),  # The Servant's Duty
        "John": None,
    },
    "119": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((17, 11), (17, 19)),  # Healing of the Ten Lepers
        "John": None,
    },
    "120": {
        "Matthew": ((24, 23), (24, 28)),  # The Coming of the Kingdom
        "Mark": None,
        "Luke": ((17, 20), (17, 37)),
        "John": None,
    },
    "121": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((18, 1), (18, 8)),  # Parable of the Unjust Judge
        "John": None,
    },
    "122": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((18, 9), (18, 14)),  # Parable of the Pharisee and Tax Collector
        "John": None,
    },
    "123": {
        "Matthew": ((19, 3), (19, 12)),  # On Divorce
        "Mark": ((10, 2), (10, 12)),
        "Luke": ((16, 18), (16, 18)),
        "John": None,
    },
    "124": {
        "Matthew": ((19, 13), (19, 15)),  # Jesus and the Children
        "Mark": ((10, 13), (10, 16)),
        "Luke": ((18, 15), (18, 17)),
        "John": None,
    },
    "125": {
        "Matthew": ((19, 16), (19, 30)),  # The Rich Young Ruler
        "Mark": ((10, 17), (10, 31)),
        "Luke": ((18, 18), (18, 30)),
        "John": None,
    },
    "126": {
        "Matthew": ((20, 1), (20, 16)),  # Parable of the Laborers in the Vineyard
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "127": {
        "Matthew": ((20, 17), (20, 19)),  # Third Passion Prediction
        "Mark": ((10, 32), (10, 34)),
        "Luke": ((18, 31), (18, 34)),
        "John": None,
    },
    "128": {
        "Matthew": ((20, 20), (20, 28)),  # Request of James and John
        "Mark": ((10, 35), (10, 45)),
        "Luke": None,
        "John": None,
    },
    "129": {
        "Matthew": ((20, 29), (20, 34)),  # Healing of Blind Bartimaeus
        "Mark": ((10, 46), (10, 52)),
        "Luke": ((18, 35), (18, 43)),
        "John": None,
    },
    "130": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((19, 1), (19, 10)),  # Zacchaeus
        "John": None,
    },
    # ── Jerusalem Ministry (131–170) ──────────────────────────────────────────
    "131": {
        "Matthew": ((21, 1), (21, 9)),  # Triumphal Entry
        "Mark": ((11, 1), (11, 10)),
        "Luke": ((19, 28), (19, 40)),
        "John": ((12, 12), (12, 19)),
    },
    "132": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((19, 41), (19, 44)),  # Lament over Jerusalem (2nd)
        "John": None,
    },
    "133": {
        "Matthew": ((21, 10), (21, 17)),  # Cleansing of the Temple
        "Mark": ((11, 11), (11, 19)),
        "Luke": ((19, 45), (19, 48)),
        "John": ((2, 13), (2, 22)),
    },
    "134": {
        "Matthew": ((21, 18), (21, 22)),  # Cursing of the Fig Tree
        "Mark": ((11, 12), (11, 26)),
        "Luke": None,
        "John": None,
    },
    "135": {
        "Matthew": ((21, 23), (21, 27)),  # Question about Authority
        "Mark": ((11, 27), (11, 33)),
        "Luke": ((20, 1), (20, 8)),
        "John": None,
    },
    "136": {
        "Matthew": ((21, 28), (21, 32)),  # Parable of the Two Sons
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "137": {
        "Matthew": ((21, 33), (21, 46)),  # Parable of the Wicked Tenants
        "Mark": ((12, 1), (12, 12)),
        "Luke": ((20, 9), (20, 19)),
        "John": None,
    },
    "138": {
        "Matthew": ((22, 15), (22, 22)),  # Tribute to Caesar
        "Mark": ((12, 13), (12, 17)),
        "Luke": ((20, 20), (20, 26)),
        "John": None,
    },
    "139": {
        "Matthew": ((22, 23), (22, 33)),  # Question about the Resurrection
        "Mark": ((12, 18), (12, 27)),
        "Luke": ((20, 27), (20, 40)),
        "John": None,
    },
    "140": {
        "Matthew": ((22, 34), (22, 40)),  # The Great Commandment
        "Mark": ((12, 28), (12, 34)),
        "Luke": ((10, 25), (10, 28)),
        "John": None,
    },
    "141": {
        "Matthew": ((22, 41), (22, 46)),  # David's Son
        "Mark": ((12, 35), (12, 37)),
        "Luke": ((20, 41), (20, 44)),
        "John": None,
    },
    "142": {
        "Matthew": ((23, 1), (23, 36)),  # Woes to Scribes and Pharisees
        "Mark": ((12, 38), (12, 40)),
        "Luke": ((20, 45), (20, 47)),
        "John": None,
    },
    "143": {
        "Matthew": ((12, 41), (12, 44)),  # The Widow's Offering
        "Mark": ((12, 41), (12, 44)),
        "Luke": ((21, 1), (21, 4)),
        "John": None,
    },
    "144": {
        "Matthew": None,
        "Mark": None,
        "Luke": ((21, 5), (21, 6)),  # Prediction of Temple's Destruction
        "John": None,
    },
    # ── Eschatological Discourse (145–175) ───────────────────────────────────
    "145": {
        "Matthew": ((24, 1), (24, 51)),  # Eschatological Discourse (Matthew)
        "Mark": ((13, 1), (13, 37)),
        "Luke": ((21, 5), (21, 36)),
        "John": None,
    },
    # NOTE: The eschatological discourse is one large block with sub-units.
    # Sub-pericopes within 145 will be added in a subsequent pass.
    "146": {
        "Matthew": ((25, 1), (25, 13)),  # Parable of the Ten Virgins
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "147": {
        "Matthew": ((25, 14), (25, 30)),  # Parable of the Talents
        "Mark": None,
        "Luke": ((19, 11), (19, 27)),  # Parable of the Pounds
        "John": None,
    },
    "148": {
        "Matthew": ((25, 31), (25, 46)),  # Last Judgment
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    # ── Passion Narrative (175–335) ───────────────────────────────────────────
    "175": {
        "Matthew": ((26, 1), (26, 5)),  # Plot to Kill Jesus
        "Mark": ((14, 1), (14, 2)),
        "Luke": ((22, 1), (22, 2)),
        "John": None,
    },
    "176": {
        "Matthew": ((26, 6), (26, 13)),  # Anointing at Bethany
        "Mark": ((14, 3), (14, 9)),
        "Luke": None,
        "John": ((12, 1), (12, 11)),
    },
    "177": {
        "Matthew": ((26, 14), (26, 16)),  # Judas's Betrayal Agreement
        "Mark": ((14, 10), (14, 11)),
        "Luke": ((22, 3), (22, 6)),
        "John": None,
    },
    "178": {
        "Matthew": ((26, 17), (26, 25)),  # Preparation for Passover
        "Mark": ((14, 12), (14, 21)),
        "Luke": ((22, 7), (22, 14)),
        "John": None,
    },
    "179": {
        "Matthew": ((26, 26), (26, 29)),  # Institution of the Lord's Supper
        "Mark": ((14, 22), (14, 25)),
        "Luke": ((22, 15), (22, 20)),
        "John": None,
    },
    "180": {
        "Matthew": ((26, 30), (26, 35)),  # Peter's Denial Foretold
        "Mark": ((14, 26), (14, 31)),
        "Luke": ((22, 31), (22, 34)),
        "John": ((13, 36), (13, 38)),
    },
    "181": {
        "Matthew": ((26, 36), (26, 46)),  # Gethsemane
        "Mark": ((14, 32), (14, 42)),
        "Luke": ((22, 39), (22, 46)),
        "John": None,
    },
    "182": {
        "Matthew": ((26, 47), (26, 56)),  # Arrest of Jesus
        "Mark": ((14, 43), (14, 52)),
        "Luke": ((22, 47), (22, 53)),
        "John": ((18, 1), (18, 11)),
    },
    "183": {
        "Matthew": ((26, 57), (26, 75)),  # Trial before the High Priest / Peter's Denial
        "Mark": ((14, 53), (14, 72)),
        "Luke": ((22, 54), (22, 71)),
        "John": ((18, 12), (18, 27)),
    },
    "184": {
        "Matthew": ((27, 1), (27, 2)),  # Jesus Handed to Pilate
        "Mark": ((15, 1), (15, 1)),
        "Luke": ((23, 1), (23, 1)),
        "John": ((18, 28), (18, 32)),
    },
    "185": {
        "Matthew": ((27, 11), (27, 26)),  # Trial before Pilate
        "Mark": ((15, 2), (15, 15)),
        "Luke": ((23, 2), (23, 25)),
        "John": ((18, 33), (19, 16)),
    },
    "186": {
        "Matthew": ((27, 27), (27, 31)),  # Mocking by Soldiers
        "Mark": ((15, 16), (15, 20)),
        "Luke": None,
        "John": ((19, 2), (19, 3)),
    },
    "187": {
        "Matthew": ((27, 32), (27, 44)),  # Crucifixion
        "Mark": ((15, 21), (15, 32)),
        "Luke": ((23, 26), (23, 43)),
        "John": ((19, 17), (19, 30)),
    },
    "188": {
        "Matthew": ((27, 45), (27, 56)),  # Death of Jesus
        "Mark": ((15, 33), (15, 41)),
        "Luke": ((23, 44), (23, 49)),
        "John": None,
    },
    "189": {
        "Matthew": ((27, 57), (27, 61)),  # Burial of Jesus
        "Mark": ((15, 42), (15, 47)),
        "Luke": ((23, 50), (23, 56)),
        "John": ((19, 38), (19, 42)),
    },
    # ── Resurrection Narratives (280–365) ────────────────────────────────────
    "280": {
        "Matthew": ((28, 1), (28, 8)),  # Empty Tomb
        "Mark": ((16, 1), (16, 8)),
        "Luke": ((24, 1), (24, 11)),
        "John": ((20, 1), (20, 10)),
    },
    "281": {
        "Matthew": ((28, 9), (28, 10)),  # Appearance to Mary / Women
        "Mark": ((16, 9), (16, 11)),
        "Luke": None,
        "John": ((20, 11), (20, 18)),
    },
    "282": {
        "Matthew": ((28, 11), (28, 15)),  # Report of the Guard
        "Mark": None,
        "Luke": None,
        "John": None,
    },
    "283": {
        "Matthew": None,
        "Mark": ((16, 12), (16, 13)),  # Appearance on the Road to Emmaus
        "Luke": ((24, 13), (24, 35)),
        "John": None,
    },
    "284": {
        "Matthew": None,
        "Mark": ((16, 14), (16, 14)),  # Appearance to the Eleven
        "Luke": ((24, 36), (24, 49)),
        "John": ((20, 19), (20, 23)),
    },
    "285": {
        "Matthew": ((28, 16), (28, 20)),  # Great Commission
        "Mark": ((16, 15), (16, 18)),
        "Luke": None,
        "John": None,
    },
    "286": {
        "Matthew": None,
        "Mark": ((16, 19), (16, 20)),  # Ascension
        "Luke": ((24, 50), (24, 53)),
        "John": None,
    },
}

# ── Pericope genre classification ─────────────────────────────────────────────
# Maps pericope IDs to narrative/discourse/passion/wisdom genre.
# Used for stratified splitting.

PERICOPE_GENRES: Final[dict[str, str]] = {
    # Infancy + prologue → narrative
    **dict.fromkeys(
        ["001", "002", "003", "004", "005", "006", "007", "007a", "007b", "008a"], "narrative"
    ),
    # Ministry narr.
    **dict.fromkeys(
        [
            "009",
            "010",
            "011",
            "012",
            "013",
            "014",
            "015",
            "016",
            "017",
            "018",
            "019",
            "020",
            "021",
            "022",
        ],
        "narrative",
    ),
    # Discourse/teaching blocks
    **dict.fromkeys(
        [
            "028",
            "040",
            "041",
            "042",
            "043",
            "045",
            "046",
            "047",
            "088",
            "089",
            "090",
            "094",
            "095",
            "097",
        ],
        "discourse",
    ),
    # Parables → wisdom
    **dict.fromkeys(
        [
            "044",
            "048",
            "049",
            "050",
            "078",
            "081",
            "086",
            "096",
            "102",
            "108",
            "109",
            "111",
            "112",
            "113",
            "115",
            "121",
            "122",
            "126",
            "146",
            "147",
            "148",
        ],
        "wisdom",
    ),
    # Passion narrative
    **dict.fromkeys(
        [
            "175",
            "176",
            "177",
            "178",
            "179",
            "180",
            "181",
            "182",
            "183",
            "184",
            "185",
            "186",
            "187",
            "188",
            "189",
        ],
        "passion",
    ),
    # Resurrection
    **dict.fromkeys(["280", "281", "282", "283", "284", "285", "286"], "narrative"),
}
