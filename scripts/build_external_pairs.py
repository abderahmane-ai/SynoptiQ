"""Build the external known-direction evaluation set.

Produces passage pairs whose copying direction is settled by scholarly consensus
and where NO synoptic author (Matthew/Mark/Luke) appears — so a probe that works
here is measuring direction, not "is Mark on the left".

Currently covers Jude -> 2 Peter. 2 Peter's dependence on Jude is the majority
scholarly view (Bauckham, Jude–2 Peter, WBC 50, 1983). Both are Koine Greek and
already present in the on-disk SBLGNT, so this needs no external download and stays
in KoineFormer's domain. The parallel blocks below follow Bauckham's standard table.

LXX Samuel/Kings -> Chronicles is a natural second source (also consensus direction)
but requires the Swete LXX download and careful verse mapping; it is deferred.

Output: data/external/known_direction_pairs.json — human-inspectable, git-tracked.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.utils.greek import strip_punctuation  # noqa: E402

# SBLGNT editorial apparatus markers live in Supplemental Punctuation (U+2E00–2E7F).
_APPARATUS = re.compile(r"[⸀-⹿]")

# Jude (source, A) -> 2 Peter (copy, B). Blocks follow Bauckham's parallel table;
# boundaries are grouped so each side is long enough for a stable conditional NLL.
_JUDE_2PETER_BLOCKS: list[dict[str, object]] = [
    {"id": "jude_2pet_01", "jude": ("1:4",),          "2pet": ("2:1", "2:2", "2:3"),
     "topic": "false teachers deny the Master, condemnation long ago decreed"},
    {"id": "jude_2pet_02", "jude": ("1:6", "1:7"),    "2pet": ("2:4", "2:5", "2:6"),
     "topic": "angels kept for judgment; Sodom and Gomorrah as example"},
    {"id": "jude_2pet_03", "jude": ("1:8", "1:9", "1:10"), "2pet": ("2:10", "2:11", "2:12"),
     "topic": "revile the glories; irrational animals born to be caught"},
    {"id": "jude_2pet_04", "jude": ("1:11", "1:12", "1:13"), "2pet": ("2:15", "2:16", "2:17"),
     "topic": "the way of Balaam; waterless springs and mists of darkness"},
    {"id": "jude_2pet_05", "jude": ("1:16",),         "2pet": ("2:18",),
     "topic": "bombastic speech, following their own lusts"},
    {"id": "jude_2pet_06", "jude": ("1:17", "1:18"),  "2pet": ("3:1", "3:2", "3:3"),
     "topic": "remember the apostles' prediction of scoffers in the last days"},
]


def _clean(text: str) -> str:
    """Strip apparatus marks and per-word punctuation, keep accented Greek words."""
    text = _APPARATUS.sub("", text)
    words = [strip_punctuation(w) for w in text.split()]
    return " ".join(w for w in words if w)


def _load_verses(txt_path: Path, book_label: str) -> dict[str, str]:
    """Parse an SBLGNT text file into {'chapter:verse': cleaned_greek}."""
    verses: dict[str, str] = {}
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue  # book-title line
        ref, text = line.split("\t", 1)
        # ref looks like "Jude 1:4" or "2Pet 2:1"
        parts = ref.rsplit(" ", 1)
        if len(parts) != 2 or not parts[1].count(":"):
            continue
        verses[parts[1]] = _clean(text)
    if not verses:
        msg = f"no verses parsed from {txt_path} for {book_label}"
        raise ValueError(msg)
    return verses


def _join(verses: dict[str, str], refs: tuple[str, ...]) -> str:
    """Concatenate the cleaned text of the given chapter:verse references."""
    out = []
    for r in refs:
        if r not in verses:
            msg = f"missing verse {r} (have {sorted(verses)[:3]}...)"
            raise KeyError(msg)
        out.append(verses[r])
    return " ".join(out)


def main() -> None:
    """Assemble the external known-direction pairs and write the JSON."""
    sblgnt = Path("data/raw/sblgnt/data/sblgnt/text")
    jude = _load_verses(sblgnt / "Jude.txt", "Jude")
    pet2 = _load_verses(sblgnt / "2Pet.txt", "2Pet")

    pairs = []
    for block in _JUDE_2PETER_BLOCKS:
        text_a = _join(jude, block["jude"])   # type: ignore[arg-type]
        text_b = _join(pet2, block["2pet"])   # type: ignore[arg-type]
        pairs.append({
            "id": block["id"],
            "group": block["id"],           # each block is its own bootstrap unit
            "direction": "A_to_B",          # A=Jude (source) -> B=2 Peter (copy)
            "book_a": "Jude",
            "book_b": "2Peter",
            "ref_a": "Jude " + ", ".join(block["jude"]),   # type: ignore[operator]
            "ref_b": "2Pet " + ", ".join(block["2pet"]),   # type: ignore[operator]
            "topic": block["topic"],
            "text_a": text_a,
            "text_b": text_b,
            "consensus_source": (
                "Bauckham 1983 (WBC 50); majority view that 2 Peter depends on Jude"
            ),
        })

    out = {
        "description": (
            "Known-direction passage pairs for direction-probe validation. No synoptic "
            "author appears, so accuracy here reflects direction detection, not authorship."
        ),
        "license": "SBLGNT text CC-BY 4.0 (SBL / Logos Bible Software)",
        "pairs": pairs,
    }

    dest = Path("data/external/known_direction_pairs.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(pairs)} pairs -> {dest}")
    for p in pairs:
        la, lb = len(p["text_a"].split()), len(p["text_b"].split())
        print(f"  {p['id']}: {p['ref_a']} ({la}w) -> {p['ref_b']} ({lb}w) | {p['topic']}")


if __name__ == "__main__":
    main()
