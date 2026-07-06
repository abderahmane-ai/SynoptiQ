"""Build LXX Samuel-Kings -> Chronicles known-direction pairs (compression case).

Chronicles demonstrably used Samuel-Kings as a source (general scholarly consensus)
and characteristically ABBREVIATES it. That makes this the crucial complement to
Jude -> 2 Peter (where the copy is longer): here the copy is usually SHORTER, so a
probe that scores direction correctly on BOTH cannot be riding a length shortcut.

Text: Swete LXX (CC BY-SA), word + versification CSVs. Book codes: 2Sa/1Ki/2Ki are
the source (LXX Kingdoms), 1Ch/2Ch the copy (Paraleipomenon). Blocks follow the
standard Chronicles synopsis; multi-verse blocks tolerate minor LXX/MT versification
differences. Source A = Samuel-Kings, copy B = Chronicles, so direction = A_to_B.

Output: data/external/lxx_chronicles_pairs.json (same schema as the Jude/2Pet set).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (source: book, ch, v0, v1) , (copy: book, ch, v0, v1) , topic
_PARALLELS = [
    (("2Sa", 5, 1, 3), ("1Ch", 11, 1, 3), "all Israel anoints David king"),
    (("2Sa", 5, 6, 10), ("1Ch", 11, 4, 9), "David captures Jerusalem"),
    (("2Sa", 6, 1, 11), ("1Ch", 13, 5, 14), "the ark and Uzzah"),
    (("2Sa", 7, 1, 17), ("1Ch", 17, 1, 15), "Nathan's oracle / Davidic covenant"),
    (("2Sa", 7, 18, 29), ("1Ch", 17, 16, 27), "David's prayer of response"),
    (("2Sa", 8, 1, 14), ("1Ch", 18, 1, 13), "David's victories"),
    (("2Sa", 10, 1, 19), ("1Ch", 19, 1, 19), "war with the Ammonites"),
    (("2Sa", 24, 1, 9), ("1Ch", 21, 1, 6), "the census"),
    (("1Ki", 3, 4, 15), ("2Ch", 1, 3, 13), "Solomon's dream at Gibeon"),
    (("1Ki", 8, 1, 11), ("2Ch", 5, 2, 14), "the ark brought into the temple"),
    (("1Ki", 8, 22, 30), ("2Ch", 6, 12, 21), "Solomon's dedication prayer"),
    (("1Ki", 10, 1, 13), ("2Ch", 9, 1, 12), "the queen of Sheba"),
    (("1Ki", 22, 1, 9), ("2Ch", 18, 1, 8), "Micaiah and Ahab"),
    (("2Ki", 18, 13, 19), ("2Ch", 32, 1, 8), "Sennacherib threatens Hezekiah"),
    # Additional well-established Chronicles-synopsis blocks (all Samuel-Kings source).
    (("2Sa", 5, 11, 25), ("1Ch", 14, 1, 16), "Hiram, David's house, Philistine wars"),
    (("2Sa", 23, 8, 39), ("1Ch", 11, 10, 47), "David's mighty men"),
    (("2Sa", 6, 12, 19), ("1Ch", 15, 25, 29), "the ark brought up with rejoicing"),
    (("1Ki", 5, 1, 12), ("2Ch", 2, 1, 16), "preparations for the temple"),
    (("1Ki", 7, 23, 51), ("2Ch", 4, 1, 22), "the temple furnishings"),
    (("1Ki", 8, 12, 21), ("2Ch", 6, 1, 11), "Solomon's address to the assembly"),
    (("1Ki", 8, 54, 66), ("2Ch", 7, 1, 10), "fire from heaven, dedication feast"),
    (("1Ki", 9, 1, 9), ("2Ch", 7, 11, 22), "the Lord's second appearance to Solomon"),
    (("1Ki", 12, 1, 19), ("2Ch", 10, 1, 19), "Rehoboam and the northern revolt"),
    (("1Ki", 14, 21, 31), ("2Ch", 12, 1, 16), "Rehoboam's reign and Shishak"),
    (("1Ki", 15, 1, 8), ("2Ch", 13, 1, 22), "Abijah of Judah"),
    (("1Ki", 22, 41, 50), ("2Ch", 20, 31, 37), "Jehoshaphat's reign"),
    (("2Ki", 12, 1, 16), ("2Ch", 24, 1, 14), "Joash repairs the temple"),
    (("2Ki", 14, 1, 14), ("2Ch", 25, 1, 24), "Amaziah of Judah"),
    (("2Ki", 22, 1, 13), ("2Ch", 34, 1, 21), "Josiah and the book of the law"),
    (("2Ki", 23, 21, 30), ("2Ch", 35, 1, 27), "Josiah's passover and death"),
]


def _load(swete_dir: Path) -> tuple[dict, list[str]]:
    """Return verse->(start_idx,end_idx) map and the ordered word list."""
    words = [""]  # 1-indexed
    for line in (swete_dir / "02-Swete_word_without_punctuations.csv").read_text(
        encoding="utf-8",
    ).splitlines():
        if "\t" not in line:
            continue
        _, w = line.split("\t", 1)
        words.append(w.strip())

    starts: list[tuple[int, str]] = []
    for line in (swete_dir / "00-Swete_versification.csv").read_text(
        encoding="utf-8",
    ).splitlines():
        if "\t" not in line:
            continue
        idx, ref = line.split("\t", 1)
        starts.append((int(idx), ref.strip()))
    starts.sort()

    verse_span: dict[str, tuple[int, int]] = {}
    for i, (idx, ref) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(words)
        verse_span[ref] = (idx, end)
    return verse_span, words


def _text(verse_span: dict, words: list[str], book: str, ch: int, v0: int, v1: int) -> str:
    out: list[str] = []
    for v in range(v0, v1 + 1):
        ref = f"{book}.{ch}:{v}"
        if ref in verse_span:
            s, e = verse_span[ref]
            out.extend(w for w in words[s:e] if w)
    return " ".join(out)


def main() -> None:
    """Extract LXX Chronicles parallels and write the pairs JSON."""
    parser = argparse.ArgumentParser(description="Build LXX Kings->Chronicles pairs")
    parser.add_argument("--swete-dir", type=Path, required=True,
                        help="Path to the cloned eliranwong/LXX-Swete-1930 repo")
    args = parser.parse_args()

    verse_span, words = _load(args.swete_dir)
    pairs = []
    for (sb, sc, s0, s1), (cb, cc, c0, c1), topic in _PARALLELS:
        text_a = _text(verse_span, words, sb, sc, s0, s1)
        text_b = _text(verse_span, words, cb, cc, c0, c1)
        if len(text_a.split()) < 5 or len(text_b.split()) < 5:
            print(f"  SKIP {sb}{sc} / {cb}{cc}: empty extraction (versification mismatch)")
            continue
        pid = f"lxx_{sb}{sc}_{cb}{cc}"
        pairs.append({
            "id": pid, "group": pid, "direction": "A_to_B",
            "book_a": f"{sb}{sc}", "book_b": f"{cb}{cc}",
            "ref_a": f"{sb} {sc}:{s0}-{s1}", "ref_b": f"{cb} {cc}:{c0}-{c1}",
            "topic": topic,
            "len_a": len(text_a.split()), "len_b": len(text_b.split()),
            "text_a": text_a, "text_b": text_b,
            "consensus_source": "Chronicles' use of Samuel-Kings (general scholarly consensus)",
        })

    out = {
        "description": (
            "LXX Samuel-Kings -> Chronicles known-direction pairs (the copy usually "
            "ABBREVIATES the source; complements Jude->2Peter where the copy is longer)."
        ),
        "license": "Swete LXX text CC BY-SA (eliranwong/LXX-Swete-1930)",
        "pairs": pairs,
    }
    dest = Path("data/external/lxx_chronicles_pairs.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(pairs)} pairs -> {dest}")
    shorter = sum(1 for p in pairs if p["len_b"] < p["len_a"])
    for p in pairs:
        arrow = "shorter" if p["len_b"] < p["len_a"] else "longer "
        print(f"  {p['id']}: {p['ref_a']} ({p['len_a']}w) -> {p['ref_b']} "
              f"({p['len_b']}w, copy {arrow}) | {p['topic']}")
    print(f"copy is shorter in {shorter}/{len(pairs)} pairs (compression case)")


if __name__ == "__main__":
    main()
