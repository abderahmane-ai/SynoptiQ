"""Build the synthetic same-author redaction corpus for direction learning.

Generates source→copy pairs (directional redaction) plus same-author independent
negatives, from SBLGNT books, with author(book)-disjoint train/val/test splits.

Anti-confound guarantees (all verified at the end):
  - same author within every pair (style carries no direction signal);
  - gross length decorrelated from direction (|corr| printed, must be ~0);
  - swap-balanced (both orderings emitted);
  - book-disjoint splits (generalization to unseen authors);
  - synoptic-test books (Matt/Mark/Luke) and external books (Jude/2Pet) EXCLUDED.

Output: data/synthetic/redaction_corpus.json (git-ignored; regenerate from this
script, which is the tracked source of truth).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.redaction import (  # noqa: E402
    RedactionConfig,
    RedactionGenerator,
    log_length_ratio,
    windows,
)
from synoptiq.utils.greek import strip_punctuation  # noqa: E402

_APPARATUS = re.compile(r"[⸀-⹿]")

# Held out from synthetic training so the eval ladder stays honest.
_EXCLUDED = {"Matt", "Mark", "Luke", "Jude", "2Pet"}

# Book-disjoint splits (by SBLGNT filename stem). Test/val get whole works unseen
# in training, so accuracy there reflects generalization to new authors. Large books
# (John, Acts, 1Cor, Rom, Heb, Rev) stay in train; val/test are smaller diverse works.
_VAL_BOOKS = {"Col", "2Tim", "1Pet"}
_TEST_BOOKS = {"Phil", "1Thess", "1Tim", "Jas", "1John"}

_DIRECTION_TO_IDX = {"A_to_B": 0, "B_to_A": 1, "independent": 2}


def _clean_words(text: str) -> list[str]:
    text = _APPARATUS.sub("", text)
    return [w for w in (strip_punctuation(t) for t in text.split()) if w]


def _load_book_words(txt_path: Path) -> list[str]:
    words: list[str] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        _, text = line.split("\t", 1)
        words.extend(_clean_words(text))
    return words


def _split_of(book: str) -> str:
    if book in _TEST_BOOKS:
        return "test"
    if book in _VAL_BOOKS:
        return "val"
    return "train"


def main() -> None:
    """Generate and write the synthetic redaction corpus."""
    text_dir = Path("data/raw/sblgnt/data/sblgnt/text")
    books = sorted(
        p.stem for p in text_dir.glob("*.txt") if p.stem not in _EXCLUDED
    )
    vocab_by_book = {b: _load_book_words(text_dir / f"{b}.txt") for b in books}
    vocab_by_book = {b: w for b, w in vocab_by_book.items() if len(w) >= 120}

    cfg = RedactionConfig()
    gen = RedactionGenerator(vocab_by_book, cfg)
    rng = random.Random(cfg.seed)

    pairs: list[dict] = []
    uid = 0
    for book, words in vocab_by_book.items():
        split = _split_of(book)
        passages = windows(words, min_len=cfg.min_len, max_len=cfg.max_len, rng=rng)
        if len(passages) < 2:
            continue

        # Directed pairs: each passage -> a directional copy (A=source, B=copy).
        for passage in passages:
            ex = gen.redact(passage, book)
            if len(ex.copy_words) < 5:
                continue
            group = f"{book}_{uid}"
            uid += 1
            text_a = " ".join(ex.source_words)
            text_b = " ".join(ex.copy_words)
            pairs.append({
                "group": group, "book": book, "split": split,
                "direction": "A_to_B", "text_a": text_a, "text_b": text_b,
                "log_len_ratio": log_length_ratio(ex),
            })
            pairs.append({
                "group": group, "book": book, "split": split,
                "direction": "B_to_A", "text_a": text_b, "text_b": text_a,
                "log_len_ratio": -log_length_ratio(ex),
            })

        # Independent negatives: two disjoint same-author passages (no derivation).
        rng.shuffle(passages)
        for i in range(0, len(passages) - 1, 2):
            p1, p2 = passages[i], passages[i + 1]
            if min(len(p1), len(p2)) < 5:
                continue
            group = f"{book}_{uid}"
            uid += 1
            t1, t2 = " ".join(p1), " ".join(p2)
            ratio = np.log(max(len(p1), 1) / max(len(p2), 1))
            pairs.append({
                "group": group, "book": book, "split": split,
                "direction": "independent", "text_a": t1, "text_b": t2,
                "log_len_ratio": float(ratio),
            })
            pairs.append({
                "group": group, "book": book, "split": split,
                "direction": "independent", "text_a": t2, "text_b": t1,
                "log_len_ratio": float(-ratio),
            })

    # ── Validate the anti-confound guarantees ────────────────────────────
    y = np.array([_DIRECTION_TO_IDX[p["direction"]] for p in pairs])
    llr = np.array([p["log_len_ratio"] for p in pairs])
    directed = y != 2
    y_sign = np.where(y[directed] == 0, 1.0, -1.0)
    length_dir_corr = float(np.corrcoef(llr[directed], y_sign)[0, 1])

    counts = {d: int((y == i).sum()) for d, i in _DIRECTION_TO_IDX.items()}
    split_counts = {s: sum(1 for p in pairs if p["split"] == s) for s in ("train", "val", "test")}

    out = {
        "config": {
            "smoothing_rate": cfg.smoothing_rate,
            "connective_smooth_rate": cfg.connective_smooth_rate,
            "length_ratio_range": list(cfg.length_ratio_range),
            "seed": cfg.seed,
        },
        "excluded_books": sorted(_EXCLUDED),
        "n_pairs": len(pairs),
        "class_counts": counts,
        "split_counts": split_counts,
        "length_direction_corr": round(length_dir_corr, 4),
        "pairs": pairs,
    }

    dest = Path("data/synthetic/redaction_corpus.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(pairs)} pairs -> {dest}")
    print(f"  class counts: {counts}")
    print(f"  split counts: {split_counts}")
    print(f"  books used: {len(vocab_by_book)} (excluded {sorted(_EXCLUDED)})")
    print(f"  length<->direction corr: {length_dir_corr:+.4f}  (must be ~0)")

    # Human-inspectable sample dump.
    sample_path = Path("outputs/direction/redaction_samples.md")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Redaction corpus — sample pairs\n"]
    for p in [q for q in pairs if q["direction"] == "A_to_B"][:8]:
        lines.append(f"## {p['group']} ({p['book']}, {p['split']})\n")
        lines.append(f"**source (A):** {p['text_a']}\n")
        lines.append(f"**copy (B):** {p['text_b']}\n")
    sample_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  sample dump: {sample_path}")


if __name__ == "__main__":
    main()
