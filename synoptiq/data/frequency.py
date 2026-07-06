"""Koine word-frequency table for the harder-reading (lectio difficilior) feature.

Textual criticism's oldest directional canon is *lectio difficilior potior*: the harder,
rarer reading is the earlier one, because scribes replace unfamiliar words with familiar
ones. To use it we need a notion of how *marked* (rare) a Greek word is. This module
builds a normalized-surface-form frequency table from the on-disk Koine corpus (SBLGNT)
plus whatever known-direction text we are actually scoring (the external pairs), so LXX
and NT vocabulary in play are both covered.

Rarity is reported as a bounded markedness score in [0, 1] (0 = most common, 1 = hapax /
unseen), so it composes cleanly into the polarization feature vector.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from synoptiq.utils.greek import normalize_greek

_SBLGNT_TEXT_DIR = Path("data/raw/sblgnt/data/sblgnt/text")
_EXTERNAL_DIR = Path("data/external")


class FrequencyTable:
    """Normalized-surface-form frequencies with a bounded markedness lookup."""

    def __init__(self, counts: Counter[str]) -> None:
        self._counts = counts
        self._total = max(sum(counts.values()), 1)
        self._max_log = math.log(self._total + 1)

    def count(self, word: str) -> int:
        """Raw corpus count of a word's normalized form."""
        return self._counts.get(normalize_greek(word), 0)

    def markedness(self, word: str) -> float:
        """Rarity in [0, 1]: 0 = most frequent word, 1 = unseen/hapax.

        Uses -log(count+1) scaled by the corpus size, so common words score near 0 and
        unseen words score 1.0.
        """
        c = self._counts.get(normalize_greek(word), 0)
        return 1.0 - math.log(c + 1) / self._max_log

    @property
    def vocab_size(self) -> int:
        """Number of distinct normalized forms."""
        return len(self._counts)


def _iter_sblgnt_words() -> list[str]:
    words: list[str] = []
    if not _SBLGNT_TEXT_DIR.exists():
        return words
    for txt in _SBLGNT_TEXT_DIR.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            _, text = line.split("\t", 1)
            words.extend(text.split())
    return words


def _iter_external_words() -> list[str]:
    words: list[str] = []
    if not _EXTERNAL_DIR.exists():
        return words
    for path in _EXTERNAL_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for p in data.get("pairs", []):
            words.extend(p.get("text_a", "").split())
            words.extend(p.get("text_b", "").split())
    return words


def build_frequency_table(*, include_external: bool = True) -> FrequencyTable:
    """Build the Koine frequency table from SBLGNT (+ external known-direction text).

    Fast enough (~10^5 words) to compute per run; no cache needed.
    """
    counts: Counter[str] = Counter()
    for w in _iter_sblgnt_words():
        norm = normalize_greek(w)
        if norm:
            counts[norm] += 1
    if include_external:
        for w in _iter_external_words():
            norm = normalize_greek(w)
            if norm:
                counts[norm] += 1
    return FrequencyTable(counts)
