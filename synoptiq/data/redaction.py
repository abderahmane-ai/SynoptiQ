"""Synthetic redaction corpus: same-author copy pairs with a directional signal.

Why this exists. The synoptic triple tradition confounds copying *direction* with
*authorship*: Mark is always the source, so any probe can score well by detecting
Markan style rather than direction (demonstrated in scripts/analyze_direction_signal.py
— even zero-shot conditional NLL, length-controlled, tracks Mark-ness, not direction).
To learn direction *per se* we need pairs where:

  1. both sides are the SAME author  -> style carries no direction information;
  2. gross length is DECORRELATED from direction -> length is not a shortcut;
  3. the edit process is directionally ASYMMETRIC in a way real redactors are.

Point 3 is the subtle one. Random deletion/insertion is time-symmetric: a randomly
edited copy is statistically indistinguishable from its source, so it contains no
learnable direction. Real redaction is not symmetric — copyists *smooth* their source
toward more frequent/typical constructions (Matthew and Luke routinely improve Mark's
rough Greek) and *add* explanatory material. We therefore make the copy drift toward
the high-frequency end of the author's own vocabulary and connectives, while balancing
compression against expansion so length says nothing about which text is the source.

Source text: SBLGNT books (clean, on disk). The synoptic-test books (Matthew, Mark,
Luke) and the external-eval books (Jude, 2 Peter) are excluded by the builder so the
evaluation ladder stays honest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import random

# Koine connectives, roughly ordered rough -> smooth. Redactors tend to replace the
# paratactic καί with δέ / οὖν / τότε (a well-attested Matthean/Lukan tendency).
_CONNECTIVES_ROUGH = ("και", "καὶ")
_CONNECTIVES_SMOOTH = ("δὲ", "οὖν", "τότε", "γὰρ")


@dataclass
class RedactionConfig:
    """Controls the directional edit process that turns a source into a copy."""

    min_len: int = 30                  # min source window length (words)
    max_len: int = 90                  # max source window length
    length_ratio_range: tuple[float, float] = (0.75, 1.30)  # copy/source, symmetric-ish
    smoothing_rate: float = 0.18       # fraction of tokens pushed toward frequent vocab
    connective_smooth_rate: float = 0.55  # P(replace a rough connective with a smooth one)
    reorder_rate: float = 0.08         # fraction of adjacent swaps (local transposition)
    verbatim_floor: float = 0.45       # keep at least this fraction of source verbatim
    seed: int = 20260706
    # directionality knobs
    freq_substitute_top_k: int = 200   # substitute rare tokens with one of the top-k frequent


@dataclass
class RedactionExample:
    """One generated pair: a source passage and its directional redaction (copy)."""

    source_words: list[str]
    copy_words: list[str]
    book: str
    op_counts: dict[str, int] = field(default_factory=dict)

    @property
    def length_ratio(self) -> float:
        """len(copy) / len(source)."""
        return len(self.copy_words) / max(len(self.source_words), 1)


class RedactionGenerator:
    """Generates same-author source→copy pairs with a directional edit process."""

    def __init__(self, vocab_by_book: dict[str, list[str]], config: RedactionConfig):
        self.config = config
        self._rng = random.Random(config.seed)
        # Per-book frequency tables and the author's high-frequency vocabulary.
        self._freq: dict[str, Counter] = {}
        self._frequent: dict[str, list[str]] = {}
        for book, words in vocab_by_book.items():
            counter = Counter(w for w in words if _is_wordlike(w))
            self._freq[book] = counter
            self._frequent[book] = [w for w, _ in counter.most_common(config.freq_substitute_top_k)]

    # ── Core edit process ─────────────────────────────────────────────────

    def redact(self, source: list[str], book: str) -> RedactionExample:
        """Produce a directional copy of ``source`` in the same author's register."""
        cfg = self.config
        rng = self._rng
        freq = self._freq[book]
        frequent = self._frequent[book] or list(freq)
        ops = Counter()

        target_ratio = rng.uniform(*cfg.length_ratio_range)
        target_len = max(1, round(len(source) * target_ratio))

        words = list(source)

        # 1. Connective smoothing: rough καί -> a smoother connective (directional).
        for i, w in enumerate(words):
            base = _strip(w)
            if base in _CONNECTIVES_ROUGH and rng.random() < cfg.connective_smooth_rate:
                words[i] = rng.choice(_CONNECTIVES_SMOOTH)
                ops["connective_smooth"] += 1

        # 2. Fluency substitution: replace rare tokens with a frequent one of the same
        #    initial trigram (crude morphological neighbour) — the copy drifts toward
        #    the author's typical vocabulary. Directional and not length-changing.
        n_smooth = int(len(words) * cfg.smoothing_rate)
        rare_positions = sorted(
            range(len(words)),
            key=lambda i: freq.get(_strip(words[i]), 0),
        )[:max(n_smooth * 2, 0)]
        rng.shuffle(rare_positions)
        for i in rare_positions[:n_smooth]:
            repl = self._frequent_neighbour(_strip(words[i]), frequent, freq)
            if repl and repl != _strip(words[i]):
                words[i] = repl
                ops["fluency_substitute"] += 1

        # 3. Length adjustment: compress (delete spans) or expand (insert frequent
        #    connective+word), chosen only to hit the RANDOM target length so that
        #    "which is longer" carries no directional information.
        words = self._adjust_length(words, source, target_len, frequent, ops, rng)

        # 4. Local reorder: a few adjacent swaps (redactors reorder small units).
        n_reorder = int(len(words) * cfg.reorder_rate)
        for _ in range(n_reorder):
            if len(words) < 2:
                break
            j = rng.randrange(len(words) - 1)
            words[j], words[j + 1] = words[j + 1], words[j]
            ops["reorder"] += 1

        return RedactionExample(
            source_words=list(source), copy_words=words, book=book, op_counts=dict(ops),
        )

    def _adjust_length(
        self,
        words: list[str],
        source: list[str],
        target_len: int,
        frequent: list[str],
        ops: Counter,
        rng: random.Random,
    ) -> list[str]:
        """Delete or insert to reach target_len while keeping the verbatim floor."""
        cfg = self.config
        min_keep = max(1, int(len(source) * cfg.verbatim_floor))

        # Compression: delete random single words (not below the verbatim floor).
        while len(words) > target_len and len(words) > min_keep:
            del words[rng.randrange(len(words))]
            ops["delete"] += 1

        # Expansion: insert a frequent word (explanatory addition) at random points.
        while len(words) < target_len and frequent:
            pos = rng.randrange(len(words) + 1)
            words.insert(pos, rng.choice(frequent))
            ops["insert"] += 1
        return words

    def _frequent_neighbour(
        self, word: str, frequent: list[str], freq: Counter,
    ) -> str | None:
        """A frequent vocabulary item sharing the word's initial trigram, if any."""
        if len(word) < 3:
            return None
        prefix = word[:3]
        candidates = [w for w in frequent if w.startswith(prefix) and w != word]
        if candidates:
            return max(candidates, key=lambda w: freq.get(w, 0))
        return None


# ── Passage extraction ─────────────────────────────────────────────────────────


def windows(words: list[str], *, min_len: int, max_len: int, rng: random.Random) -> list[list[str]]:
    """Chop a book's word stream into non-overlapping variable-length passages."""
    out: list[list[str]] = []
    i = 0
    n = len(words)
    while i < n:
        length = rng.randint(min_len, max_len)
        chunk = words[i : i + length]
        if len(chunk) >= min_len:
            out.append(chunk)
        i += length
    return out


def _is_wordlike(w: str) -> bool:
    """True for tokens containing at least one Greek letter."""
    return any("Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿" for ch in w)


def _strip(w: str) -> str:
    """Lowercase a token for connective/frequency matching (accents kept)."""
    return w.lower()


def log_length_ratio(example: RedactionExample) -> float:
    """log(len(copy) / len(source)) — used to verify length decorrelation."""
    return math.log(max(len(example.copy_words), 1) / max(len(example.source_words), 1))
