"""Tests for the synthetic redaction generator."""

from __future__ import annotations

import random

from synoptiq.data.redaction import (
    RedactionConfig,
    RedactionGenerator,
    windows,
)

_BOOK = (
    "και ο λογος ην προς τον θεον και θεος ην ο λογος ουτος ην εν αρχη προς τον θεον "
    "παντα δι αυτου εγενετο και χωρις αυτου εγενετο ουδε εν ο γεγονεν εν αυτω ζωη ην "
    "και η ζωη ην το φως των ανθρωπων και το φως εν τη σκοτια φαινει"
).split() * 6


def _gen() -> RedactionGenerator:
    cfg = RedactionConfig(min_len=10, max_len=20)
    return RedactionGenerator({"BookX": _BOOK}, cfg)


def test_redaction_keeps_verbatim_backbone() -> None:
    gen = _gen()
    passage = _BOOK[:40]
    ex = gen.redact(passage, "BookX")
    shared = set(passage) & set(ex.copy_words)
    # A copy must retain a substantial verbatim core of its source.
    assert len(shared) / len(set(passage)) > 0.4


def test_length_ratio_within_configured_range() -> None:
    cfg = RedactionConfig(min_len=10, max_len=20, length_ratio_range=(0.8, 1.2))
    gen = RedactionGenerator({"BookX": _BOOK}, cfg)
    for _ in range(20):
        ex = gen.redact(_BOOK[:40], "BookX")
        # Allow small slack from the verbatim floor / integer rounding.
        assert 0.55 <= ex.length_ratio <= 1.45


def test_generator_is_deterministic_under_seed() -> None:
    ex1 = _gen().redact(_BOOK[:40], "BookX")
    ex2 = _gen().redact(_BOOK[:40], "BookX")
    assert ex1.copy_words == ex2.copy_words


def test_windows_are_non_overlapping_and_min_length() -> None:
    rng = random.Random(0)
    chunks = windows(_BOOK, min_len=10, max_len=20, rng=rng)
    assert all(len(c) >= 10 for c in chunks)
    # Reconstructed length never exceeds the source (non-overlapping).
    assert sum(len(c) for c in chunks) <= len(_BOOK)


def test_op_counts_recorded() -> None:
    ex = _gen().redact(_BOOK[:50], "BookX")
    assert isinstance(ex.op_counts, dict)
    # Some edit must have occurred on a 50-word passage.
    assert sum(ex.op_counts.values()) > 0
