"""Tests for redaction/fusion training-example construction."""

from __future__ import annotations

from synoptiq.data.redaction import (
    FusionExample,
    fusion_examples,
    grouped_ids,
    pericope_text,
    per_pericope_counts,
    redaction_pairs,
    source_dropout_variants,
)


def test_pericope_text_joins_surface_forms(tiny_corpus) -> None:  # noqa: ANN001
    text = pericope_text(tiny_corpus, "Mark", "020")
    # tiny_corpus pericope 020 Mark has 8 non-punctuation tokens
    assert len(text.split()) == 8
    assert "καὶ" in text


def test_redaction_pairs_mark_to_luke(tiny_corpus) -> None:  # noqa: ANN001
    pairs = redaction_pairs(tiny_corpus, source_book="Mark", target_book="Luke")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.pericope_id == "020"
    assert p.source_book == "Mark" and p.target_book == "Luke"
    assert len(p.source_text.split()) == 8   # Mark
    assert len(p.target_text.split()) == 7   # Luke


def test_redaction_pairs_respects_id_filter(tiny_corpus) -> None:  # noqa: ANN001
    assert redaction_pairs(tiny_corpus, source_book="Mark", target_book="Luke", ids=set()) == []
    kept = redaction_pairs(tiny_corpus, source_book="Mark", target_book="Luke", ids={"020"})
    assert len(kept) == 1


def test_fusion_examples_mt_lk_to_mark(tiny_corpus) -> None:  # noqa: ANN001
    ex = fusion_examples(tiny_corpus)
    assert len(ex) == 1
    e = ex[0]
    assert set(e.witness_texts) == {"Matthew", "Luke"}
    assert len(e.target_text.split()) == 8  # Mark
    assert all(t for t in e.witness_texts.values())


def test_source_dropout_variants() -> None:
    e = FusionExample("020", {"Matthew": "a b", "Luke": "c d"}, "e f")
    variants = source_dropout_variants(e)
    # full + one per witness
    assert len(variants) == 3
    witness_sets = [tuple(sorted(v.witness_texts)) for v in variants]
    assert ("Luke", "Matthew") in witness_sets
    assert ("Matthew",) in witness_sets
    assert ("Luke",) in witness_sets


def test_source_dropout_single_witness_is_noop() -> None:
    e = FusionExample("020", {"Matthew": "a b"}, "e f")
    assert source_dropout_variants(e) == [e]


def test_grouped_ids_and_counts(tiny_corpus) -> None:  # noqa: ANN001
    pairs = redaction_pairs(tiny_corpus, source_book="Mark", target_book="Luke")
    assert grouped_ids(pairs) == ["020"]
    counts = per_pericope_counts(tiny_corpus)
    assert counts["020"] == 8  # Mark tokens
