"""Tests for study design: membership, folds, overlap, census, hashing."""

from __future__ import annotations

import pytest

from synoptiq.data.study_design import (
    Census,
    TripleUnit,
    build_folds,
    census,
    config_hash,
    double_tradition_units,
    fold_hash,
    full_triples,
    overlap_hash,
    overlap_partition,
    sha256,
)


def _unit(pid: str, genre: str, n: int, *, core: bool = False, ext: bool = False) -> TripleUnit:
    return TripleUnit(
        pericope_id=pid,
        genre=genre,
        token_counts={"Matthew": n, "Mark": n + 1, "Luke": n + 2},
        is_overlap_core=core,
        is_overlap_extended=ext or core,
    )


# ── Membership on the tiny corpus ─────────────────────────────────────────────


def test_full_triples_on_tiny_corpus(tiny_corpus) -> None:  # noqa: ANN001
    units = full_triples(tiny_corpus)
    assert [u.pericope_id for u in units] == ["020"]
    u = units[0]
    assert u.genre == "narrative"
    assert u.token_counts == {"Matthew": 6, "Mark": 8, "Luke": 7}
    assert u.min_tokens == 6
    assert not u.is_overlap_core


def test_double_units_on_tiny_corpus(tiny_corpus) -> None:  # noqa: ANN001
    units = double_tradition_units(tiny_corpus)
    assert [u.pericope_id for u in units] == ["088"]
    assert units[0].token_counts == {"Matthew": 5, "Luke": 4}


def test_census_flags_unlearnable_genre(tiny_corpus) -> None:  # noqa: ANN001
    c = census(tiny_corpus)
    assert isinstance(c, Census)
    assert c.n_full_triples == 1
    assert c.n_double == 1
    # triple genre is narrative, double genre is discourse → discourse unlearnable
    assert c.triple_genres == {"narrative": 1}
    assert c.double_genres == {"discourse": 1}
    assert c.unlearnable_double_genres == ["discourse"]


# ── Folds ─────────────────────────────────────────────────────────────────────


def test_folds_partition_all_units_exactly_once() -> None:
    units = [_unit(f"{i:03d}", "other", 100 + i) for i in range(20)]
    plan = build_folds(units, n_folds=5, seed=1)
    assert set(plan.assignment) == {u.pericope_id for u in units}
    # every fold non-empty, together they cover everything with no overlap
    for k in range(5):
        train, test = plan.train_ids(k), plan.test_ids(k)
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(plan.assignment)
        assert test  # non-empty


def test_folds_are_deterministic() -> None:
    units = [_unit(f"{i:03d}", "narrative" if i % 2 else "passion", 50 + i) for i in range(18)]
    a = build_folds(units, n_folds=4, seed=7)
    b = build_folds(units, n_folds=4, seed=7)
    assert a.assignment == b.assignment
    assert fold_hash(a) == fold_hash(b)


def test_folds_reseed_changes_assignment() -> None:
    units = [_unit(f"{i:03d}", "other", 50 + i) for i in range(20)]
    a = build_folds(units, n_folds=5, seed=1)
    b = build_folds(units, n_folds=5, seed=2)
    assert a.assignment != b.assignment


def test_folds_spread_overlap_units_across_folds() -> None:
    # 5 overlap + 15 non-overlap; overlaps should not all land in one fold.
    units = [_unit(f"c{i}", "other", 80, core=True) for i in range(5)]
    units += [_unit(f"r{i}", "other", 80) for i in range(15)]
    plan = build_folds(units, n_folds=5, seed=3)
    overlap_folds = {plan.assignment[f"c{i}"] for i in range(5)}
    assert len(overlap_folds) >= 3  # spread over most folds, not clumped


def test_folds_reject_too_few_units() -> None:
    with pytest.raises(ValueError, match="cannot make"):
        build_folds([_unit("001", "other", 10)], n_folds=5, seed=1)


# ── Overlap partition ─────────────────────────────────────────────────────────


def test_overlap_partition_core_vs_extended() -> None:
    units = [
        _unit("009", "other", 100, core=True),
        _unit("038", "other", 100, ext=True),  # extended only
        _unit("100", "other", 100),            # neither
    ]
    core_ov, core_rest = overlap_partition(units, scope="core")
    assert core_ov == ["009"]
    assert core_rest == ["038", "100"]

    ext_ov, ext_rest = overlap_partition(units, scope="extended")
    assert ext_ov == ["009", "038"]
    assert ext_rest == ["100"]


def test_overlap_partition_bad_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        overlap_partition([_unit("009", "other", 10, core=True)], scope="all")


# ── Hashing / freezing ────────────────────────────────────────────────────────


def test_sha256_is_stable_and_order_independent() -> None:
    assert sha256({"a": 1, "b": 2}) == sha256({"b": 2, "a": 1})
    assert sha256([1, 2, 3]) != sha256([3, 2, 1])


def test_config_hash_coerces_paths() -> None:
    from pathlib import Path

    h1 = config_hash({"output_dir": Path("outputs/study"), "n_folds": 5})
    h2 = config_hash({"output_dir": "outputs/study", "n_folds": 5})
    assert h1 == h2
    assert len(h1) == 64


def test_overlap_hash_core_differs_from_extended() -> None:
    assert overlap_hash("core") != overlap_hash("extended")
    assert len(overlap_hash("core")) == 64
