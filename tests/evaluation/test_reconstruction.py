"""Tests for Track-A reconstruction-quality metrics."""

from __future__ import annotations

import pytest

from synoptiq.evaluation.reconstruction import (
    evaluate_reconstruction,
    exact_match,
    nearest_witness_baseline,
    token_f1,
)


def test_token_f1_perfect_and_disjoint() -> None:
    assert token_f1("α β γ", "α β γ") == pytest.approx(1.0)
    assert token_f1("α β γ", "δ ε ζ") == 0.0


def test_token_f1_partial_overlap() -> None:
    # pred {α β γ δ}, gold {α β}: overlap 2; precision 2/4, recall 2/2 → F1 = 2*.5*1/1.5
    f1 = token_f1("α β γ δ", "α β")
    assert f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_token_f1_is_order_insensitive() -> None:
    assert token_f1("α β γ", "γ β α") == pytest.approx(1.0)


def test_token_f1_empty_cases() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("α", "") == 0.0
    assert token_f1("", "α") == 0.0


def test_token_f1_normalization() -> None:
    upper = lambda s: s.upper()  # noqa: E731
    assert token_f1("Alpha beta", "ALPHA BETA", normalize=upper) == pytest.approx(1.0)


def test_exact_match() -> None:
    assert exact_match("α β", "α β") == 1.0
    assert exact_match("α β", "β α") == 0.0  # order matters for exact match


def test_evaluate_reconstruction_aggregates() -> None:
    preds = ["α β γ", "δ ε"]
    golds = ["α β γ", "δ ζ"]
    res = evaluate_reconstruction(preds, golds)
    assert res.n == 2
    assert res.per_example_f1[0] == pytest.approx(1.0)
    assert res.mean_exact_match == pytest.approx(0.5)
    assert 0.0 < res.mean_f1 <= 1.0


def test_evaluate_reconstruction_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ"):
        evaluate_reconstruction(["a"], ["a", "b"])


def test_evaluate_reconstruction_empty() -> None:
    res = evaluate_reconstruction([], [])
    assert res.n == 0 and res.mean_f1 == 0.0


def test_nearest_witness_baseline_takes_best() -> None:
    # Luke matches the gold better than Matthew → baseline uses Luke.
    b = nearest_witness_baseline(["α x y", "α β γ"], "α β γ")
    assert b == pytest.approx(1.0)
