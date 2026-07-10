"""Tests for the NLL scoring backbone (no trained model required)."""

from __future__ import annotations

import math

import pytest
import torch

from synoptiq.evaluation.scoring import (
    aggregate_pericope_nll,
    bottleneck_nll,
    log_mean_exp,
    per_token_nll,
    sequence_nll,
)


def test_per_token_nll_matches_manual_logsoftmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 7)
    labels = torch.tensor([[1, 3, 0, 5]])
    nll = per_token_nll(logits, labels)
    expected = -torch.log_softmax(logits, dim=-1)[0, torch.arange(4), labels[0]]
    assert torch.allclose(nll[0], expected, atol=1e-6)


def test_per_token_nll_masks_ignore_index() -> None:
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[2, -100, 4]])
    nll = per_token_nll(logits, labels)
    assert nll[0, 1].item() == 0.0
    assert nll[0, 0].item() > 0.0


def test_confident_correct_prediction_has_near_zero_nll() -> None:
    logits = torch.full((1, 1, 4), -10.0)
    logits[0, 0, 2] = 10.0  # nearly all mass on token 2
    nll = per_token_nll(logits, torch.tensor([[2]]))
    assert nll.item() < 1e-3


def test_sequence_nll_mean_vs_sum() -> None:
    logits = torch.randn(2, 5, 6)
    labels = torch.tensor([[1, 2, 3, -100, -100], [0, 1, 2, 3, 4]])
    total = sequence_nll(logits, labels, reduction="sum")
    mean = sequence_nll(logits, labels, reduction="mean")
    # row 0 has 3 real tokens, row 1 has 5
    assert mean[0].item() == pytest.approx(total[0].item() / 3, rel=1e-5)
    assert mean[1].item() == pytest.approx(total[1].item() / 5, rel=1e-5)


def test_sequence_nll_bad_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        sequence_nll(torch.randn(1, 2, 3), torch.tensor([[0, 1]]), reduction="median")


def test_per_token_nll_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        per_token_nll(torch.randn(1, 4, 7), torch.tensor([[1, 2, 3]]))


def test_log_mean_exp_matches_reference() -> None:
    v = torch.tensor([0.0, math.log(2.0), math.log(3.0)])  # exp = [1,2,3], mean=2
    assert log_mean_exp(v).item() == pytest.approx(math.log(2.0), abs=1e-6)


def test_log_mean_exp_is_stable_for_large_values() -> None:
    v = torch.tensor([1000.0, 1000.0, 1000.0])
    assert log_mean_exp(v).item() == pytest.approx(1000.0, abs=1e-4)


def test_bottleneck_nll_within_mixture_bounds() -> None:
    # A uniform 1/K mixture of probabilities satisfies
    #   min_nll <= −log((1/K)Σ p_k) <= min_nll + log(K):
    # the marginal is never better than the best source, and at worst dilutes it
    # by the factor K when one source dominates.
    sample_nll = torch.tensor([2.0, 5.0, 9.0])
    marginal = bottleneck_nll(sample_nll).item()
    min_nll = sample_nll.min().item()
    assert min_nll - 1e-6 <= marginal <= min_nll + math.log(3) + 1e-6


def test_bottleneck_single_sample_equals_that_sample() -> None:
    sample_nll = torch.tensor([3.7])
    assert bottleneck_nll(sample_nll).item() == pytest.approx(3.7, abs=1e-6)


def test_aggregate_pericope_nll_averages_unmasked() -> None:
    token_nll = torch.tensor([1.0, 2.0, 0.0, 3.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    assert aggregate_pericope_nll(token_nll, mask) == pytest.approx((1 + 2 + 3) / 3)


def test_aggregate_pericope_nll_empty_is_zero() -> None:
    assert aggregate_pericope_nll(torch.zeros(3), torch.zeros(3)) == 0.0
