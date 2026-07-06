"""Tests for the swap-equivariant MDL direction head."""

from __future__ import annotations

import torch

from synoptiq.models.direction_mdl import MDLDirectionHead


def _swap_features(f: torch.Tensor) -> torch.Tensor:
    """Apply the A<->B swap transform to the 11 NLL codelength features."""
    s = f.clone()
    for li, ri in ((0, 1), (3, 4), (6, 7)):
        s[:, li] = f[:, ri]
        s[:, ri] = f[:, li]
    for i in (2, 5, 8, 9, 10):
        s[:, i] = -f[:, i]
    return s


def test_head_is_swap_equivariant() -> None:
    torch.manual_seed(0)
    head = MDLDirectionHead()
    head.set_feature_stats(torch.zeros(11), torch.ones(11))
    with torch.no_grad():
        head.direction_head.weight.normal_()
        head.independence_head.weight.normal_()

    f = torch.randn(5, 11)
    logits = head(f)
    swapped = head(_swap_features(f))

    torch.testing.assert_close(swapped[:, 0], logits[:, 1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(swapped[:, 1], logits[:, 0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(swapped[:, 2], logits[:, 2], atol=1e-5, rtol=1e-4)


def test_set_feature_stats_shares_pairs_and_zeros_negators() -> None:
    head = MDLDirectionHead()
    mean = torch.arange(11, dtype=torch.float32) + 1.0
    std = torch.arange(11, dtype=torch.float32) + 1.0
    head.set_feature_stats(mean, std)
    for li, ri in ((0, 1), (3, 4), (6, 7)):
        assert head.feature_mean[li].item() == head.feature_mean[ri].item()
        assert head.feature_std[li].item() == head.feature_std[ri].item()
    for i in (2, 5, 8, 9, 10):
        assert head.feature_mean[i].item() == 0.0


def test_forward_shapes() -> None:
    head = MDLDirectionHead()
    out = head(torch.randn(3, 11))
    assert out.shape == (3, 3)
