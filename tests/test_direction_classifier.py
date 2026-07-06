"""Tests for the Phase 3 direction classifier head."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from synoptiq.models.direction import (
    DirectionClassifier,
    DirectionScorer,
    DirectionScorerConfig,
    _compute_asymmetry_features,
)


class _StubEncoder(nn.Module):
    """Deterministic stand-in encoder: token ids → embedding hidden states.

    Lets us exercise the full feature-extraction + classifier path (including
    the cross-similarity geometry) without loading GreTa.
    """

    def __init__(self, vocab: int = 64, dim: int = 32) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.weight = nn.Parameter(torch.randn(vocab, dim, generator=gen))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.weight[input_ids])


def _swap_features(features: torch.Tensor) -> torch.Tensor:
    swapped = features.clone()
    swapped[:, 0] = features[:, 1]
    swapped[:, 1] = features[:, 0]
    swapped[:, 2] = -features[:, 2]
    swapped[:, 3] = features[:, 4]
    swapped[:, 4] = features[:, 3]
    swapped[:, 5] = -features[:, 5]
    swapped[:, 7] = -features[:, 7]
    swapped[:, 9] = -features[:, 9]
    return swapped


def test_direction_classifier_is_swap_equivariant() -> None:
    torch.manual_seed(7)
    classifier = DirectionClassifier(DirectionScorerConfig())
    classifier.set_feature_stats(
        torch.zeros(10),
        torch.ones(10),
    )

    features = torch.tensor([
        [0.82, 0.78, -0.04, 0.12, 0.10, -0.05, 0.84, -0.20, 0.91, 0.03],
        [0.76, 0.81, 0.05, 0.09, 0.13, 0.04, 0.83, 0.21, 0.90, -0.02],
    ])

    logits = classifier(features)
    swapped_logits = classifier(_swap_features(features))

    torch.testing.assert_close(swapped_logits[:, 0], logits[:, 1])
    torch.testing.assert_close(swapped_logits[:, 1], logits[:, 0])
    torch.testing.assert_close(swapped_logits[:, 2], logits[:, 2])


def test_feature_standardizer_preserves_swap_geometry() -> None:
    classifier = DirectionClassifier(DirectionScorerConfig())
    classifier.set_feature_stats(
        torch.tensor([1.0, 3.0, 4.0, 2.0, 8.0, 5.0, 6.0, 7.0, 9.0, 10.0]),
        torch.tensor([2.0, 4.0, 3.0, 6.0, 8.0, 5.0, 7.0, 9.0, 11.0, 13.0]),
    )

    assert classifier.feature_mean[0].item() == classifier.feature_mean[1].item()
    assert classifier.feature_mean[3].item() == classifier.feature_mean[4].item()
    assert classifier.feature_mean[2].item() == 0.0
    assert classifier.feature_mean[5].item() == 0.0
    assert classifier.feature_mean[7].item() == 0.0
    assert classifier.feature_mean[9].item() == 0.0
    assert classifier.feature_std[0].item() == classifier.feature_std[1].item()
    assert classifier.feature_std[3].item() == classifier.feature_std[4].item()


def test_diagonal_closeness_feature_is_swap_invariant() -> None:
    # Feature 7 (index 6) is placed in the classifier's symmetric group, so it
    # must be identical when A and B are swapped. A row-only computation is not.
    torch.manual_seed(3)
    h_a = torch.randn(1, 7, 32)
    h_b = torch.randn(1, 11, 32)
    mask_a = torch.ones(1, 7, dtype=torch.bool)
    mask_b = torch.ones(1, 11, dtype=torch.bool)
    cfg = DirectionScorerConfig()

    feats = _compute_asymmetry_features(h_a, h_b, mask_a, mask_b, cfg)
    feats_swapped = _compute_asymmetry_features(h_b, h_a, mask_b, mask_a, cfg)

    torch.testing.assert_close(feats[:, 6], feats_swapped[:, 6])


def test_direction_scorer_is_swap_equivariant_end_to_end() -> None:
    # Swap the ACTUAL tokenized inputs (not idealized features) and require the
    # logits to exchange as [d, -d, i] -> [-d, d, i]. This is the property the
    # feature-level test cannot see, because it is where feature 7 leaked.
    torch.manual_seed(11)
    encoder = _StubEncoder()
    scorer = DirectionScorer(encoder, DirectionScorerConfig())
    # Give the independence head non-trivial weights so an asymmetry in the
    # invariant features would actually change its logit.
    with torch.no_grad():
        scorer.classifier.independence_head.weight.normal_()
        scorer.classifier.direction_head.weight.normal_()

    ids_a = torch.randint(0, 64, (1, 9))
    ids_b = torch.randint(0, 64, (1, 14))
    mask_a = torch.ones(1, 9, dtype=torch.long)
    mask_b = torch.ones(1, 14, dtype=torch.long)

    out = scorer(ids_a, mask_a, ids_b, mask_b)["direction_logits"]
    out_swapped = scorer(ids_b, mask_b, ids_a, mask_a)["direction_logits"]

    torch.testing.assert_close(out_swapped[:, 0], out[:, 1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_swapped[:, 1], out[:, 0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_swapped[:, 2], out[:, 2], atol=1e-5, rtol=1e-4)


def test_direction_scorer_train_keeps_frozen_encoder_in_eval_mode() -> None:
    encoder = nn.Sequential(
        nn.Linear(4, 4),
        nn.Dropout(p=0.5),
    )
    scorer = DirectionScorer(encoder, DirectionScorerConfig())

    scorer.train()

    assert scorer.training
    assert scorer.classifier.training
    assert not scorer.encoder.training
    assert not encoder[1].training

    scorer.eval()

    assert not scorer.training
    assert not scorer.classifier.training
    assert not scorer.encoder.training
