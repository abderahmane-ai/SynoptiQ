"""Tests for the Phase 3 direction classifier head."""

import torch
import torch.nn as nn

from synoptiq.models.direction import (
    DirectionClassifier,
    DirectionScorer,
    DirectionScorerConfig,
)


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
