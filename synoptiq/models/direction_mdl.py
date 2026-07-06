"""MDL direction head — a swap-equivariant classifier over NLL codelength features.

This is the learnable direction component. It consumes the FEATURE_NAMES vector
(derived from the four NLL codelengths in synoptiq.evaluation.nll_direction) and emits
3-way direction logits [d, -d, i] (A_to_B, B_to_A, independent), so swapping A and B
exchanges the two directed classes by construction — the same guarantee as the
similarity-feature DirectionClassifier, but over the compression features that Phase A
showed actually move with direction.

It is trained on the synthetic same-author, length-decorrelated redaction corpus, where
direction is decoupled from both authorship and length, so the weights cannot exploit
either shortcut. On the synoptics it is used for inference only (never fit on 2SH
labels), keeping the eventual Bayesian model comparison non-circular.

Feature symmetry under an A<->B swap (indices into FEATURE_NAMES):
  swap pairs   : (0,1) NLL conditionals, (3,4) marginals, (6,7) info-gains
  negate       : 2 cond_asym, 5 marg_asym, 8 infogain_asym, 9 mdl_score, 10 log_len_ratio
"""

from __future__ import annotations

import torch
import torch.nn as nn

# (left, right) feature indices that exchange under a swap.
_SWAP_PAIRS = ((0, 1), (3, 4), (6, 7))
# feature indices that negate under a swap.
_NEGATE = (2, 5, 8, 9, 10)
_N_FEATURES = 11
_N_SIGNED = 8   # 3 pair-differences + 5 negating features
_N_INDEP = 3 + _N_SIGNED  # 3 symmetric means + |signed|


class MDLDirectionHead(nn.Module):
    """Swap-equivariant linear head over the 11 NLL codelength features."""

    def __init__(self) -> None:
        super().__init__()
        self.direction_head = nn.Linear(_N_SIGNED, 1, bias=False)
        self.independence_head = nn.Linear(_N_INDEP, 1, bias=True)
        self.register_buffer("feature_mean", torch.zeros(_N_FEATURES))
        self.register_buffer("feature_std", torch.ones(_N_FEATURES))

    @torch.no_grad()
    def set_feature_stats(
        self, mean: torch.Tensor, std: torch.Tensor, min_std: float = 1e-6,
    ) -> None:
        """Fix standardization stats; symmetric pairs share stats, negators center at 0.

        Sharing mean/std within a swap pair and zero-centering the negating features
        is what keeps standardization from breaking the swap geometry.
        """
        m = mean.to(self.feature_mean.device).clone()
        s = std.to(self.feature_std.device).clone()
        for li, ri in _SWAP_PAIRS:
            shared_m = 0.5 * (m[li] + m[ri])
            shared_s = torch.sqrt(0.5 * (s[li].square() + s[ri].square()))
            m[li] = m[ri] = shared_m
            s[li] = s[ri] = shared_s
        for i in _NEGATE:
            m[i] = 0.0
        self.feature_mean.copy_(m)
        self.feature_std.copy_(s.clamp_min(min_std))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def decompose(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split standardized features into anti-symmetric (signed) and symmetric parts."""
        x = self._standardize(features)
        signed = torch.stack([
            x[:, 0] - x[:, 1],
            x[:, 3] - x[:, 4],
            x[:, 6] - x[:, 7],
            x[:, 2], x[:, 5], x[:, 8], x[:, 9], x[:, 10],
        ], dim=1)
        symmetric = torch.stack([
            0.5 * (x[:, 0] + x[:, 1]),
            0.5 * (x[:, 3] + x[:, 4]),
            0.5 * (x[:, 6] + x[:, 7]),
        ], dim=1)
        independence = torch.cat([symmetric, signed.abs()], dim=1)
        return signed, independence

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[B, 11] -> [B, 3] logits ordered (A_to_B, B_to_A, independent)."""
        signed, independence = self.decompose(features)
        d = self.direction_head(signed).squeeze(-1)
        i = self.independence_head(independence).squeeze(-1)
        return torch.stack([d, -d, i], dim=1)
