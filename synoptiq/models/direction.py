"""Direction scorer — detects direction of literary copying between parallel passages.

The model encodes each passage independently through the frozen KoineFormer encoder,
then extracts 10 asymmetry features from the cross-similarity matrix of their hidden
states. A swap-equivariant classifier maps those features to 3-way direction labels.

Feature design is grounded in BERTScore Precision/Recall asymmetry (Zhang et al. 2020):
when B copies A, A's tokens reproduce well in B (high recall) but B's tokens do not
fully cover A (higher recall than precision), producing a measurable R–P gap. Additional
features capture attention-entropy asymmetry and structural statistics.

The encoder is completely frozen. Total trainable parameters: 17.

History:
  - Parameter-heavy learned matching heads overfit immediately on the 250-sample
    training set, collapsing to majority-class prediction.
  - A logistic regression on pooled encoder states achieves 72.8% — the signal is
    present in the hidden states but is entangled with authorship style. These features
    are designed to isolate the causal direction component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class DirectionScorerConfig:
    """Configuration for the DirectionScorer."""

    n_features: int = 10
    num_classes: int = 3
    attn_temperature: float = 10.0  # softmax temperature for entropy features
    coverage_threshold: float = 0.3  # cosine threshold for "covered" token count


# ── Feature extraction ────────────────────────────────────────────────────────


def _compute_asymmetry_features(
    h_a: torch.Tensor,    # [B, L_A, D]
    h_b: torch.Tensor,    # [B, L_B, D]
    mask_a: torch.Tensor, # [B, L_A] — bool, True for real tokens
    mask_b: torch.Tensor, # [B, L_B] — bool, True for real tokens
    config: DirectionScorerConfig,
) -> torch.Tensor:
    """Extract 10 directional asymmetry features from encoder hidden states.

    All features are computed from the pairwise cosine-similarity matrix
    S[i, j] = cos(h_A[i], h_B[j]), restricted to non-padding positions.

    Feature groups:
        1–5  BERTScore P/R asymmetry and spread
        6    Attention-entropy asymmetry
        7    Diagonal alignment strength (order-preserving copy signal)
        8    Log length ratio
        9    Symmetric pooled-embedding cosine similarity (reference)
        10   Coverage asymmetry

    Args:
        h_a, h_b: Hidden states from the frozen encoder.
        mask_a, mask_b: Boolean masks, True for non-padding positions.
        config: Scorer configuration.

    Returns:
        Tensor of shape [B, 10].
    """
    batch_size = h_a.shape[0]
    device = h_a.device
    eps = 1e-8
    temp = config.attn_temperature
    thresh = config.coverage_threshold
    all_features: list[torch.Tensor] = []

    for b in range(batch_size):
        len_a = int(mask_a[b].sum().item())
        len_b = int(mask_b[b].sum().item())

        # Guard against degenerate (zero-length) passages
        len_a = max(len_a, 1)
        len_b = max(len_b, 1)

        # L2-normalised token embeddings for cosine similarity
        h_a_valid = F.normalize(h_a[b, :len_a], p=2, dim=1)  # [len_a, D]
        h_b_valid = F.normalize(h_b[b, :len_b], p=2, dim=1)  # [len_b, D]

        # Cross-similarity matrix  S[i, j] = cos(h_A[i], h_B[j])
        S = h_a_valid @ h_b_valid.T  # [len_a, len_b]

        # ── Features 1–5: BERTScore Precision / Recall ────────────────────
        # Recall  R(A→B): for each A-token, best match in B
        #   Low R means A has content not reproduced in B (B compressed A)
        row_max = S.max(dim=1).values   # [len_a]
        bertscore_recall = row_max.mean()

        # Precision  P(B→A): for each B-token, best match in A
        #   High P means every B token traces back to A (B drew from A's vocabulary)
        col_max = S.max(dim=0).values   # [len_b]
        bertscore_precision = col_max.mean()

        # P − R: primary directional signal
        #   > 0  →  B copies A   (B traces to A; A not fully reproduced = compression)
        #   < 0  →  A copies B
        #   ≈ 0  →  independent
        asym_pr = bertscore_precision - bertscore_recall

        row_max_std = row_max.std() if len_a > 1 else row_max.new_zeros(())
        col_max_std = col_max.std() if len_b > 1 else col_max.new_zeros(())

        # ── Feature 6: Attention-entropy asymmetry ────────────────────────
        # For each query row, compute entropy of the temperature-scaled softmax.
        # Low entropy = focused (specific) attention = the token has one clear correspondent.
        # When B copies A:
        #   B-tokens attend specifically to A (they came from A → low H(B→A))
        #   A-tokens may scatter over B (B compressed A → high H(A→B))
        # So H(A→B) − H(B→A) > 0 signals "B copies A"
        def _mean_row_entropy(mat: torch.Tensor) -> torch.Tensor:
            p = torch.softmax(mat * temp, dim=1)
            return -(p * (p + eps).log()).sum(dim=1).mean()

        entropy_asym = _mean_row_entropy(S) - _mean_row_entropy(S.T)

        # ── Feature 7: Diagonal alignment strength ────────────────────────
        # For each A-token, the index of its best B-match.
        # If copying preserves word order, best matches cluster near the diagonal.
        row_argmax = S.argmax(dim=1).float()        # [len_a]
        expected_pos = (
            torch.arange(len_a, device=device).float()
            / len_a * len_b
        )
        # Closeness ∈ [0, 1]; 1.0 = perfectly diagonal
        diagonal_closeness = (
            1.0 - (row_argmax - expected_pos).abs() / (len_b + eps)
        ).mean()

        # ── Feature 8: Log length ratio ───────────────────────────────────
        # Copies are often shorter (compression) or longer (expansion).
        # Provides a structural prior independent of content similarity.
        log_len_ratio = torch.log(
            torch.tensor(len_b / (len_a + eps), device=device, dtype=torch.float)
        )

        # ── Feature 9: Symmetric pooled-embedding similarity ──────────────
        # Reference baseline — high value means passages are semantically close
        # regardless of direction.
        pooled_sim = F.cosine_similarity(
            h_a[b, :len_a].mean(dim=0),
            h_b[b, :len_b].mean(dim=0),
            dim=0,
        )

        # ── Feature 10: Coverage asymmetry ───────────────────────────────
        # "Coverage" = fraction of tokens with a close match (cosine > threshold).
        # If B copies A: most A-tokens reproduced in B (high a_cov);
        #   B may also be well-covered (high b_cov).
        # Independent: both coverages lower.
        # The difference isolates directionality beyond the overall similarity level.
        a_coverage = (row_max > thresh).float().mean()
        b_coverage = (col_max > thresh).float().mean()
        coverage_diff = a_coverage - b_coverage

        features = torch.stack([
            bertscore_recall,    # 1
            bertscore_precision, # 2
            asym_pr,             # 3 — primary directional signal (P-R > 0 means B copies A)
            row_max_std,         # 4
            col_max_std,         # 5
            entropy_asym,        # 6
            diagonal_closeness,  # 7
            log_len_ratio,       # 8
            pooled_sim,          # 9
            coverage_diff,       # 10
        ])
        all_features.append(features)

    return torch.stack(all_features)  # [B, 10]


# ── Classifier ────────────────────────────────────────────────────────────────


class DirectionClassifier(nn.Module):
    """Swap-equivariant linear probe: 10 asymmetry features → 3 classes.

    The asymmetry features are low-dimensional tabular statistics with meaningful
    absolute values and signs. A per-sample LayerNorm destroys that geometry by
    subtracting each passage pair's own feature mean. Instead, store fixed
    train-split feature statistics and learn two constrained probes:

      - direction score d(x) over anti-symmetric features
      - independence score i(x) over symmetric and absolute directional features

    The emitted logits are [d, -d, i]. Swapping A and B negates d while leaving i
    unchanged, so A_to_B and B_to_A exchange roles by construction.
    """

    def __init__(self, config: DirectionScorerConfig):
        super().__init__()
        if config.n_features != 10:
            msg = "DirectionClassifier expects the 10 handcrafted asymmetry features"
            raise ValueError(msg)
        if config.num_classes != 3:
            msg = "DirectionClassifier expects 3 classes: A_to_B, B_to_A, independent"
            raise ValueError(msg)

        self.direction_head = nn.Linear(6, 1, bias=False)
        self.independence_head = nn.Linear(10, 1, bias=True)
        self.register_buffer("feature_mean", torch.zeros(config.n_features))
        self.register_buffer("feature_std", torch.ones(config.n_features))

    @torch.no_grad()
    def set_feature_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        min_std: float = 1e-6,
    ) -> None:
        """Set fixed train-split statistics for feature z-scoring."""
        fitted_mean = mean.to(self.feature_mean.device).clone()
        fitted_std = std.to(self.feature_std.device).clone()

        for left_idx, right_idx in ((0, 1), (3, 4)):
            shared_mean = 0.5 * (fitted_mean[left_idx] + fitted_mean[right_idx])
            shared_std = torch.sqrt(
                0.5 * (fitted_std[left_idx].square() + fitted_std[right_idx].square())
            )
            fitted_mean[left_idx] = shared_mean
            fitted_mean[right_idx] = shared_mean
            fitted_std[left_idx] = shared_std
            fitted_std[right_idx] = shared_std

        for signed_idx in (2, 5, 7, 9):
            fitted_mean[signed_idx] = 0.0

        self.feature_mean.copy_(fitted_mean)
        self.feature_std.copy_(fitted_std.clamp_min(min_std))

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply train-split z-scoring to raw asymmetry features."""
        return (features - self.feature_mean) / self.feature_std

    def decompose_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split raw features into anti-symmetric and independence features."""
        x = self.normalize_features(features)

        signed = torch.stack([
            x[:, 0] - x[:, 1],  # BERTScore P/R contrast
            x[:, 2],            # P - R
            x[:, 3] - x[:, 4],  # row/column max-spread contrast
            x[:, 5],            # entropy asymmetry
            x[:, 7],            # log length ratio
            x[:, 9],            # coverage asymmetry
        ], dim=1)

        symmetric = torch.stack([
            0.5 * (x[:, 0] + x[:, 1]),  # overall token-match strength
            0.5 * (x[:, 3] + x[:, 4]),  # overall match-spread strength
            x[:, 6],                    # diagonal closeness
            x[:, 8],                    # pooled embedding similarity
        ], dim=1)

        independence = torch.cat([symmetric, signed.abs()], dim=1)
        return signed, independence

    @torch.no_grad()
    def initialize_from_centroids(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Initialise constrained heads with shared-covariance centroid rules."""
        signed, independence = self.decompose_features(features)

        a_mask = labels == 0
        b_mask = labels == 1
        independent_mask = labels == 2
        dependent_mask = labels != 2

        if a_mask.any() and b_mask.any():
            a_centroid = signed[a_mask].mean(dim=0)
            b_centroid = signed[b_mask].mean(dim=0)
            log_odds_weight = a_centroid - b_centroid
            self.direction_head.weight.copy_(0.5 * log_odds_weight.unsqueeze(0))

        if independent_mask.any() and dependent_mask.any():
            i_centroid = independence[independent_mask].mean(dim=0)
            d_centroid = independence[dependent_mask].mean(dim=0)
            i_prior = independent_mask.float().mean().clamp_min(1e-6)
            d_prior = dependent_mask.float().mean().clamp_min(1e-6)
            weight = i_centroid - d_centroid
            bias = (
                (i_prior / d_prior).log()
                - 0.5 * (i_centroid.square().sum() - d_centroid.square().sum())
            )
            self.independence_head.weight.copy_(weight.unsqueeze(0))
            self.independence_head.bias.copy_(bias.unsqueeze(0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[B, 10] → [B, 3]."""
        signed, independence = self.decompose_features(features)
        direction_logit = self.direction_head(signed).squeeze(-1)
        independent_logit = self.independence_head(independence).squeeze(-1)
        return torch.stack([
            direction_logit,
            -direction_logit,
            independent_logit,
        ], dim=1)


# ── Full model ────────────────────────────────────────────────────────────────


class DirectionScorer(nn.Module):
    """Detects literary copying direction using frozen-encoder hidden-state asymmetry.

    Architecture:
        1. Encode A and B independently with the frozen KoineFormer encoder.
        2. Compute 10 asymmetry features from the cross-similarity matrix — no
           learned attention, no additional parameters on the encoder path.
        3. Train-split z-scoring + swap-equivariant probe maps features to logits.

    Total trainable parameters: 17 (two constrained linear heads).

    Usage:
        config = DirectionScorerConfig()
        scorer = DirectionScorer(koineformer_encoder, config)
        output = scorer(input_ids_a, mask_a, input_ids_b, mask_b)
        # output["direction_logits"]: [B, 3]
        # output["asymmetry_features"]: [B, 10]
    """

    _IDX_TO_DIRECTION: dict[int, str] = {0: "A_to_B", 1: "B_to_A", 2: "independent"}

    def __init__(
        self,
        encoder: nn.Module,
        config: DirectionScorerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DirectionScorerConfig()
        self.encoder = encoder
        self.classifier = DirectionClassifier(self.config)

        # Freeze the encoder — it is used only as a feature extractor
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True) -> DirectionScorer:
        """Set classifier train/eval mode while keeping the frozen encoder stable."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        input_ids_a: torch.Tensor,       # [B, L_A]
        attention_mask_a: torch.Tensor,  # [B, L_A]
        input_ids_b: torch.Tensor,       # [B, L_B]
        attention_mask_b: torch.Tensor,  # [B, L_B]
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns:
            dict with:
                direction_logits    [B, 3]  — raw logits for cross-entropy loss
                asymmetry_features  [B, 10] — intermediate features for diagnostics
        """
        self.encoder.eval()
        with torch.no_grad():
            h_a = self.encoder(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
            ).last_hidden_state  # [B, L_A, 768]

            h_b = self.encoder(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
            ).last_hidden_state  # [B, L_B, 768]

        features = _compute_asymmetry_features(
            h_a, h_b,
            attention_mask_a.bool(),
            attention_mask_b.bool(),
            self.config,
        )  # [B, 10]

        logits = self.classifier(features)  # [B, 3]

        return {
            "direction_logits": logits,
            "asymmetry_features": features,
        }

    def predict(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Inference: predict direction for a batch of passage pairs.

        Returns:
            List of dicts with predicted_direction, per-class probabilities,
            and asymmetry_features for interpretability.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(**kwargs)
            probs = F.softmax(output["direction_logits"], dim=1)
            preds = probs.argmax(dim=1).tolist()

        results: list[dict[str, Any]] = []
        for i, pred_idx in enumerate(preds):
            results.append({
                "predicted_direction": self._IDX_TO_DIRECTION[pred_idx],
                "prob_a_to_b": round(probs[i, 0].item(), 4),
                "prob_b_to_a": round(probs[i, 1].item(), 4),
                "prob_independent": round(probs[i, 2].item(), 4),
                "asymmetry_features": output["asymmetry_features"][i].tolist(),
            })
        return results
