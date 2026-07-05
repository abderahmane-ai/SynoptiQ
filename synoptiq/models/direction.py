"""Direction scorer — detects direction of literary copying between parallel passages.

Uses cross-attention asymmetry between KoineFormer encoder hidden states.
The cross-attention is a learned multi-head attention that compares two
encoded passages and extracts 8 engineered asymmetry features. A 3-way MLP
classifier outputs probabilities for A→B, B→A, or independent.

No adversarial component — the GRL was removed (v1 experiment) because on
250 samples it destroyed the cross-attention gradients, producing flat
attention maps and collapsing the classifier to always predict "independent."
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
    """Configuration for the DirectionScorer model.

    All defaults match the ModelConfig in _config.py.
    """

    hidden_dim: int = 768  # KoineFormer encoder output dim
    cross_attn_heads: int = 8
    asymmetry_num_features: int = 8
    classifier_hidden: tuple[int, ...] = (256, 128)
    num_classes: int = 3  # A→B, B→A, independent
    dropout: float = 0.3
    grl_lambda_max: float = 1.0  # Max GRL gradient scale
    grl_warmup_steps: int = 1000  # Linear warmup steps for GRL lambda


# ── Gradient Reversal Layer ───────────────────────────────────────────────────


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal for adversarial domain adaptation.

    Forward: identity.  Backward: multiplies gradients by -lambda.
    Used to strip authorship style from the direction features:
    an adversarial discriminator tries to predict which gospel each
    hidden state came from, and the GRL forces the encoder to produce
    style-invariant representations.

    Usage:
        hidden = GradientReversalLayer.apply(hidden, self._grl_lambda)
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


# ── Asymmetry Feature Extraction ──────────────────────────────────────────────


def _extract_asymmetry_features(
    cross_attn_a_to_b: torch.Tensor,  # [B, heads, L_A, L_B]
    cross_attn_b_to_a: torch.Tensor,  # [B, heads, L_B, L_A]
) -> torch.Tensor:
    """Extract 8 asymmetry features from a pair of cross-attention maps.

    Args:
        cross_attn_a_to_b: Cross-attention weights A→B, averaged over heads.
        cross_attn_b_to_a: Cross-attention weights B→A, averaged over heads.

    Returns:
        Tensor of shape [B, 8] with the asymmetry features.
    """
    # Mean-pool over heads for clean statistics
    ab = cross_attn_a_to_b.mean(dim=1)  # [B, L_A, L_B]
    ba = cross_attn_b_to_a.mean(dim=1)  # [B, L_B, L_A]

    eps = 1e-8

    # Features 1-2: mean attention weights
    mean_ab = ab.mean(dim=(1, 2))  # [B]
    mean_ba = ba.mean(dim=(1, 2))  # [B]

    # Features 3-4: variance of attention weights
    var_ab = ab.var(dim=(1, 2))  # [B]
    var_ba = ba.var(dim=(1, 2))  # [B]

    # Features 5-6: entropy of attention distributions
    ent_ab = -(ab * (ab + eps).log()).sum(dim=(1, 2))  # [B]
    ent_ba = -(ba * (ba + eps).log()).sum(dim=(1, 2))  # [B]

    # Feature 7: KL asymmetry — bidirectional KL divergence ratio
    # A copying author (B) produces more focused cross-attention than the source (A)
    kl_ab_to_ba = (ab * ((ab + eps) / (ba.mean(dim=2, keepdim=True) + eps)).log()).sum(dim=(1, 2))
    kl_ba_to_ab = (ba * ((ba + eps) / (ab.mean(dim=2, keepdim=True) + eps)).log()).sum(dim=(1, 2))
    kl_asymmetry = kl_ab_to_ba - kl_ba_to_ab  # [B]

    # Feature 8: position-decay — how fast does attention decay with position?
    # A copying author attends forward to the source text; the source attends back more uniformly
    L_A = ab.shape[1]
    L_B = ba.shape[1]
    positions_ab = torch.arange(L_A, device=ab.device, dtype=ab.dtype)
    positions_ba = torch.arange(L_B, device=ba.device, dtype=ba.dtype)
    # Weighted average position attended to
    avg_pos_ab = (ab.sum(dim=2) * positions_ab[None, :]).sum(dim=1) / (ab.sum(dim=(1, 2)) + eps)
    avg_pos_ba = (ba.sum(dim=2) * positions_ba[None, :]).sum(dim=1) / (ba.sum(dim=(1, 2)) + eps)
    # Normalize by sequence length so it's comparable across passages
    pos_decay = (avg_pos_ba / (L_B + eps)) - (avg_pos_ab / (L_A + eps))  # [B]

    features = torch.stack([
        mean_ab, mean_ba, var_ab, var_ba,
        ent_ab, ent_ba, kl_asymmetry, pos_decay,
    ], dim=1)  # [B, 8]

    return features


# ── Cross-Attention Asymmetry Module ──────────────────────────────────────────


class CrossAttentionAsymmetry(nn.Module):
    """Learned cross-attention layers for detecting direction asymmetry.

    Given two encoded passages H_A and H_B, computes bidirectional
    cross-attention and extracts 8 asymmetry features.
    """

    def __init__(self, config: DirectionScorerConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.cross_attn_heads
        self.head_dim = config.hidden_dim // config.cross_attn_heads

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(
        self,
        h_a: torch.Tensor,  # [B, L_A, D]
        h_b: torch.Tensor,  # [B, L_B, D]
        mask_a: torch.Tensor | None = None,  # [B, L_A]
        mask_b: torch.Tensor | None = None,  # [B, L_B]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bidirectional cross-attention and extract asymmetry features.

        Returns:
            (asymmetry_features, pooled_a, pooled_b)
        """
        # A → B cross-attention: Q from A, K/V from B
        key_padding_a = None if mask_a is None else (~mask_a.bool())
        key_padding_b = None if mask_b is None else (~mask_b.bool())

        ab_out, ab_weights = self.cross_attn(
            query=h_a, key=h_b, value=h_b,
            key_padding_mask=key_padding_b,
            need_weights=True, average_attn_weights=False,
        )  # ab_out: [B, L_A, D], ab_weights: [B, heads, L_A, L_B]

        # B → A cross-attention: Q from B, K/V from A
        ba_out, ba_weights = self.cross_attn(
            query=h_b, key=h_a, value=h_a,
            key_padding_mask=key_padding_a,
            need_weights=True, average_attn_weights=False,
        )  # ba_out: [B, L_B, D], ba_weights: [B, heads, L_B, L_A]

        # Extract 8 asymmetry features
        asym_features = _extract_asymmetry_features(ab_weights, ba_weights)  # [B, 8]

        # Mean-pool the cross-attended outputs
        if mask_a is not None:
            pooled_a = (ab_out * mask_a.unsqueeze(-1)).sum(dim=1) / (mask_a.sum(dim=1, keepdim=True) + 1e-8)
        else:
            pooled_a = ab_out.mean(dim=1)

        if mask_b is not None:
            pooled_b = (ba_out * mask_b.unsqueeze(-1)).sum(dim=1) / (mask_b.sum(dim=1, keepdim=True) + 1e-8)
        else:
            pooled_b = ba_out.mean(dim=1)

        return asym_features, pooled_a, pooled_b


# ── Author Discriminator (GRL adversary) ──────────────────────────────────────


class AuthorDiscriminator(nn.Module):
    """Adversarial head that tries to predict gospel book from hidden states.

    Trained with a GradientReversalLayer before it, so the encoder
    learns to produce style-invariant representations.
    """

    def __init__(self, config: DirectionScorerConfig):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 3),  # Matthew, Mark, Luke
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Predict book logits from pooled hidden states. [B, D] → [B, 3]."""
        return self.discriminator(pooled)


# ── Direction Classifier ──────────────────────────────────────────────────────


class DirectionClassifier(nn.Module):
    """MLP that classifies direction from [pooled_states + asymmetry features].

    Input: concatenation of pooled_A [B, 768], pooled_B [B, 768], and
    asymmetry features [B, 8] = [B, 1544].
    Output: 3-class logits [B, 3].
    """

    def __init__(self, config: DirectionScorerConfig):
        super().__init__()
        input_dim = 2 * config.hidden_dim + config.asymmetry_num_features  # 1544

        layers: list[nn.Module] = []
        in_features = input_dim
        for h in config.classifier_hidden:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_features = h
        layers.append(nn.Linear(in_features, config.num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        """[B, 1544] → [B, 3]."""
        return self.mlp(combined)


# ── Full Direction Scorer ─────────────────────────────────────────────────────


class DirectionScorer(nn.Module):
    """Detects literary copying direction between parallel Gospel passages.

    Given two parallel passages A and B:
    1. Encode both with frozen KoineFormer encoder → H_A, H_B
    2. Compute bidirectional cross-attention → 8 asymmetry features
    3. GRL-stripped pooled states + asymmetry features → 3-way classifier
    4. Adversarial discriminator predicts authorship (for GRL training)

    Usage:
        config = DirectionScorerConfig()
        scorer = DirectionScorer(koineformer_encoder, config)
        logits, author_logits = scorer(input_ids_a, mask_a, input_ids_b, mask_b)
    """

    _IDX_TO_DIRECTION = {0: "A_to_B", 1: "B_to_A", 2: "independent"}

    def __init__(
        self,
        encoder: nn.Module,
        config: DirectionScorerConfig | None = None,
    ):
        super().__init__()
        self.config = config or DirectionScorerConfig()
        self.encoder = encoder  # Frozen KoineFormer encoder

        self.cross_attn = CrossAttentionAsymmetry(self.config)
        self.classifier = DirectionClassifier(self.config)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids_a: torch.Tensor,  # [B, L_A]
        attention_mask_a: torch.Tensor,  # [B, L_A]
        input_ids_b: torch.Tensor,  # [B, L_B]
        attention_mask_b: torch.Tensor,  # [B, L_B]
    ) -> dict[str, torch.Tensor]:
        """Forward pass — returns direction logits and asymmetry features.

        Returns:
            Dict with keys:
                direction_logits: [B, 3]
                asymmetry_features: [B, 8] (for analysis)
        """
        # Encode both passages with frozen encoder
        with torch.no_grad():
            h_a = self.encoder(input_ids=input_ids_a, attention_mask=attention_mask_a)
            h_a = h_a.last_hidden_state  # [B, L_A, 768]
            h_b = self.encoder(input_ids=input_ids_b, attention_mask=attention_mask_b)
            h_b = h_b.last_hidden_state  # [B, L_B, 768]

        # Detach so encoder doesn't receive gradient
        h_a_detached = h_a.detach()
        h_b_detached = h_b.detach()

        # Cross-attention asymmetry
        asym_features, pooled_a, pooled_b = self.cross_attn(
            h_a_detached, h_b_detached, attention_mask_a.bool(), attention_mask_b.bool(),
        )

        # Direction classification
        combined = torch.cat([pooled_a, pooled_b, asym_features], dim=1)  # [B, 1544]
        direction_logits = self.classifier(combined)

        return {
            "direction_logits": direction_logits,
            "asymmetry_features": asym_features,
        }

    def predict(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Inference mode — predict direction for a batch of pairs.

        Returns list of dicts with probabilities and predicted direction.
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
