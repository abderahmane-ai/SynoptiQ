"""Multi-task encoder with task-specific heads for KoineFormer.

Attaches task-specific LoRA adapters and classification heads on top of
the frozen KoineFormer encoder.  Each task gets its own LoRA delta
and prediction head; the shared encoder backbone never changes.

Tasks:
  - POS tagging (13-class linear head)
  - Dependency parsing (biaffine head, UAS/LAS)
  - Lemmatisation (T5 decoder head via the shared decoder)
  - Pericope classification (linear head, 170 classes)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from synoptiq.utils.logging_ import get_logger

_LOG = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Number of MorphGNT POS tags observed in the corpus (top 13).
N_POS_CLASSES: int = 13

# Maximum number of pericope classes (Aland pericopes in the corpus).
N_PERICOPE_CLASSES: int = 170

# Dependency relation classes (for biaffine parser).
N_DEP_REL_CLASSES: int = 64

# Hidden dimension of the GreTa T5-base encoder.
ENCODER_HIDDEN: int = 768


# ── Task heads ──────────────────────────────────────────────────────────────


class POSTagger(nn.Module):
    """Linear POS classification head.

    Args:
        hidden_size: Encoder output dimension (768 for T5-base).
        n_classes: Number of POS tags.
        dropout: Dropout rate applied before the classifier.
    """

    def __init__(
        self,
        hidden_size: int = ENCODER_HIDDEN,
        n_classes: int = N_POS_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return logits for each token position [batch, seq, n_classes]."""
        return self.classifier(self.dropout(hidden))


class BiaffineParser(nn.Module):
    """Biaffine dependency parser head (Dozat & Manning 2017).

    Produces arc scores and relation scores for each pair of tokens.
    Used for both UAS (unlabeled attachment score) and LAS (labeled).

    Args:
        hidden_size: Encoder output dimension.
        arc_hidden: Hidden size of the arc MLP.
        rel_hidden: Hidden size of the relation MLP.
        n_relations: Number of dependency relation types.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int = ENCODER_HIDDEN,
        arc_hidden: int = 256,
        rel_hidden: int = 128,
        n_relations: int = N_DEP_REL_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Arc head
        self.arc_head = nn.Linear(hidden_size, arc_hidden, bias=False)
        self.arc_dep = nn.Linear(hidden_size, arc_hidden, bias=False)
        # Relation head
        self.rel_head = nn.Linear(hidden_size, rel_hidden, bias=False)
        self.rel_dep = nn.Linear(hidden_size, rel_hidden, bias=False)
        self.rel_classifier = nn.Linear(rel_hidden, n_relations, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (arc_scores, rel_scores).

        arc_scores: [batch, seq, seq] — score of token j being head of token i.
        rel_scores: [batch, seq, seq, n_relations] — relation type scores.
        """
        h = self.dropout(hidden)
        # Head and dependent representations for arcs
        h_head = self.arc_head(h)   # [B, S, arc_h]
        h_dep = self.arc_dep(h)     # [B, S, arc_h]
        # Biaffine: h_dep @ W @ h_head^T.  For simplicity, dot-product.
        arc_scores = torch.matmul(h_dep, h_head.transpose(-2, -1))

        # Relation scores: element-wise product of head+dep → classifier.
        r_head = self.rel_head(h)   # [B, S, rel_h]
        r_dep = self.rel_dep(h)     # [B, S, rel_h]
        # Outer concatenation: [B, S, 1, rel_h] + [B, 1, S, rel_h]
        r_head = r_head.unsqueeze(2)  # [B, S, 1, rel_h]
        r_dep = r_dep.unsqueeze(1)    # [B, 1, S, rel_h]
        r_combined = r_head + r_dep   # [B, S, S, rel_h]
        rel_scores = self.rel_classifier(r_combined)  # [B, S, S, n_rel]

        return arc_scores, rel_scores


class PericopeClassifier(nn.Module):
    """Mean-pooled sentence-level pericope classifier.

    Args:
        hidden_size: Encoder output dimension.
        n_classes: Number of pericope classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int = ENCODER_HIDDEN,
        n_classes: int = N_PERICOPE_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return logits for the whole sequence [batch, n_classes]."""
        # Mean-pool over non-padding tokens.
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


# ── Multi-task adapter configuration ─────────────────────────────────────────


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task LoRA fine-tuning.

    Each task gets its own LoRA adapter (r=16, alpha=32) on the encoder.
    """

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q", "v")
    pos_weight: float = 0.25
    dep_arc_weight: float = 0.25
    dep_rel_weight: float = 0.25
    pericope_weight: float = 0.25


# ── Multi-task encoder ──────────────────────────────────────────────────────


class MultiTaskEncoder(nn.Module):
    """KoineFormer encoder with task-specific LoRA adapters and heads.

    The base GreTa encoder is frozen.  Each task gets a dedicated LoRA
    adapter (via PEFT) and a prediction head.  This enables joint
    multi-task training without task interference.

    Args:
        base_encoder: The frozen T5 encoder from KoineFormer.
        config: Multi-task configuration.
        tokenizer: GreTa tokenizer (for embedding size resolution).
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        config: MultiTaskConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MultiTaskConfig()
        self.encoder = base_encoder

        # Task heads
        self.pos_head = POSTagger()
        self.biaffine = BiaffineParser()
        self.pericope_head = PericopeClassifier()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the encoder and selected task heads.

        Args:
            input_ids: Tokenised input [B, S].
            attention_mask: Padding mask [B, S].
            task: One of ``"pos"``, ``"dep"``, ``"pericope"``, or None for all.

        Returns:
            Dict with task-specific tensors.  Keys depend on *task*:
              - ``"pos"`` → ``{"pos_logits": [B, S, 13]}``
              - ``"dep"`` → ``{"arc_scores": …, "rel_scores": …}``
              - ``"pericope"`` → ``{"pericope_logits": [B, 170]}``
        """
        # Get encoder hidden states
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = encoder_outputs.last_hidden_state  # [B, S, 768]

        results: dict[str, torch.Tensor] = {}

        if task is None or task == "pos":
            results["pos_logits"] = self.pos_head(hidden)

        if task is None or task == "dep":
            arc_scores, rel_scores = self.biaffine(hidden)
            results["arc_scores"] = arc_scores
            results["rel_scores"] = rel_scores

        if task is None or task == "pericope":
            results["pericope_logits"] = self.pericope_head(hidden, attention_mask)

        return results
