"""Teacher-forced NLL scoring and the latent-source importance estimator.

Every source-identification verdict is a paired difference of the negative
log-likelihood of the *same* target sequence (Luke, or Mark for the gates) under two
different encoder contexts. Keeping length and target style common to both branches is
what makes the comparison immune to the length/style artifacts that sank the earlier
direction work (threat T1). Used by the redactor/FiD models in ``synoptiq.models``.

This module provides:

* :func:`per_token_nll` — masked per-token NLL from decoder logits and gold labels;
* :func:`sequence_nll` — per-sequence sum or per-token mean;
* :func:`log_mean_exp` — numerically stable log-mean-exp;
* :func:`bottleneck_nll` — the 2SH marginal ``−log (1/K) Σ_k p(Lk | Q_k)`` estimated
  from K reconstructed-source samples, i.e. the importance-sampling estimate of the
  latent-source channel used in E1;
* :func:`aggregate_pericope_nll` — collapse per-token NLLs to one per-pericope value.

The torch-facing pieces work on any logits/labels tensors, so they unit-test on tiny
random tensors with no model or training. The math pieces are plain floats/tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Sentinel for "ignore this position" in labels (matches HF convention).
IGNORE_INDEX = -100


def per_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Per-token negative log-likelihood of ``labels`` under ``logits``.

    Args:
        logits: Decoder logits, shape [..., seq_len, vocab].
        labels: Gold token ids, shape [..., seq_len]; positions equal to
            ``ignore_index`` (padding, prompt) are masked to 0 contribution.
        ignore_index: Label value marking positions to skip.

    Returns:
        Tensor of shape [..., seq_len] with the per-token NLL in nats; masked
        positions are exactly 0.0.
    """
    if logits.shape[:-1] != labels.shape:
        msg = f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} are misaligned"
        raise ValueError(msg)
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clamp_min(0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = labels != ignore_index
    return torch.where(mask, -gathered, torch.zeros_like(gathered))


def sequence_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    """Aggregate per-token NLL to per-sequence.

    Args:
        logits: Decoder logits, shape [batch, seq_len, vocab].
        labels: Gold ids, shape [batch, seq_len].
        ignore_index: Positions to skip.
        reduction: ``"mean"`` (nats per non-masked token — the verdict scale) or
            ``"sum"`` (total nats).

    Returns:
        Tensor of shape [batch].
    """
    if reduction not in {"mean", "sum"}:
        msg = f"reduction must be 'mean' or 'sum', got {reduction!r}"
        raise ValueError(msg)
    tok_nll = per_token_nll(logits, labels, ignore_index=ignore_index)
    totals = tok_nll.sum(dim=-1)
    if reduction == "sum":
        return totals
    counts = (labels != ignore_index).sum(dim=-1).clamp_min(1)
    return totals / counts


def log_mean_exp(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable ``log( mean( exp(values) ) )`` along ``dim``."""
    n = values.shape[dim]
    return torch.logsumexp(values, dim=dim) - torch.log(
        torch.tensor(float(n), dtype=values.dtype, device=values.device)
    )


def bottleneck_nll(sample_nll: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Marginal NLL through a latent source, from K per-sample NLLs (2SH branch).

    Given ``sample_nll[k] = −log p(Lk | Q_k)`` for K reconstructed sources
    ``Q_k ~ G_Mt(Mt)``, the importance estimate of the marginal channel is

        −log ( (1/K) Σ_k p(Lk | Q_k) )  =  −log_mean_exp( −sample_nll ).

    This is the E1 bottleneck likelihood that competes against the direct-channel
    likelihood ``−log p(Lk | Mt)``. Using log-mean-exp (not a mean of NLLs) is
    essential: the mixture is over probabilities, not over log-probabilities.

    Args:
        sample_nll: Per-sample NLLs, K along ``dim``.
        dim: The sample axis to marginalise.

    Returns:
        Marginal NLL with the sample axis reduced.
    """
    return -log_mean_exp(-sample_nll, dim=dim)


def aggregate_pericope_nll(token_nll: torch.Tensor, token_mask: torch.Tensor) -> float:
    """Mean per-token NLL over the unmasked tokens of one pericope.

    Args:
        token_nll: Per-token NLLs (any shape), already 0 on masked positions.
        token_mask: Boolean/float mask, same shape, 1 where a real target token is.

    Returns:
        Scalar nats-per-token for the pericope (0.0 if no unmasked tokens).
    """
    denom = token_mask.sum()
    if float(denom) == 0.0:
        return 0.0
    return float(token_nll.sum() / denom)
