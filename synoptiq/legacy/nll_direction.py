"""Conditional-NLL direction scoring, grounded in MDL causal inference.

Copying direction is a compression fact. Following the algorithmic-independence /
MDL principle for cause and effect (Marx & Vreeken 2017), the pair (A, B) is cheaper
to describe in the *true* copying direction: infer A -> B when

    L(A->B) = NLL(A) + NLL(B|A)   <   NLL(B) + NLL(A|B) = L(B->A)

Using a seq2seq model's negative log-likelihood as the codelength makes this concrete.
Crucially, BOTH factorizations describe the same joint object (all of A and all of B),
so the two total codelengths are **length-fair by construction** — the marginal terms
NLL(A), NLL(B) absorb the "how long/complex is this passage on its own" component that
a bare conditional asymmetry NLL(B|A) - NLL(A|B) leaves confounded with direction.

KoineFormer is a T5 encoder-decoder, so every term is a standard `labels=` forward pass
with no new model code. The marginal NLL(X) is approximated by conditioning on an empty
source (a single pad token), i.e. the decoder's prior codelength for X.

All scores are signed so that **positive => A is the source (A -> B)**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DirectionCodelengths:
    """The four NLL codelengths for one passage pair, in total nats.

    ``nll_*_marg`` are marginal (empty-source) codelengths; ``nll_*_given_*`` are
    conditional. ``n_a`` / ``n_b`` are the target token counts (for per-token forms).
    """

    nll_a_marg: float
    nll_b_marg: float
    nll_b_given_a: float
    nll_a_given_b: float
    n_a: int
    n_b: int

    # ── Score variants (positive => A is the source) ──────────────────────

    @property
    def mdl_score(self) -> float:
        """Length-fair MDL score: L(B->A) - L(A->B). Positive => A is source."""
        l_a_to_b = self.nll_a_marg + self.nll_b_given_a
        l_b_to_a = self.nll_b_marg + self.nll_a_given_b
        return l_b_to_a - l_a_to_b

    @property
    def info_gain_score(self) -> float:
        """Equivalent to mdl_score: IG(A->B) - IG(B->A) (source informs copy more)."""
        ig_a_to_b = self.nll_b_marg - self.nll_b_given_a
        ig_b_to_a = self.nll_a_marg - self.nll_a_given_b
        return ig_a_to_b - ig_b_to_a

    @property
    def conditional_asym_mean(self) -> float:
        """Naive per-token conditional asymmetry: NLL(A|B)/n_a - NLL(B|A)/n_b.

        Signed so positive => A is source, to match the other scores. This is the
        Stage-2 probe quantity (up to sign), kept for the length-confound comparison.
        """
        return self.nll_a_given_b / max(self.n_a, 1) - self.nll_b_given_a / max(self.n_b, 1)

    @property
    def log_len_ratio(self) -> float:
        """log(n_a / n_b) — the length axis we control against."""
        import math
        return math.log(max(self.n_a, 1) / max(self.n_b, 1))


# Feature vector derived purely from the four codelengths. All are per-token (nats)
# except the whole-pair MDL score, and each anti-symmetric feature negates under A<->B
# so a swap flips the predicted direction (the learnable head is fed these directly).
FEATURE_NAMES: tuple[str, ...] = (
    "nll_b_given_a_pt",     # per-token NLL(B|A)
    "nll_a_given_b_pt",     # per-token NLL(A|B)
    "cond_asym_pt",         # NLL(A|B)/n_a - NLL(B|A)/n_b   (+ => A source)
    "marg_a_pt",            # per-token marginal NLL(A) = typicality of A
    "marg_b_pt",            # per-token marginal NLL(B)
    "marg_asym_pt",         # marg_a - marg_b   (+ => A rougher/less typical => A source)
    "infogain_a_to_b_pt",   # (NLL(B) - NLL(B|A))/n_b : how much A informs B
    "infogain_b_to_a_pt",   # (NLL(A) - NLL(A|B))/n_a
    "infogain_asym_pt",     # infogain_a_to_b - infogain_b_to_a
    "mdl_score_norm",       # MDL score / (n_a + n_b)   (+ => A source)
    "log_len_ratio",        # log(n_a / n_b)
)


def codelengths_to_features(cl: DirectionCodelengths) -> np.ndarray:
    """Turn the four codelengths into the FEATURE_NAMES vector (float32)."""
    na, nb = max(cl.n_a, 1), max(cl.n_b, 1)
    marg_a_pt = cl.nll_a_marg / na
    marg_b_pt = cl.nll_b_marg / nb
    ig_a_to_b = (cl.nll_b_marg - cl.nll_b_given_a) / nb
    ig_b_to_a = (cl.nll_a_marg - cl.nll_a_given_b) / na
    return np.array([
        cl.nll_b_given_a / nb,
        cl.nll_a_given_b / na,
        cl.conditional_asym_mean,
        marg_a_pt,
        marg_b_pt,
        marg_a_pt - marg_b_pt,
        ig_a_to_b,
        ig_b_to_a,
        ig_a_to_b - ig_b_to_a,
        cl.mdl_score / (na + nb),
        cl.log_len_ratio,
    ], dtype=np.float32)


def make_empty_source(tokenizer: object, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A minimal encoder input (single pad token) for marginal codelengths."""
    pad_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
    ids = torch.tensor([[pad_id]], device=device)
    mask = torch.ones(1, 1, dtype=torch.long, device=device)
    return ids, mask


@torch.no_grad()
def _total_nll(
    model: torch.nn.Module,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_ids: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> tuple[float, int]:
    """Return (total NLL in nats, n_target_tokens) for generating tgt given src."""
    labels = tgt_ids.clone()
    labels[tgt_mask == 0] = -100
    out = model(input_ids=src_ids, attention_mask=src_mask, labels=labels)
    n = int((labels != -100).sum().item())
    return float(out.loss.item()) * n, n


@torch.no_grad()
def score_pair(
    model: torch.nn.Module,
    empty_source: tuple[torch.Tensor, torch.Tensor],
    ids_a: torch.Tensor,
    mask_a: torch.Tensor,
    ids_b: torch.Tensor,
    mask_b: torch.Tensor,
) -> DirectionCodelengths:
    """Compute all four codelengths for one passage pair (four forward passes).

    Inputs are single-example batches [1, L] on the model's device.
    """
    empty_ids, empty_mask = empty_source
    nll_b_given_a, n_b = _total_nll(model, ids_a, mask_a, ids_b, mask_b)
    nll_a_given_b, n_a = _total_nll(model, ids_b, mask_b, ids_a, mask_a)
    nll_b_marg, _ = _total_nll(model, empty_ids, empty_mask, ids_b, mask_b)
    nll_a_marg, _ = _total_nll(model, empty_ids, empty_mask, ids_a, mask_a)
    return DirectionCodelengths(
        nll_a_marg=nll_a_marg,
        nll_b_marg=nll_b_marg,
        nll_b_given_a=nll_b_given_a,
        nll_a_given_b=nll_a_given_b,
        n_a=n_a,
        n_b=n_b,
    )


class CachedPairScorer:
    """Scores passage pairs while caching conditional and marginal NLLs by passage.

    Swap augmentation and triple-tradition combinatorics make the same passages
    recur across samples; caching roughly halves the forward passes. Keyed on the
    non-padding token ids so identical passages hit the cache regardless of batch.
    """

    def __init__(
        self, model: torch.nn.Module, empty_source: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.model = model
        self.empty = empty_source
        self._cond: dict[tuple[bytes, bytes], tuple[float, int]] = {}
        self._marg: dict[bytes, tuple[float, int]] = {}

    @staticmethod
    def _key(ids: torch.Tensor, mask: torch.Tensor) -> bytes:
        return ids[0][mask[0].bool()].detach().cpu().numpy().tobytes()

    def codelengths(
        self,
        ids_a: torch.Tensor, mask_a: torch.Tensor,
        ids_b: torch.Tensor, mask_b: torch.Tensor,
    ) -> DirectionCodelengths:
        """Cached four-codelength computation for one pair."""
        ka, kb = self._key(ids_a, mask_a), self._key(ids_b, mask_b)

        def cond(si, sm, ti, tm, sk, tk):  # noqa: ANN001
            key = (sk, tk)
            if key not in self._cond:
                self._cond[key] = _total_nll(self.model, si, sm, ti, tm)
            return self._cond[key]

        def marg(ti, tm, tk):  # noqa: ANN001
            if tk not in self._marg:
                self._marg[tk] = _total_nll(self.model, *self.empty, ti, tm)
            return self._marg[tk]

        nll_b_given_a, n_b = cond(ids_a, mask_a, ids_b, mask_b, ka, kb)
        nll_a_given_b, n_a = cond(ids_b, mask_b, ids_a, mask_a, kb, ka)
        nll_b_marg, _ = marg(ids_b, mask_b, kb)
        nll_a_marg, _ = marg(ids_a, mask_a, ka)
        return DirectionCodelengths(
            nll_a_marg=nll_a_marg, nll_b_marg=nll_b_marg,
            nll_b_given_a=nll_b_given_a, nll_a_given_b=nll_a_given_b,
            n_a=n_a, n_b=n_b,
        )
