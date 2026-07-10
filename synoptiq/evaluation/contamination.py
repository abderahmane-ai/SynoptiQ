"""Contamination audit: how much did a DAPT model memorize the evaluation gospels?

The source-criticism study reads a verdict off the model's *surprise* at Luke's wording.
If the model memorized Luke during DAPT (the original KoineFormer trained on the full NT,
gospels included), that surprise is fake. This module quantifies the memorization so the
study can either trust the original adapters or switch to the decontaminated KoineFormer-NS.

Two probes, both computed here as pure functions over model-produced losses (the model
scoring itself lives in ``scripts/audit_contamination.py`` — this file needs no torch):

* **Perplexity gap (definitive).** A difference-in-differences of log-perplexity between
  the two models on gospel vs. control (non-gospel Koine) text::

      gap = (logppl_gospel[NS] − logppl_gospel[orig]) − (logppl_control[NS] − logppl_control[orig])

  A large positive gap means the original is unusually good on the gospels *specifically*
  (memorization); ≈ 0 means the gospels were no more memorized than any other Koine, so the
  contamination never mattered. The DiD cancels the gospels' intrinsic ease and the models'
  overall quality difference.

* **Ease ratio (single-model, preliminary).** ``ppl_control / ppl_gospel`` for one model:
  > 1 means it finds the gospels easier than control, which *could* be memorization or just
  intrinsic simplicity — hence the paired gap is the one that decides.

* **Verse-completion exact match.** Fraction of held-out gospel verse continuations the
  model reproduces verbatim under span-infill. High on gospels but not control = memorization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GroupScore:
    """Aggregate teacher-forced score for one text group (gospel or control)."""

    name: str
    perplexity: float
    mean_token_nll: float  # nats per token
    n_chunks: int
    n_tokens: int

    @property
    def log_perplexity(self) -> float:
        return self.mean_token_nll  # log(exp(mean_nll)) == mean_nll


def score_group(
    name: str,
    chunk_nlls: Sequence[float],
    chunk_tokens: Sequence[int],
) -> GroupScore:
    """Aggregate per-chunk mean NLLs into a token-weighted group perplexity.

    Args:
        name: Group label (e.g. ``"gospel"`` / ``"control"``).
        chunk_nlls: Mean per-token NLL (nats) for each scored chunk.
        chunk_tokens: Number of scored (unmasked-target) tokens in each chunk.

    Returns:
        GroupScore with token-weighted mean NLL and its perplexity.
    """
    if len(chunk_nlls) != len(chunk_tokens):
        msg = f"chunk_nlls ({len(chunk_nlls)}) and chunk_tokens ({len(chunk_tokens)}) differ"
        raise ValueError(msg)
    total_tokens = int(sum(chunk_tokens))
    if total_tokens == 0:
        msg = "cannot score a group with zero tokens"
        raise ValueError(msg)
    total_nats = sum(nll * n for nll, n in zip(chunk_nlls, chunk_tokens, strict=True))
    mean_nll = total_nats / total_tokens
    return GroupScore(
        name=name,
        perplexity=math.exp(mean_nll),
        mean_token_nll=mean_nll,
        n_chunks=len(chunk_nlls),
        n_tokens=total_tokens,
    )


def ease_ratio(gospel: GroupScore, control: GroupScore) -> float:
    """``ppl_control / ppl_gospel`` — how much easier the model finds the gospels."""
    return control.perplexity / gospel.perplexity


def memorization_gap(
    orig_gospel: GroupScore,
    orig_control: GroupScore,
    ns_gospel: GroupScore,
    ns_control: GroupScore,
) -> float:
    """Difference-in-differences of log-perplexity; positive ⇒ original memorized gospels."""
    gospel_delta = ns_gospel.log_perplexity - orig_gospel.log_perplexity
    control_delta = ns_control.log_perplexity - orig_control.log_perplexity
    return gospel_delta - control_delta


def _default_normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def exact_match_rate(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    normalize: Callable[[str], str] | None = None,
) -> float:
    """Fraction of predictions matching their target after normalization."""
    if len(predictions) != len(targets):
        msg = f"predictions ({len(predictions)}) and targets ({len(targets)}) differ"
        raise ValueError(msg)
    if not predictions:
        return 0.0
    norm = normalize or _default_normalize
    hits = sum(norm(p) == norm(t) for p, t in zip(predictions, targets, strict=True))
    return hits / len(predictions)


@dataclass(frozen=True)
class ContaminationReport:
    """Full audit outcome; renders to markdown/JSON and carries the automated flag.

    ``ns_*`` groups are ``None`` for a single-model (preliminary) audit; when present
    the paired ``memorization_gap`` is the authoritative signal.
    """

    orig_gospel: GroupScore
    orig_control: GroupScore
    ns_gospel: GroupScore | None = None
    ns_control: GroupScore | None = None
    exact_match_gospel: float | None = None
    exact_match_control: float | None = None
    gap_threshold: float = 0.25          # log-ppl DiD above this flags memorization
    ease_ratio_threshold: float = 1.5    # single-model heuristic
    exact_match_threshold: float = 0.10  # verbatim recall above this flags memorization

    @property
    def paired(self) -> bool:
        return self.ns_gospel is not None and self.ns_control is not None

    @property
    def gap(self) -> float | None:
        if not self.paired:
            return None
        return memorization_gap(
            self.orig_gospel, self.orig_control, self.ns_gospel, self.ns_control
        )

    @property
    def flagged(self) -> bool:
        """Whether memorization is detected by the strongest available probe."""
        if self.paired:
            return (self.gap or 0.0) > self.gap_threshold
        if self.exact_match_gospel is not None:
            return self.exact_match_gospel > self.exact_match_threshold
        return ease_ratio(self.orig_gospel, self.orig_control) > self.ease_ratio_threshold

    def to_dict(self) -> dict[str, object]:
        def g(s: GroupScore | None) -> dict[str, object] | None:
            if s is None:
                return None
            return {
                "perplexity": s.perplexity,
                "mean_token_nll": s.mean_token_nll,
                "n_chunks": s.n_chunks,
                "n_tokens": s.n_tokens,
            }

        return {
            "paired": self.paired,
            "flagged": self.flagged,
            "memorization_gap": self.gap,
            "ease_ratio_original": ease_ratio(self.orig_gospel, self.orig_control),
            "exact_match_gospel": self.exact_match_gospel,
            "exact_match_control": self.exact_match_control,
            "original": {"gospel": g(self.orig_gospel), "control": g(self.orig_control)},
            "decontaminated": {"gospel": g(self.ns_gospel), "control": g(self.ns_control)},
        }

    def to_markdown(self) -> str:
        lines = ["# Contamination audit", ""]
        verdict = "MEMORIZATION DETECTED" if self.flagged else "no material memorization"
        lines.append(f"**Verdict: {verdict}.**")
        lines.append("")
        lines.append("| model | gospel ppl | control ppl | ease ratio |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| original | {self.orig_gospel.perplexity:.2f} | "
            f"{self.orig_control.perplexity:.2f} | "
            f"{ease_ratio(self.orig_gospel, self.orig_control):.2f} |"
        )
        if self.paired:
            assert self.ns_gospel is not None and self.ns_control is not None
            lines.append(
                f"| KoineFormer-NS | {self.ns_gospel.perplexity:.2f} | "
                f"{self.ns_control.perplexity:.2f} | "
                f"{ease_ratio(self.ns_gospel, self.ns_control):.2f} |"
            )
            lines.append("")
            lines.append(
                f"Memorization gap (log-ppl DiD): **{self.gap:.3f}** "
                f"(threshold {self.gap_threshold}). Positive ⇒ original was unusually good on "
                "the gospels specifically."
            )
        else:
            lines.append("")
            lines.append(
                "_Single-model preliminary audit — run with `--compare-adapters` for the "
                "definitive paired gap._"
            )
        if self.exact_match_gospel is not None:
            control = (
                f"{self.exact_match_control:.1%}"
                if self.exact_match_control is not None
                else "n/a"
            )
            lines.append("")
            lines.append(
                f"Verse-completion exact match — gospel: {self.exact_match_gospel:.1%}, "
                f"control: {control} (threshold {self.exact_match_threshold:.0%})."
            )
        return "\n".join(lines)
