"""Measure how much a DAPT checkpoint memorized the evaluation gospels (Phase 5, M1).

Scores gospel text vs. non-gospel Koine control under the same T5 span-corruption objective
DAPT trained on, and (optionally) probes verbatim recall by verse completion. With one set of
adapters it runs the preliminary single-model audit; with ``--compare-adapters ORIG NS`` it
computes the definitive difference-in-differences memorization gap between the original
KoineFormer and the decontaminated KoineFormer-NS.

The statistics and verdict live in ``synoptiq.evaluation.contamination`` (pure, unit-tested);
this script only does the model scoring.

Usage:
    # preliminary, original adapters only
    python scripts/audit_contamination.py --adapters models/koineformer/dapt/final

    # definitive paired audit (after M1 training produces the NS adapters)
    python scripts/audit_contamination.py \
        --compare-adapters models/koineformer/dapt/final outputs/dapt_ns/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from scripts._cli_utils import detect_device  # noqa: E402
from synoptiq.data.corpus import Corpus  # noqa: E402
from synoptiq.evaluation.contamination import (  # noqa: E402
    ContaminationReport,
    GroupScore,
    exact_match_rate,
    score_group,
)
from synoptiq.models.koineformer import GRETA_MODEL_ID, KoineFormer  # noqa: E402
from synoptiq.training.dapt import _extract_text_from_dir, sblgnt_stems_for_books  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)
_SYNOPTICS = ("Matthew", "Mark", "Luke")


# ── Probe construction ────────────────────────────────────────────────────────


def _verse_strings(corpus: Corpus, book: str) -> list[str]:
    """Reconstruct per-verse Greek strings for a book, in canonical order."""
    by_verse: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for tok in corpus.get_tokens(book, exclude_punctuation=False):  # type: ignore[arg-type]
        key = (int(tok["chapter"]), int(tok["verse"]))
        by_verse.setdefault(key, []).append((int(tok["position"]), str(tok["text"])))
    verses = []
    for key in sorted(by_verse):
        words = [t for _, t in sorted(by_verse[key])]
        verses.append(" ".join(words))
    return verses


def gospel_chunks(corpus: Corpus, *, max_per_book: int) -> list[str]:
    """Gospel probe text: the study's own evaluation verses (Mt/Mk/Lk)."""
    chunks: list[str] = []
    for book in _SYNOPTICS:
        chunks.extend(_verse_strings(corpus, book)[:max_per_book])
    return chunks


def control_chunks(data_dir: Path, *, n: int) -> list[str]:
    """Non-gospel Koine control: SBLGNT with the synoptic gospels held out."""
    stems = sblgnt_stems_for_books(_SYNOPTICS)
    out: list[str] = []
    for chunk in _extract_text_from_dir(data_dir, "sblgnt", exclude_stems=stems):
        out.append(chunk)
        if len(out) >= n:
            break
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────


def _sentinel_id(tokenizer: AutoTokenizer) -> int:
    return getattr(tokenizer, "mask_token_id", None) or 4


def score_chunks(
    model: KoineFormer,
    tokenizer: AutoTokenizer,
    chunks: list[str],
    *,
    device: str,
    noise_density: float = 0.15,
    max_length: int = 512,
    seed: int = 0,
) -> tuple[list[float], list[int]]:
    """Per-chunk mean NLL and token count under DAPT's span-corruption objective.

    Masking positions are drawn from a fixed seed so the two models being compared see
    identical corruptions — the comparison is then purely about the model, not the noise.
    """
    model.eval()
    sentinel = _sentinel_id(tokenizer)
    nlls: list[float] = []
    tokens: list[int] = []
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for text in chunks:
            ids = tokenizer.encode(
                text, add_special_tokens=False, max_length=max_length, truncation=True
            )
            if len(ids) < 8:
                continue
            input_ids = torch.tensor(ids, dtype=torch.long)
            labels = input_ids.clone()
            n_noise = max(1, int(len(ids) * noise_density))
            noise_idx = torch.randperm(len(ids), generator=gen)[:n_noise]
            input_ids[noise_idx] = sentinel
            out = model.forward(
                input_ids.unsqueeze(0).to(device),
                torch.ones_like(input_ids).unsqueeze(0).to(device),
                labels=labels.unsqueeze(0).to(device),
            )
            nlls.append(float(out["loss"]))
            tokens.append(len(ids))
    return nlls, tokens


def verse_completion_em(
    model: KoineFormer,
    tokenizer: AutoTokenizer,
    verses: list[str],
    *,
    device: str,
    max_new_tokens: int = 40,
) -> float:
    """Exact-match rate of second-half completions under T5 sentinel infill.

    Feeds ``<first half> <extra_id_0>`` and checks whether the model regenerates the exact
    second half — verbatim recall is the clearest fingerprint of memorization.
    """
    sentinel_tok = "<extra_id_0>"
    preds: list[str] = []
    golds: list[str] = []
    for verse in verses:
        words = verse.split()
        if len(words) < 6:
            continue
        half = len(words) // 2
        prompt = " ".join(words[:half]) + f" {sentinel_tok}"
        gold = " ".join(words[half:])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        gen_ids = model.generate(
            enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        preds.append(text)
        golds.append(gold)
    return exact_match_rate(preds, golds)


def _audit_one(
    adapters: Path,
    corpus: Corpus,
    data_dir: Path,
    args: argparse.Namespace,
    device: str,
    tokenizer: AutoTokenizer,
) -> tuple[GroupScore, GroupScore, float]:
    """Load one checkpoint and return (gospel score, control score, gospel exact-match)."""
    _LOG.info("loading adapters", extra={"path": str(adapters)})
    model = KoineFormer.from_pretrained(device=device)
    model.load_adapters(adapters)

    g_chunks = gospel_chunks(corpus, max_per_book=args.max_per_book)
    c_chunks = control_chunks(data_dir, n=len(g_chunks))
    g_nll, g_tok = score_chunks(model, tokenizer, g_chunks, device=device, seed=args.seed)
    c_nll, c_tok = score_chunks(model, tokenizer, c_chunks, device=device, seed=args.seed)

    em = verse_completion_em(
        model, tokenizer, _verse_strings(corpus, "Mark")[: args.max_per_book], device=device
    )
    return score_group("gospel", g_nll, g_tok), score_group("control", c_nll, c_tok), em


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--adapters", type=Path, help="single checkpoint (preliminary audit)")
    p.add_argument("--compare-adapters", type=Path, nargs=2, metavar=("ORIG", "NS"),
                   help="original and decontaminated checkpoints (definitive paired audit)")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--tokens", type=Path, default=Path("data/processed/tokens.parquet"))
    p.add_argument("--pericopes", type=Path, default=Path("data/processed/pericopes.parquet"))
    p.add_argument("--max-per-book", type=int, default=200, help="verses per gospel to probe")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/study"))
    return p


def _resolve_adapters(path: Path) -> Path | None:
    """Find the dir holding ``adapter_model.safetensors`` at or under ``path``.

    Tolerant of ``modal volume get`` nesting the download a level deeper than
    requested — searches recursively and prefers a ``final`` checkpoint.
    """
    if (path / "adapter_model.safetensors").exists():
        return path
    if not path.exists():
        return None
    hits = sorted(path.rglob("adapter_model.safetensors"))
    if not hits:
        return None
    for h in hits:
        if h.parent.name == "final":
            return h.parent
    return hits[0].parent


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.adapters and not args.compare_adapters:
        _LOG.error("provide --adapters or --compare-adapters")
        return 2

    raw = list(args.compare_adapters) if args.compare_adapters else [args.adapters]
    resolved: list[Path] = []
    for path in raw:
        found = _resolve_adapters(Path(path))
        if found is None:
            _LOG.error("adapter checkpoint not found — did the download run?",
                       extra={"path": str(path)})
            print(
                f"\nNo adapter_model.safetensors under {path}. Download KoineFormer-NS:\n"
                "  rm -rf models/koineformer_ns && modal volume get synoptiq-outputs "
                "dapt_ns/final models/koineformer_ns/final"
            )
            return 2
        if found != Path(path):
            _LOG.info("resolved adapters to nested dir", extra={"path": str(found)})
        resolved.append(found)

    args.out.mkdir(parents=True, exist_ok=True)

    device = args.device or detect_device()
    corpus = Corpus.from_parquet(args.tokens, args.pericopes)
    tokenizer = AutoTokenizer.from_pretrained(GRETA_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    if args.compare_adapters:
        orig_path, ns_path = resolved
        og, oc, oem = _audit_one(orig_path, corpus, args.data_dir, args, device, tokenizer)
        ng, nc, _ = _audit_one(ns_path, corpus, args.data_dir, args, device, tokenizer)
        report = ContaminationReport(
            orig_gospel=og, orig_control=oc, ns_gospel=ng, ns_control=nc,
            exact_match_gospel=oem, exact_match_control=None,
        )
    else:
        og, oc, oem = _audit_one(resolved[0], corpus, args.data_dir, args, device, tokenizer)
        report = ContaminationReport(
            orig_gospel=og, orig_control=oc, exact_match_gospel=oem, exact_match_control=None
        )

    (args.out / "audit.md").write_text(report.to_markdown(), encoding="utf-8")
    (args.out / "audit.json").write_text(json.dumps(report.to_dict(), indent=2))
    print("\n" + report.to_markdown())
    print(f"\nwrote: {args.out}/audit.md, audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
