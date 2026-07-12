"""Build the Koine-T5-Hexapla (MAX) training corpus artifact.

Ingests every raw Koine/Classical source on disk — the Rahlfs-1935 LXX via a Text-Fabric
reader (623,693 words, previously unusable), plus first1k / apostolic / sblgnt via the
proven TEI/txt extractor — chunks them into passage windows, decontaminates against the
held-out Gospel test/val splits, deduplicates, and emits a unified corpus with held-out
eval splits and a source/license census.

This is the generative fuel the current Koine-T5 lacks: its denoise pool sees only
~263K tokens (Gospel + PROIEL); this artifact adds ~1M+ words of coherent Koine prose plus
a continuation (prefix-LM) task, without touching the analysis-task data.

Outputs (under --out, default data/processed/koine_maxi/):
  corpus.jsonl            — training rows: {"task": "denoise"|"continuation", ...}
  eval_perplexity.jsonl   — held-out raw windows (never trained on) for perplexity
  eval_continuation.jsonl — held-out prefix→continuation pairs for generation scoring
  manifest.json           — counts, token totals, per-source census, decontam stats, config

Usage:
    python scripts/prepare_koine_maxi_corpus.py                # build from data/raw
    python scripts/prepare_koine_maxi_corpus.py --validate     # summarize an existing artifact
    python scripts/prepare_koine_maxi_corpus.py --upload       # upload to the Modal data volume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

# Ensure the project root is importable when run from anywhere.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synoptiq.data.koine_corpus import (  # noqa: E402
    Passage,
    build_contamination_index,
    build_continuation_examples,
    build_raw_passages,
    chunk_passages,
    dedup_passages,
    is_contaminated,
)
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# Modal data volume + remote path for the prepared corpus (read by app_koine_hexapla.py).
DATA_VOLUME = "synoptiq-data"
REMOTE_DIR = "/koine_maxi"


def _gospel_forbidden_texts(processed_dir: Path) -> list[str]:
    """Reconstruct held-out Gospel (test+val) pericope texts for the decontamination index.

    Returns one string per (pericope, book) so no held-out Gospel wording leaks into the
    generative pools. Empty if the processed corpus is unavailable (a warning is logged).
    """
    tokens = processed_dir / "tokens.parquet"
    pericopes = processed_dir / "pericopes.parquet"
    if not (tokens.exists() and pericopes.exists()):
        _LOG.warning("processed corpus not found — decontamination index will be EMPTY")
        return []

    from synoptiq.data.corpus import Corpus

    corpus = Corpus.from_parquet(
        tokens, pericopes, splits_path=processed_dir / "splits.json"
    )
    texts: list[str] = []
    for split in ("test", "val"):
        for pericope in corpus.iter_pericopes(split=split):
            for book_tokens in pericope["tokens"].values():
                text = " ".join(str(t.get("text", "")) for t in book_tokens)
                if text.strip():
                    texts.append(text)
    return texts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _census(windows: list[Passage]) -> dict[str, dict[str, int]]:
    """Per-source window + word counts."""
    out: dict[str, dict[str, int]] = {}
    for w in windows:
        entry = out.setdefault(w.source, {"windows": 0, "words": 0, "register": w.register})
        entry["windows"] += 1
        entry["words"] += len(w.text.split())
    return out


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_raw = Path(args.data_raw)
    rng = random.Random(args.seed)

    _LOG.info("building raw passages", extra={"data_raw": str(data_raw)})
    raw = list(build_raw_passages(data_raw))
    windows_all = list(
        chunk_passages(raw, target_words=args.target_words, max_words=args.max_words)
    )
    n_raw_windows = len(windows_all)

    _LOG.info("decontaminating against held-out Gospel splits")
    forbidden = _gospel_forbidden_texts(Path(args.processed))
    index = build_contamination_index(forbidden, n=args.shingle_n)
    n_contaminated = sum(1 for w in windows_all if is_contaminated(w.text, index, args.shingle_n))

    windows = list(dedup_passages(windows_all, contamination_index=index, n=args.shingle_n))
    n_dropped = n_raw_windows - len(windows)
    _LOG.info(
        "windows after dedup+decontam",
        extra={"raw": n_raw_windows, "kept": len(windows), "dropped": n_dropped,
               "contaminated_screened": n_contaminated},
    )

    # Deterministic train/eval split on windows (eval windows are NEVER trained on).
    rng.shuffle(windows)
    n_eval = max(1, int(len(windows) * args.eval_frac)) if windows else 0
    eval_windows = windows[:n_eval]
    train_windows = windows[n_eval:]

    # Training rows: denoise (raw text; corruption is applied online in the trainer) +
    # continuation (prefix-LM) examples, both from the TRAIN windows only.
    denoise_rows = [
        {"task": "denoise", "raw_text": w.text, "source": w.source, "register": w.register}
        for w in train_windows
    ]
    continuation_rows = list(
        build_continuation_examples(train_windows, min_words=args.min_cont_words)
    )
    corpus_rows = denoise_rows + continuation_rows

    # Held-out eval rows.
    perplexity_rows = [
        {"text": w.text, "source": w.source, "ref": w.ref} for w in eval_windows
    ]
    continuation_eval_rows = list(
        build_continuation_examples(eval_windows, min_words=args.min_cont_words)
    )

    _write_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    _write_jsonl(out_dir / "eval_perplexity.jsonl", perplexity_rows)
    _write_jsonl(out_dir / "eval_continuation.jsonl", continuation_eval_rows)

    manifest = {
        "config": {
            "target_words": args.target_words,
            "max_words": args.max_words,
            "eval_frac": args.eval_frac,
            "shingle_n": args.shingle_n,
            "min_cont_words": args.min_cont_words,
            "seed": args.seed,
        },
        "totals": {
            "raw_windows": n_raw_windows,
            "clean_windows": len(windows),
            "dropped_dup_or_contaminated": n_dropped,
            "contaminated_screened": n_contaminated,
            "train_windows": len(train_windows),
            "eval_windows": len(eval_windows),
            "corpus_words": sum(len(w.text.split()) for w in train_windows),
        },
        "rows": {
            "denoise": len(denoise_rows),
            "continuation": len(continuation_rows),
            "eval_perplexity": len(perplexity_rows),
            "eval_continuation": len(continuation_eval_rows),
        },
        "census_by_source": _census(windows),
        "decontamination": {
            "forbidden_texts": len(forbidden),
            "gospel_splits_screened": ["test", "val"],
        },
        "licenses": {
            "lxx": "CC BY-NC-SA (CATSS/Rahlfs via eliranwong TF)",
            "first1k": "CC BY-SA 4.0 (Open Greek & Latin)",
            "apostolic": "CC BY-SA (First1KGreek)",
            "sblgnt": "CC BY 4.0 (synoptic gospels held out)",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n── Koine-MAXI corpus built ─────────────────────────────")
    print(f"  out: {out_dir}")
    print(f"  clean windows: {len(windows):,}  (train {len(train_windows):,} / "
          f"eval {len(eval_windows):,})  | screened {n_dropped:,} dup/contaminated")
    print(f"  training rows: denoise {len(denoise_rows):,} + continuation "
          f"{len(continuation_rows):,} = {len(corpus_rows):,}")
    print(f"  corpus words (train): {manifest['totals']['corpus_words']:,}")
    for src, c in sorted(manifest["census_by_source"].items()):
        print(f"    {src:10s} {c['windows']:>6,} windows  "
              f"{c['words']:>9,} words  ({c['register']})")


def validate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        print(f"No manifest at {manifest} — run without --validate first.")
        sys.exit(1)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps(data["totals"], indent=2))
    print("census:", json.dumps(data["census_by_source"], indent=2, ensure_ascii=False))


def upload(args: argparse.Namespace) -> None:
    import subprocess

    out_dir = Path(args.out)
    if not (out_dir / "corpus.jsonl").exists():
        print(f"No corpus at {out_dir} — build it first.")
        sys.exit(1)
    print(f"Uploading {out_dir} → {DATA_VOLUME}:{REMOTE_DIR} (--force) ...")
    # Run without capturing output so Modal's native progress bar streams directly to the terminal
    result = subprocess.run(
        ["modal", "volume", "put", "--force", DATA_VOLUME, str(out_dir), REMOTE_DIR],
        check=False,
    )
    if result.returncode != 0:
        print("Upload failed. Verify Modal is logged in and the volume is not locked.")
        sys.exit(1)
    print(f"Uploaded to {DATA_VOLUME}:{REMOTE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-raw", default="data/raw", help="raw corpora root")
    parser.add_argument("--processed", default="data/processed",
                        help="processed corpus dir (for the decontamination index)")
    parser.add_argument("--out", default="data/processed/koine_maxi", help="output dir")
    parser.add_argument("--target-words", type=int, default=150)
    parser.add_argument("--max-words", type=int, default=300)
    parser.add_argument("--eval-frac", type=float, default=0.02)
    parser.add_argument("--shingle-n", type=int, default=8)
    parser.add_argument("--min-cont-words", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="summarize an existing artifact")
    parser.add_argument("--upload", action="store_true", help="upload to the Modal data volume")
    args = parser.parse_args()

    if args.validate:
        validate(args)
    elif args.upload:
        upload(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
