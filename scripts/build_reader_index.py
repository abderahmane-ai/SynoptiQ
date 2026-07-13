"""Build the compact gold index the Koine reader Space ships.

Serialises the Nestle-1904 GNT (and optionally the Rahlfs LXX) Text-Fabric dataset
into a small gzipped JSON — surface, lemma, part of speech, morphology, gloss,
Strong's per word — that :class:`~synoptiq.reader.index.IndexReader` loads without the
raw multi-file TF tree. The GNT artifact is ~2 MB gzipped.

Usage:
    python scripts/build_reader_index.py                      # GNT → spaces/koine-reader/data
    python scripts/build_reader_index.py --lxx                # also build the LXX
    python scripts/build_reader_index.py --source <dir> --out <file.json.gz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synoptiq.reader import GoldReader, IndexReader, save_index

# Repo-relative defaults.
_N1904 = Path("data/raw/n1904/tf/1.0.0")
_LXX = Path("data/raw/lxx/tf/1935")
_OUT_DIR = Path("spaces/koine-reader/data")


def _build_one(source: Path, out: Path) -> None:
    """Build, write, and smoke-check one index artifact."""
    reader = GoldReader.from_dir(source)
    n_words = sum(
        len(reader.verse(b, c, v))
        for b in reader.books()
        for c in reader.chapters(b)
        for v in reader.verses(b, c)
    )
    path = save_index(reader, out)
    size_kb = path.stat().st_size // 1024
    # Smoke-check: the artifact reloads and resolves a reference identically.
    reloaded = IndexReader.from_file(path)
    probe = "John 1:1" if "John" in reloaded.books() else f"{reloaded.books()[0]} 1:1"
    assert reloaded.read(probe).words, f"artifact failed to resolve {probe!r}"
    print(
        f"✓ {reader.name}: {len(reader.books())} books · {n_words:,} words "
        f"· {len(reloaded.lexicon()):,} lemmas → {path} ({size_kb} KB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_N1904, help="GNT TF directory")
    parser.add_argument(
        "--out", type=Path, default=_OUT_DIR / "n1904_index.json.gz", help="output artifact"
    )
    parser.add_argument("--lxx", action="store_true", help="also build the LXX index")
    parser.add_argument("--lxx-source", type=Path, default=_LXX, help="LXX TF directory")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"GNT source not found: {args.source} (see CLAUDE.md cold-start)")
    _build_one(args.source, args.out)

    if args.lxx:
        if not args.lxx_source.exists():
            raise SystemExit(f"LXX source not found: {args.lxx_source}")
        _build_one(args.lxx_source, _OUT_DIR / "lxx_index.json.gz")


if __name__ == "__main__":
    main()
