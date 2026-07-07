# SynoptiQ

A neural source-criticism framework for the Synoptic Problem. Applies transformers
(KoineFormer, a DAPT'd Koine-Greek T5) to a curated parallel corpus of the Gospels of
Matthew, Mark, and Luke — representation learning (Paper A) and Q reconstruction.

> Note: copying-*direction* detection was investigated and closed as a negative result
> (see `docs/DIRECTION_NEGATIVE_RESULT.md`); it is not part of the current codebase.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Build the corpus (download + parse + align + cache)
python scripts/prepare_data.py

# Validate existing corpus
python scripts/prepare_data.py --validate
```

## Development

```bash
make check   # ruff + mypy + pytest
```
