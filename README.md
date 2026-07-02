# SynoptiQ

A multi-task neural source criticism framework for the Synoptic Problem.

Applies transformers, causal direction modeling, and Bayesian inference
to determine the literary relationships among the Gospels of Matthew, Mark, and Luke.

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
