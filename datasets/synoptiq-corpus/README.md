---
license: cc-by-sa-4.0
language:
  - grc
pretty_name: SynoptiQ Corpus
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
  - token-classification
  - fill-mask
tags:
  - ancient-greek
  - koine-greek
  - new-testament
  - synoptic-problem
  - biblical-studies
  - digital-humanities
  - text-alignment
  - morphology
  - lemmatization
annotations_creators:
  - expert-annotated
---

# SynoptiQ Corpus

A token-level dataset of the Synoptic Gospels (Matthew, Mark, Luke) in
Koine Greek, annotated with morphological tags, Aland pericope
boundaries, and pairwise Needleman-Wunsch token alignments for
computational research on the Synoptic Problem.

## Overview

The Synoptic Problem --- the question of how Matthew, Mark, and Luke are
literarily related --- has occupied New Testament scholarship for over two
centuries. This dataset provides a machine-learning-ready representation
of the synoptic gospels designed for training transformer-based models to
detect copying direction, editorial tendencies, and source relationships.

Each row is a single token with its lemma, part-of-speech tag,
morphological parsing, and Aland pericope assignment. Parallel passages
across gospels are aligned at the token level using Needleman-Wunsch
global alignment keyed on (lemma, POS) pairs, enabling models to learn
from the correspondence structure directly.

### Key properties

- **49,061 tokens** across Matthew (18,329), Mark (11,286), and Luke
  (19,446)
- **170 Aland pericopes** with tradition classification: 88 triple
  (all three synoptics), 17 double (Matthew--Luke only), 18 Matthean-only,
  45 Lukan-only, 2 Markan-only
- **235 pairwise token alignments** (Matthew-Mark, Matthew-Luke,
  Mark-Luke) computed per shared pericope, yielding 8,855 aligned token
  pairs and 45,029 gap positions
- **2,739 unique lemmas** with MorphGNT morphological annotations
  (part-of-speech, tense, voice, mood, case, number, gender)
- **Stratified 60/20/20 split** at the pericope level: 101 train, 33
  validation, 36 test. No verse appears in more than one split
- **SBLGNT text**, the standard critical edition published by the Society
  of Biblical Literature (CC-BY), merged with MorphGNT annotations
  (CC-BY-SA)

### Data sources

| Source | Description | License |
|--------|-------------|---------|
| [SBLGNT](https://github.com/Faithlife/SBLGNT) | Society of Biblical Literature Greek New Testament (Holmes 2010) | CC-BY |
| [MorphGNT](https://github.com/morphgnt/sblgnt) | Morphologically annotated SBLGNT (Tauber 2017) | CC-BY-SA |
| Aland *Synopsis Quattuor Evangeliorum* | Pericope numbering and verse mapping (Aland 1963) | Academic reference |

### Limitations

- Covers the synoptic gospels only; John is excluded as it is not
  directly relevant to the Synoptic Problem.
- Pericope assignment covers 96% of tokens; approximately 1,984 tokens
  (4%) fall outside the Aland table and carry an empty `pericope_id`.
- Morphological annotations reflect one scholarly analysis (MorphGNT);
  alternative parsings exist but are not included.
- The Needleman-Wunsch alignment uses lemma+POS matching with fixed gap
  penalties; different scoring parameters may produce slightly different
  alignments.

## Dataset structure

### Main table: token-level data (`data/*.parquet`)

Each row is one token. The dataset is split into `train`, `validation`,
and `test` Parquet files.

| Column | Type | Description |
|--------|------|-------------|
| `token_id` | string | Stable identifier: `Matthew.1.1.0` |
| `book` | string | Canonical gospel name: `Matthew`, `Mark`, `Luke` |
| `chapter` | int32 | Chapter number (1-indexed) |
| `verse` | int32 | Verse number (1-indexed) |
| `position` | int32 | Token position within verse (0-indexed) |
| `text` | string | Surface form as written in SBLGNT: `γενέσεως` |
| `normalized` | string | NFD-normalised, accent-stripped: `γενεσεως` |
| `lemma` | string | Dictionary headword from MorphGNT: `γένεσις` |
| `pos` | string | Part-of-speech tag: `N-` (noun), `V-` (verb), `RA` (article), `C-` (conjunction), `RP` (relative pronoun), `P-` (preposition), `A-` (adjective), `D-` (adverb) |
| `morph` | string | 8-character CCAT morphological tag encoding person, tense, voice, mood, case, number, gender, degree |
| `pericope_id` | string | Aland pericope number: `018`; empty string if unassigned |
| `tradition` | string | `triple`, `double`, `matthean_unique`, `lukan_unique`, `mark_unique` |
| `genre` | string | `narrative`, `discourse`, `wisdom`, `passion`, `other` |
| `books_in_pericope` | string | JSON array: `'["Matthew","Mark","Luke"]'` |
| `is_punctuation` | bool | Always `false` in SBLGNT; present for schema compatibility |

### Auxiliary files

- **`pericopes.parquet`** --- Pericope-level metadata (170 rows):
  pericope_id, tradition, genre, books, n_tokens, n_alignment_pairs,
  split. Useful for stratified sampling and per-pericope analysis
  without joining the token table.
- **`alignments.json`** --- Token-level Needleman-Wunsch alignment pairs,
  keyed by `pericope_id|book_a|book_b`. Each value is a list of
  `[idx_a, idx_b]` pairs where `null` indicates a gap. Indices reference
  the position of the token within the pericope's per-book token
  sequence (sorted by chapter, verse, position).

## Usage

```python
from datasets import load_dataset

# Load the token-level dataset with train/val/test splits
dataset = load_dataset("ainouche-abderahmane/synoptiq-corpus")

# Filter to triple tradition only
triple = dataset["train"].filter(
    lambda x: x["tradition"] == "triple"
)

# Extract all lemmas from a specific book
matthew_lemmas = dataset["train"].filter(
    lambda x: x["book"] == "Matthew"
)["lemma"]
```

For fine-tuning a masked language model on Koine Greek:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("ainouche-abderahmane/synoptiq-corpus")
tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

tokenized = dataset.map(tokenize, batched=True)
```

## Alignment format

The `alignments.json` file uses the key format
`pericope_id|book_a|book_b`:

```json
{
  "018|Matthew|Mark": [
    [0, null],
    [1, 0],
    [2, null],
    [null, 1]
  ]
}
```

- `[0, null]` --- Matthew token 0 has no counterpart in Mark (gap in Mark)
- `[1, 0]`   --- Matthew token 1 aligns with Mark token 0
- `[null, 1]` --- Mark token 1 has no counterpart in Matthew (gap in Matthew)

Indices reference the per-book token sequence for that pericope, ordered
by (chapter, verse, position). Use the token table to join indices back
to surface forms and lemmas.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{synoptiq_corpus,
  title     = {SynoptiQ Corpus: A Token-Aligned Dataset of the Synoptic
               Gospels for Computational Source Criticism},
  author    = {Ainouche, Abderahmane},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus}
}
```

The underlying text and annotations were created by:

- Holmes, M. W. (2010). *The Greek New Testament: SBL Edition*. Society
  of Biblical Literature. CC-BY.
- Tauber, J. K. (2017). MorphGNT: Morphologically Parsed Greek New
  Testament. CC-BY-SA.
- Aland, K. (1963). *Synopsis Quattuor Evangeliorum*. Deutsche
  Bibelgesellschaft.

## License

This dataset is licensed under CC-BY-SA 4.0, in accordance with the
share-alike requirement of MorphGNT.

## Version history

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-07 | Initial release. SBLGNT + MorphGNT merge, Aland pericope assignment, Needleman-Wunsch alignments, stratified splits. |
