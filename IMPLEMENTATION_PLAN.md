# SynoptiQ — Full Implementation Plan

## A Multi-Task Neural Source Criticism Framework for the Synoptic Problem

**Date:** 2026-06-24
**Budget:** $30/month Modal free tier
**Base Model:** GreTa (T5 encoder-decoder, monolingual Ancient Greek) → KoineFormer after DAPT
**Strategy:** Full-scale from day one. All 7 phases. Shared encoder-decoder backbone for all tasks.

---

## 0. Architecture Decisions (FINAL)

### 0.1 Base Model: GreTa

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | **T5 encoder-decoder** (GreTa) | Encoder handles direction/source/editorial. Decoder handles Q reconstruction. Shared encoder = unified representations. |
| Model size | **T5-base** (~220M params) | Fits on T4 (16GB) with gradient accumulation. T5-small too weak for generation quality. |
| Tokenizer | GreTa's AG tokenizer + Koine additions | Already handles polytonic Greek. Add nomina sacra tokens. |
| Pre-training | Monolingual Ancient Greek | No Modern Greek contamination. Classical register, needs DAPT to Koine. |

### 0.2 Why Not the Alternatives

| Alternative | Rejected Because |
|-------------|-----------------|
| Ancient-Greek-BERT | Encoder-only. Cannot generate. Would need separate model for Q reconstruction. Two models to DAPT, two tokenizers, two representation spaces. |
| T5-base from scratch (multilingual) | No Greek pre-training at all. DAPT would take 10x longer. Domain gap from modern multilingual to Koine is larger than from classical AG to Koine. |
| mT5 / ByT5 | Byte-level tokenization ignores Greek morphology. Worse for low-resource domain adaptation. |
| GreBERTa | RoBERTa — encoder-only. Same generation problem as Ancient-Greek-BERT. |

### 0.3 Shared Encoder Strategy

```
                        ┌─────────────────────────┐
                        │     KoineFormer          │
                        │  (GreTa after DAPT)      │
                        │                         │
                        │  ┌───────────────────┐  │
                        │  │     Encoder       │  │
                        │  │  (shared by all)  │  │
                        │  └───────┬───────────┘  │
                        │          │               │
                        │          │ Task-specific │
                        │          │ LoRA adapters │
                        │          │               │
                        │  ┌───────┼───────────┐  │
                        │  │       │           │  │
                        │  ▼       ▼           ▼  │
                        │ Dir   Source      Edit   │
                        │ Scorer Detector   Drift  │
                        │       │                   │
                        │       │ Shared decoder   │
                        │       ▼                   │
                        │  ┌───────────────────┐  │
                        │  │     Decoder       │  │
                        │  │  (FiD Q recon)    │  │
                        │  └───────────────────┘  │
                        └─────────────────────────┘
```

All encoder-based tasks (direction scorer, source detector, editorial drift) share the same frozen DAPT-trained encoder with task-specific LoRA adapters. The decoder is used only for Q reconstruction (FiD) and editorial tendency seq2seq modeling.

### 0.4 GPU Budget Allocation

| Phase | GPU Type | Est. Hours | Est. Cost | Priority |
|-------|----------|------------|-----------|----------|
| 1: Data Pipeline | None | 0 | $0 | BLOCKING |
| 2: KoineFormer DAPT | A10G | 10-14 | ~$15 | BLOCKING |
| 2: Multi-task LoRA | T4 | 4-6 | ~$3 | After DAPT |
| 3: Direction Scorer | T4 | 6-8 | ~$4 | After KoineFormer |
| 4: Editorial Drift | T4 | 4-6 | ~$3 | After Direction |
| 5: Q Reconstruction | A10G | 6-8 | ~$9 | After Editorial |
| 6: Bayesian (CPU) | None | 0 | $0 | After Direction |
| 7: Interpretability | T4 | 2-4 | ~$2 | After all |
| **Total** | | **~32-46** | **~$36** | |

**Budget risk:** $36 exceeds the $30 free tier. Mitigations:
- Use spot instances for ~40% savings (brings total to ~$22)
- Phase 2 DAPT is the biggest cost — if it runs over, defer Phase 5 (Q reconstruction) or run it on T4 with smaller batch size
- Phases 3, 4, and 7 can all run on T4 with gradient accumulation

---

## 1. Project Structure (Complete File Tree)

```
SynoptiQ/
│
├── synoptiq/                              # Python package
│   ├── __init__.py                        # v0.1.0, public API exports
│   ├── _about.py                          # __version__, __author__, __license__
│   │
│   ├── data/                              # ── DATA PIPELINE (Phase 1) ──
│   │   ├── __init__.py
│   │   ├── _download.py                   # Clone/pull SBLGNT, MorphGNT, PROIEL, N1904, LXX, Josephus, Apostolic Fathers
│   │   ├── _parse_sblgnt.py               # Parse SBLGNT XML → token dataframe
│   │   ├── _parse_morphgnt.py             # Parse MorphGNT TSV → morphology dataframe
│   │   ├── _parse_proiel.py               # Parse PROIEL XML → dependency trees
│   │   ├── _parse_n1904.py                # Parse N1904-TF → Aland pericope mapping
│   │   ├── _parse_lxx.py                  # Parse LXX (SWORD module) → Koine control text
│   │   ├── _parse_josephus.py             # Parse PACE Josephus → Koine control text
│   │   ├── _parse_apostolic.py            # Parse Open Apostolic Fathers → Koine control text
│   │   ├── corpus.py                      # Corpus class: unified interface, lazy loading, Parquet caching
│   │   ├── alignment.py                   # Needleman-Wunsch token alignment (lemma + POS scoring)
│   │   ├── pericope.py                    # Aland pericope classification, tradition grouping
│   │   ├── splits.py                      # Stratified train/val/test split (pericope-atomic)
│   │   └── augmentation.py                # Bootstrap, moving windows, scribal noise simulation
│   │
│   ├── models/                            # ── MODEL DEFINITIONS (Phases 2-5) ──
│   │   ├── __init__.py
│   │   ├── _base.py                       # BaseKoineFormer: encoder-decoder wrapper, load/save, freeze/unfreeze
│   │   ├── koineformer.py                 # KoineFormer: GreTa + DAPT + LoRA management
│   │   ├── encoder.py                     # Encoder wrapper: pooling strategies, token classification head
│   │   ├── source_detector.py             # SourceDetector: tradition classifier (double vs triple)
│   │   ├── direction.py                   # DirectionScorer: cross-attention asymmetry → 3-way class
│   │   ├── editor.py                      # EditorialDrift: seq2seq editor + fatigue loss
│   │   └── reconstruction.py              # QReconstructor: FiD architecture, constrained decoding
│   │
│   ├── training/                          # ── TRAINING LOGIC (Phases 2-5) ──
│   │   ├── __init__.py
│   │   ├── _config.py                     # All dataclass configs (DataConfig, ModelConfig, TrainingConfig, etc.)
│   │   ├── _trainer.py                    # Generic Trainer: loop, logging, checkpointing, early stopping
│   │   ├── _datasets.py                   # All Dataset classes (DirectionDataset, EditorDataset, FiDDataset, etc.)
│   │   ├── _collate.py                    # All collate functions (padding, masking, FiD concatenation)
│   │   ├── dapt.py                        # Domain-adaptive pre-training loop (MLM on Koine corpus)
│   │   ├── multitask.py                   # Multi-task LoRA fine-tuning (POS + dep + lemma + pericope)
│   │   ├── direction_train.py             # Direction scorer training on triple tradition
│   │   ├── editor_train.py               # Editorial drift training (seq2seq + fatigue loss)
│   │   └── reconstruction_train.py        # FiD training (Mark reconstruction → Q reconstruction)
│   │
│   ├── evaluation/                        # ── EVALUATION (Phases 3-7) ──
│   │   ├── __init__.py
│   │   ├── metrics.py                     # BLEU, ROUGE-L, chrF++, BERTScore, CER, ECE
│   │   ├── baselines.py                   # Encoplot, length heuristic, cosine sim, Random Forest
│   │   └── calibration.py                 # Reliability diagrams, temperature scaling
│   │
│   ├── bayesian/                          # ── BAYESIAN ANALYSIS (Phase 6, CPU only) ──
│   │   ├── __init__.py
│   │   ├── _hypotheses.py                 # Hypothesis model definitions (2SH, FGH, Augustinian, Griesbach)
│   │   ├── _likelihood.py                 # Hierarchical Beta-binomial likelihood
│   │   ├── models.py                      # PyMC model construction and sampling
│   │   ├── bridge.py                      # Bridge sampling via rpy2 → R bridgesampling
│   │   └── sensitivity.py                 # Prior sensitivity grid + contour plots
│   │
│   ├── interpretability/                  # ── INTERPRETABILITY (Phase 7) ──
│   │   ├── __init__.py
│   │   ├── shap_analysis.py               # SHAP feature importance for direction scorer
│   │   ├── hawkins.py                     # Hawkins 1899 feature comparison
│   │   └── bertviz_viz.py                 # BERTViz attention visualization exports
│   │
│   └── utils/                             # ── SHARED UTILITIES (Phase 0) ──
│       ├── __init__.py
│       ├── greek.py                       # Greek text: accent stripping, normalization, nomina sacra
│       ├── tokenization.py                # Greek-aware tokenizer: GreTa wrapper + Koine token additions
│       ├── logging_.py                    # Structured logging, W&B setup, metric formatting
│       ├── types_.py                      # Type aliases, TypedDicts, Protocols, Literals
│       ├── constants.py                   # Book names, pericope IDs, morphological tag maps
│       └── io_.py                         # Safe file I/O, Parquet helpers, JSON serialization
│
├── scripts/                               # ── ENTRY POINTS ──
│   ├── _cli_utils.py                      # Shared CLI: argparse boilerplate, config loading
│   ├── prepare_data.py                    # Phase 1: Full data pipeline (download → parse → align → split)
│   ├── train_dapt.py                      # Phase 2: KoineFormer DAPT on Modal
│   ├── train_multitask.py                 # Phase 2: Multi-task LoRA fine-tuning
│   ├── train_source_detector.py           # Phase 3: Source detector training
│   ├── train_direction.py                 # Phase 3: Direction scorer training
│   ├── train_editor.py                    # Phase 4: Editorial drift training
│   ├── train_reconstruction.py            # Phase 5: FiD Q reconstruction
│   ├── sample_bayesian.py                 # Phase 6: PyMC model fitting + bridge sampling
│   ├── run_interpretability.py            # Phase 7: SHAP + Hawkins + BERTViz
│   ├── eval_all.py                        # Full evaluation suite across all models
│   └── run_full_pipeline.py               # Orchestrator: runs all phases in sequence
│
├── configs/                               # ── YAML CONFIGURATION ──
│   ├── data.yaml                          # Corpus paths, alignment params, augmentation settings
│   ├── model.yaml                         # GreTa config, LoRA settings, head architectures
│   ├── training.yaml                      # LR schedules, batch sizes, step counts per phase
│   ├── bayesian.yaml                      # MCMC settings, prior hyperparameters, BF grid
│   └── modal.yaml                         # GPU types, timeouts, volume mounts per function
│
├── modal/                                 # ── MODAL DEPLOYMENT ──
│   ├── _common.py                         # Shared: image, volumes, GPU specs, cost estimation
│   ├── app_dapt.py                        # DAPT training app (A10G, long-running)
│   ├── app_train.py                       # General training app (T4, all other tasks)
│   ├── app_eval.py                        # Evaluation app (T4, short-running)
│   └── secrets.toml.example               # Template for W&B + HuggingFace secrets
│
├── tests/                                 # ── TESTS ──
│   ├── conftest.py                        # Fixtures: tiny_corpus, dummy_model, toy_alignment
│   ├── utils/
│   │   ├── test_greek.py
│   │   ├── test_tokenization.py
│   │   └── test_types.py
│   ├── data/
│   │   ├── test_corpus.py
│   │   ├── test_alignment.py
│   │   ├── test_pericope.py
│   │   └── test_splits.py
│   ├── models/
│   │   ├── test_direction.py
│   │   ├── test_editor.py
│   │   └── test_reconstruction.py
│   └── training/
│       ├── test_datasets.py
│       └── test_collate.py
│
├── notebooks/                             # ── EXPLORATORY (not in package) ──
│   ├── 01_corpus_inspection.ipynb         # Explore SBLGNT + MorphGNT data
│   ├── 02_alignment_quality.ipynb         # Inspect token alignment quality
│   ├── 03_direction_probe.ipynb           # Probe direction scorer on key pericopes
│   └── 04_bayesian_prior_elicitation.ipynb # Prior elicitation interface for scholars
│
├── data/                                  # ── GIT-IGNORED ──
│   ├── raw/                               # Downloaded repositories
│   │   ├── sblgnt/
│   │   ├── morphgnt/
│   │   ├── proiel/
│   │   ├── n1904/
│   │   ├── lxx/
│   │   ├── josephus/
│   │   └── apostolic/
│   ├── processed/                         # Aligned, merged datasets
│   │   ├── corpus.parquet
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── external/                          # IQP data, Hawkins features, secondary sources
│       ├── iqp_critical_edition.json
│       └── hawkins_1899_features.json
│
├── models/                                # ── GIT-IGNORED ──
│   ├── greta-base/                        # Cached GreTa base model
│   └── koineformer/                       # Trained KoineFormer checkpoints
│       ├── dapt/
│       ├── multitask/
│       ├── direction/
│       ├── editor/
│       └── reconstruction/
│
├── outputs/                               # ── GIT-IGNORED ──
│   ├── logs/                              # Training logs
│   ├── checkpoints/                       # Intermediate checkpoints
│   ├── predictions/                       # Model outputs
│   ├── bayesian/                          # MCMC traces, BF results
│   └── figures/                           # Generated plots and visualizations
│
├── pyproject.toml                         # Package metadata, dependencies, tool config
├── Makefile                               # Common dev commands
├── README.md                              # Project overview and setup
├── SYNOPTIQ_MASTER_PLAN.md                # Research design (already written)
├── IMPLEMENTATION_PLAN.md                 # This file
└── .gitignore
```

---

## 2. Phase 0: Foundation & Developer Tooling

**Goal:** Project skeleton, all utilities, configuration system, dev tools. Everything that makes subsequent phases fast and error-free.

**Duration:** 1-2 days
**Cost:** $0 (all local)
**GPU:** None
**Blocks:** Everything

### Files to Create

```
synoptiq/__init__.py
synoptiq/_about.py
synoptiq/utils/__init__.py
synoptiq/utils/greek.py
synoptiq/utils/tokenization.py
synoptiq/utils/logging_.py
synoptiq/utils/types_.py
synoptiq/utils/constants.py
synoptiq/utils/io_.py
synoptiq/training/__init__.py
synoptiq/training/_config.py
configs/data.yaml
configs/model.yaml
configs/training.yaml
configs/bayesian.yaml
configs/modal.yaml
pyproject.toml
Makefile
.gitignore
```

### 2.1 `synoptiq/_about.py`

```python
"""Package metadata for SynoptiQ."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "SynoptiQ Contributors"
__license__ = "MIT"
__description__ = "A multi-task neural source criticism framework for the Synoptic Problem"
```

### 2.2 `synoptiq/__init__.py`

```python
"""
SynoptiQ — A Multi-Task Neural Source Criticism Framework for the Synoptic Problem.

This package provides tools for:
- Corpus loading and alignment of the Synoptic Gospels
- Domain-adaptive language model pre-training on Koine Greek (KoineFormer)
- Causal direction detection in parallel passages (Direction Scorer)
- Editorial tendency modeling (Editorial Drift)
- Proto-Q reconstruction (Fusion-in-Decoder)
- Bayesian model comparison of Synoptic source hypotheses
- Model interpretability via SHAP and Hawkins comparison
"""

from __future__ import annotations

from synoptiq._about import __author__, __description__, __license__, __version__
from synoptiq.data.corpus import Corpus
from synoptiq.utils.types_ import Book, Direction, PericopeAlignment, TokenRecord, Tradition

# Public API: everything a user or script needs
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    "Corpus",
    "Book",
    "Tradition",
    "Direction",
    "TokenRecord",
    "PericopeAlignment",
]
```

### 2.3 `synoptiq/utils/types_.py`

```python
"""Shared type aliases, TypedDict definitions, and Protocols.

All types used across the SynoptiQ codebase are defined here
to ensure consistency and enable static type checking (mypy strict).
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, TypeAlias, runtime_checkable

# ── Core identifiers ──

Book: TypeAlias = Literal["Matthew", "Mark", "Luke", "John"]

Tradition: TypeAlias = Literal[
    "triple", "double", "mark_unique", "matthean_unique", "lukan_unique"
]

Direction: TypeAlias = Literal["A_to_B", "B_to_A", "independent"]

# ── Token-level data ──

class TokenRecord(TypedDict):
    """A single token with full morphological annotation.

    Each token in the corpus carries its surface form, lemma,
    part-of-speech tag, morphological parsing, and book/verse location.
    This is the atomic unit of all SynoptiQ analyses.
    """

    token_id: str           # e.g., "Matt.1.1.3"
    book: Book
    chapter: int
    verse: int
    position: int           # 0-indexed token position within the verse
    text: str               # Surface form (polytonic Greek)
    normalized: str         # De-accented, lowercased form
    lemma: str              # Dictionary headword form
    pos: str                # Part-of-speech code (MorphGNT CCAT tagset)
    morph: str              # Full morphological string (person-tense-voice-mood-case-number-gender-degree)
    pericope_id: str | None # Aland Synopsis pericope number, None if unaligned
    is_punctuation: bool    # True for punctuation tokens

# ── Pericope alignment ──

class PericopeAlignment(TypedDict):
    """An aligned set of parallel Gospel passages for one pericope.

    Contains the tokens for each gospel containing this pericope,
    plus pairwise token alignment matrices.
    """

    pericope_id: str
    tradition: Tradition
    books: list[Book]
    tokens: dict[Book, list[TokenRecord]]
    alignment: dict[tuple[Book, Book], list[tuple[int | None, int | None]]]

# ── Model outputs ──

class DirectionScores(TypedDict):
    """Output of the DirectionScorer for a single pericope pair."""

    pericope_id: str
    book_a: Book
    book_b: Book
    prob_a_to_b: float
    prob_b_to_a: float
    prob_independent: float
    predicted_direction: Direction
    attention_asymmetry: float
    entropy_ratio: float

class EditorialFatigueScores(TypedDict):
    """Output of the EditorialFatigue detector for a single pericope."""

    pericope_id: str
    source_book: Book
    target_book: Book
    has_fatigue: bool
    fatigue_score: float          # 0-1, higher = more fatigue detected
    estimated_fatigue_position: float  # Normalized position (0-1) where fatigue begins
    consistency_score: float      # How consistent the editing is throughout

# ── Hypothesis definitions ──

class HypothesisSpec(TypedDict):
    """Specification of a Synoptic source hypothesis."""

    name: str                     # e.g., "Two-Source Hypothesis"
    abbreviation: str             # e.g., "2SH"
    order: list[Book]             # Composition order
    dependencies: list[tuple[Book, Book]]  # (source, target) pairs
    has_q: bool                   # Does it posit Q?
    description: str

# ── Protocols ──

@runtime_checkable
class TrainableModel(Protocol):
    """Protocol for any trainable SynoptiQ model."""

    def train(self) -> dict[str, list[float]]:
        """Run training loop, return metric history."""
        ...

    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation/test set."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint from disk."""
        ...
```

### 2.4 `synoptiq/utils/greek.py`

```python
"""Ancient Greek text processing utilities.

Handles normalization, accent handling, and Koine-specific
features like nomina sacra, movable nu, and sigma variants.

Koine Greek differs from classical Attic in several ways:
- Simpler sentence structure (more paratactic kai)
- Reduced use of the optative mood
- Semantic shift in particles (e.g., hina + subjunctive replacing infinitive)
- Nomina sacra abbreviations in manuscript tradition
- More consistent use of movable nu

Our normalization pipeline strips diacritics for comparison
while preserving lemma-level information for linguistic analysis.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# ── Constants ──

# Matches Greek characters including polytonic diacritics
# Range covers: Greek and Coptic (U+0370–U+03FF) + Greek Extended (U+1F00–U+1FFF)
_GREEK_PATTERN: Final = re.compile(r"[Ͱ-Ͽἀ-῿]+")

# Combining diacritical marks for Greek
# U+0300–U+036F: Combining Diacritical Marks
# U+0313: Combining comma above (psili / smooth breathing)
# U+0314: Combining reversed comma above (dasia / rough breathing)
# U+0342: Combining Greek perispomeni (circumflex)
# U+0345: Combining Greek ypogegrammeni (iota subscript)
_COMBINING_GREEK: Final = re.compile(r"[̀-ͯ͂ͅ]")

# Nomina sacra: abbreviated sacred names in NT manuscripts
# SBLGNT expands these, but we keep the mapping for manuscript-aware tokenization
NOMINA_SACRA: Final[dict[str, str]] = {
    "θ̅ς̅": "θεός",
    "κ̅ς̅": "κύριος",
    "ι̅η̅ς̅": "Ἰησοῦς",
    "χ̅ς̅": "Χριστός",
    "π̅ν̅α̅": "πνεῦμα",
    "π̅η̅ρ̅": "πατήρ",
    "υ̅ς̅": "υἱός",
    "σ̅η̅ρ̅": "σωτήρ",
    "δ̅α̅δ̅": "Δαυίδ",
    "ι̅η̅λ̅": "Ἰσραήλ",
    "α̅ν̅ο̅ς̅": "ἄνθρωπος",
    "ο̅υ̅ν̅ο̅ς̅": "οὐρανός",
}

# Words that exhibit movable nu (ν ἐφελκυστικόν)
# These words add -ν before vowels or at clause boundaries
_MOVABLE_NU_WORDS: Final[set[str]] = {
    "εἶπε", "εἶπεν",
    "ἔλεγε", "ἔλεγεν",
    "ἐστί", "ἐστίν",
    "φησί", "φησίν",
}


# ── Public API ──

def is_greek(text: str) -> bool:
    """Check whether a string contains Greek characters.

    Args:
        text: Any text string.

    Returns:
        True if the string contains at least one Greek character.
    """
    return bool(_GREEK_PATTERN.search(text))


def strip_accents(text: str) -> str:
    """Remove all Greek diacritics: accents, breathings, iota subscript.

    Uses Unicode normalization (NFD) to decompose composite characters
    into base + combining marks, then filters out the combining marks.

    Args:
        text: Polytonic Greek text with full diacritics.

    Returns:
        De-accented text with only base characters.

    Example:
        >>> strip_accents("ὁ λόγος τοῦ θεοῦ")
        'ο λογος του θεου'
        >>> strip_accents("ἐγένετο")
        'εγενετο'
    """
    nfd = unicodedata.normalize("NFD", text)
    # Remove combining diacritics (all characters in the combining mark ranges)
    stripped = _COMBINING_GREEK.sub("", nfd)
    return stripped


def normalize_greek(
    text: str,
    *,
    lower: bool = True,
    strip_diacritics: bool = True,
    normalize_sigma: bool = True,
) -> str:
    """Normalize Greek text for comparison and alignment.

    The normalization pipeline:
    1. Strip leading/trailing whitespace
    2. Lowercase (Koine was written in uncials/majuscules; case is a modern convention)
    3. Strip diacritics (breathings, accents, iota subscript)
    4. Normalize sigma variants: final sigma (ς) → medial sigma (σ)

    Args:
        text: Greek text string.
        lower: If True, convert to lowercase.
        strip_diacritics: If True, remove breathing marks, accents, iota subscript.
        normalize_sigma: If True, convert final sigma (ς) to medial sigma (σ).

    Returns:
        Normalized Greek string suitable for token comparison.

    Example:
        >>> normalize_greek("Ἐν ἀρχῇ ἦν ὁ λόγος")
        'εν αρχη ην ο λογος'
    """
    result = text.strip()
    if lower:
        result = result.lower()
    if strip_diacritics:
        result = strip_accents(result)
    if normalize_sigma:
        result = result.replace("ς", "σ")
    return result


def extract_lemmata(text: str) -> list[str]:
    """Extract a list of Greek lemmata from a space-separated string.

    Used to parse MorphGNT lemma columns, which contain one lemma per
    token position, with non-Greek tokens (punctuation, numbers) mixed in.

    Args:
        text: Space-separated string of lemmata/punctuation from MorphGNT.

    Returns:
        List of Greek-only lemmata, in order, with non-Greek tokens removed.
    """
    return [token for token in text.split() if is_greek(token)]


def parse_verse_ref(ref: str) -> tuple[str, int, int]:
    """Parse a verse reference like 'Matt 14:1' or 'Mark 6:45-8:26' into components.

    Args:
        ref: Verse reference string in standard scholarly format.

    Returns:
        Tuple of (book_name, start_verse_id, end_verse_id).
        Verse IDs are computed as chapter * 1000 + verse for easy range comparison.

    Raises:
        ValueError: If the reference cannot be parsed.

    Example:
        >>> parse_verse_ref("Matt 14:1-12")
        ('Matthew', 14001, 14012)
        >>> parse_verse_ref("Luke 19:27")
        ('Luke', 19027, 19027)
    """
    # Map abbreviated book names to canonical names
    book_map: dict[str, str] = {
        "Matt": "Matthew", "Mt": "Matthew",
        "Mark": "Mark", "Mk": "Mark",
        "Luke": "Luke", "Lk": "Luke",
        "John": "John", "Jn": "John",
    }

    pattern = re.compile(
        r"(?P<book>\w+)\s+"
        r"(?P<chapter>\d+):"
        r"(?P<start_v>\d+)"
        r"(?:-(?P<end_v>\d+))?"
    )
    match = pattern.match(ref)
    if not match:
        msg = f"Cannot parse verse reference: {ref!r}"
        raise ValueError(msg)

    book_abbr = match.group("book")
    book = book_map.get(book_abbr, book_abbr)
    chapter = int(match.group("chapter"))
    start_v = int(match.group("start_v"))
    end_v = int(match.group("end_v") or start_v)

    start_id = chapter * 1000 + start_v
    end_id = chapter * 1000 + end_v
    return book, start_id, end_id
```

### 2.5 `synoptiq/utils/constants.py`

```python
"""Fixed constants for the SynoptiQ project.

These are values that are definitionally true (book names, pericope ranges,
morphological tag mappings) — not configuration that might change between runs.
"""

from __future__ import annotations

from typing import Final

# ── Canonical book list ──
CANONICAL_BOOKS: Final[tuple[str, ...]] = (
    "Matthew", "Mark", "Luke", "John",
)

SYNOPTIC_BOOKS: Final[tuple[str, ...]] = (
    "Matthew", "Mark", "Luke",
)

# ── MorphGNT tagset reference ──
# See: https://github.com/morphgnt/sblgnt

# Part-of-speech codes used in MorphGNT
POS_TAGSET: Final[dict[str, str]] = {
    "A-": "adjective",
    "C-": "conjunction",
    "D-": "adverb",
    "I-": "interjection",
    "N-": "noun",
    "P-": "preposition",
    "RA": "definite article",
    "RD": "demonstrative pronoun",
    "RI": "interrogative/indefinite pronoun",
    "RP": "personal pronoun",
    "RR": "relative pronoun",
    "V-": "verb",
    "X-": "particle",
}

# Person codes
PERSON: Final[dict[str, str]] = {
    "1": "first",
    "2": "second",
    "3": "third",
}

# Tense codes
TENSE: Final[dict[str, str]] = {
    "P": "present",
    "I": "imperfect",
    "F": "future",
    "A": "aorist",
    "X": "perfect",
    "Y": "pluperfect",
}

# Voice codes
VOICE: Final[dict[str, str]] = {
    "A": "active",
    "M": "middle",
    "P": "passive",
}

# Mood codes
MOOD: Final[dict[str, str]] = {
    "I": "indicative",
    "D": "imperative",
    "S": "subjunctive",
    "O": "optative",
    "N": "infinitive",
    "P": "participle",
}

# Case codes
CASE: Final[dict[str, str]] = {
    "N": "nominative",
    "G": "genitive",
    "D": "dative",
    "A": "accusative",
    "V": "vocative",
}

# Number codes
NUMBER: Final[dict[str, str]] = {
    "S": "singular",
    "P": "plural",
}

# Gender codes
GENDER: Final[dict[str, str]] = {
    "M": "masculine",
    "F": "feminine",
    "N": "neuter",
}

# Degree codes
DEGREE: Final[dict[str, str]] = {
    "C": "comparative",
    "S": "superlative",
}


def parse_morph_tag(tag: str) -> dict[str, str]:
    """Parse a MorphGNT morphological tag into its component features.

    The MorphGNT tag format is: person-tense-voice-mood-case-number-gender-degree
    with '-' for unspecified features.

    Example:
        >>> parse_morph_tag("3-A-A-I--S-M-")
        {'person': 'third', 'tense': 'aorist', 'voice': 'active',
         'mood': 'indicative', 'number': 'singular', 'gender': 'masculine'}
    """
    parts = tag.split("-")
    if len(parts) < 8:
        parts.extend(["-"] * (8 - len(parts)))

    features: dict[str, str] = {}
    mapping = [
        ("person", PERSON, parts[0]),
        ("tense", TENSE, parts[1]),
        ("voice", VOICE, parts[2]),
        ("mood", MOOD, parts[3]),
        ("case", CASE, parts[4]),
        ("number", NUMBER, parts[5]),
        ("gender", GENDER, parts[6]),
        ("degree", DEGREE, parts[7]),
    ]

    for feature_name, lookup, code in mapping:
        if code in lookup:
            features[feature_name] = lookup[code]

    return features
```

### 2.6 `synoptiq/training/_config.py`

```python
"""Configuration dataclasses for all SynoptiQ training phases.

Uses frozen dataclasses for immutability and YAML serialization.
All paths are relative to the project root unless specified as absolute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration."""

    # Paths
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    external_dir: Path = Path("data/external")

    # Alignment
    alignment_method: Literal["needleman_wunsch", "smith_waterman"] = "needleman_wunsch"
    gap_open_penalty: float = -5.0
    gap_extend_penalty: float = -0.5
    lemma_match_score: float = 2.0
    surface_match_bonus: float = 1.0
    pos_match_bonus: float = 0.5

    # Splits
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20
    random_seed: int = 42

    # Augmentation
    n_bootstrap_samples: int = 100
    moving_window_size: int = 5
    moving_window_stride: int = 2
    scribal_error_rate: float = 0.01  # For synthetic noise simulation


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""

    # Base model
    base_model: str = "Heidelberg-NLP/gre-ta-base"
    model_type: Literal["t5-small", "t5-base"] = "t5-base"
    max_seq_length: int = 512
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12

    # Dropouts
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # LoRA (for task-specific adaptation)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "k", "v", "o")

    # Direction scorer
    direction_num_classes: int = 3
    cross_attn_num_heads: int = 8
    asymmetry_mlp_hidden: int = 512
    asymmetry_num_features: int = 8

    # Q reconstruction (FiD)
    fid_num_beams: int = 5
    fid_temperature: float = 1.0
    fid_max_generation_length: int = 512
    fid_min_generation_length: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameter configuration."""

    # Optimization
    learning_rate: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduling
    warmup_steps: int = 1000
    scheduler: Literal["cosine", "linear", "constant_with_warmup"] = "cosine"
    num_cycles: float = 0.5

    # Steps
    max_steps: int = 100_000
    eval_steps: int = 500
    save_steps: int = 2000
    logging_steps: int = 50

    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch = 32

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"

    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision (fp16)


@dataclass(frozen=True)
class DAPTConfig:
    """Domain-adaptive pre-training configuration."""

    # Corpus
    corpus_components: tuple[str, ...] = (
        "sblgnt",          # ~138K tokens
        "lxx",             # ~600K tokens
        "josephus",        # ~300K tokens
        "apostolic",       # ~35K tokens
    )

    # MLM
    mlm_probability: float = 0.15
    mean_noise_span_length: float = 3.0  # T5-style span corruption

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4  # Higher for pre-training
    max_steps: int = 80_000
    warmup_steps: int = 2000

    # Go/no-go benchmark
    target_mlm_perplexity_ratio: float = 1.05  # Within 5% of from-scratch baseline


@dataclass(frozen=True)
class BayesianConfig:
    """Bayesian model comparison configuration."""

    # MCMC
    num_chains: int = 4
    num_warmup: int = 500
    num_samples: int = 2000
    target_accept: float = 0.90

    # Bridge sampling
    bridge_n_iterations: int = 50000

    # Prior sensitivity grid
    prior_alpha_range: tuple[float, float, int] = (1.0, 5.0, 5)  # start, stop, n_steps
    prior_beta_range: tuple[float, float, int] = (1.0, 5.0, 5)

    # Convergence
    r_hat_threshold: float = 1.01
    min_ess: int = 400
```

### 2.7 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "synoptiq"
version = "0.1.0"
description = "A multi-task neural source criticism framework for the Synoptic Problem"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.12"
authors = [
    {name = "SynoptiQ Contributors"},
]
keywords = ["synoptic-problem", "biblical-studies", "nlp", "transformers", "ancient-greek"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Religion :: Biblical Studies",
]

dependencies = [
    "torch>=2.6",
    "transformers>=4.51",
    "peft>=0.15",
    "datasets>=3.3",
    "accelerate>=1.3",
    "sentencepiece>=0.2",
    "pandas>=2.2",
    "numpy>=2.2",
    "pyyaml>=6.0",
    "biopython>=1.85",
    "scikit-learn>=1.6",
    "shap>=0.46",
    "matplotlib>=3.10",
    "seaborn>=0.13",
    "wandb>=0.19",
    "text-fabric>=12.4",
    "arviz>=0.20",
    "pymc>=5.19",
    "tqdm>=4.67",
]

[project.optional-dependencies]
modal = ["modal>=0.73"]
dev = [
    "pytest>=8.3",
    "pytest-cov>=6.0",
    "ruff>=0.9",
    "mypy>=1.14",
    "ipykernel>=6.29",
    "jupyter>=1.1",
]
bayesian = ["rpy2>=3.5"]

[project.urls]
Homepage = "https://github.com/synoptiq/synoptiq"
Repository = "https://github.com/synoptiq/synoptiq.git"

[tool.ruff]
target-version = "py312"
line-length = 100
exclude = ["notebooks/"]

[tool.ruff.lint]
select = [
    "E", "W",     # pycodestyle errors and warnings
    "F",          # pyflakes
    "I",          # isort
    "N",          # pep8-naming
    "D",          # pydocstyle
    "UP",         # pyupgrade (use modern syntax)
    "SIM",        # flake8-simplify
    "TCH",        # type-checking imports
    "RUF",        # ruff-specific rules
    "B",          # flake8-bugbear
    "C4",         # flake8-comprehensions
]
ignore = [
    "D100",       # Missing docstring in public module
    "D104",       # Missing docstring in public package
    "D107",       # Missing docstring in __init__
    "D203",       # Blank line before class docstring (incompatible with D211)
    "D213",       # Multi-line summary on second line (incompatible with D212)
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true

[tool.mypy]
strict = true
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
show_error_codes = true

[[tool.mypy.overrides]]
module = [
    "transformers.*",
    "Bio.*",
    "wandb.*",
    "tqdm.*",
    "seaborn.*",
    "text_fabric.*",
    "rpy2.*",
    "sentencepiece.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = [
    "--strict-markers",
    "-q",
    "--tb=short",
    "--cov=synoptiq",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (use '-m \"not slow\"' to skip)",
    "gpu: marks tests that require a GPU",
    "integration: marks tests that require downloaded corpora",
]
```

### 2.8 `Makefile`

```makefile
# SynoptiQ development Makefile
# Usage: make <target>

.PHONY: help install dev-install test test-cov lint format typecheck check data clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

dev-install: ## Install with all dev dependencies
	pip install -e ".[dev]"

test: ## Run unit tests (skip slow and GPU tests)
	python -m pytest tests/ -m "not slow and not gpu" -q --tb=short

test-all: ## Run all tests including slow ones
	python -m pytest tests/ -q --tb=short

lint: ## Lint with ruff
	ruff check synoptiq/ tests/ scripts/

format: ## Format code with ruff
	ruff format synoptiq/ tests/ scripts/

typecheck: ## Static type check with mypy
	mypy synoptiq/

check: lint typecheck test ## Run all pre-commit checks

data: ## Run full data preparation pipeline
	python scripts/prepare_data.py

dapt: ## Run KoineFormer DAPT on Modal
	modal run modal/app_dapt.py::train_dapt

clean: ## Remove all build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
```

### Phase 0 Subtask Checklist

```
[ ] Create project directory structure (all __init__.py files)
[ ] Write synoptiq/_about.py
[ ] Write synoptiq/__init__.py (public API)
[ ] Write synoptiq/utils/types_.py (all type definitions)
[ ] Write synoptiq/utils/greek.py (Greek text utilities)
[ ] Write synoptiq/utils/constants.py (fixed constants)
[ ] Write synoptiq/utils/logging_.py (structured logging, W&B)
[ ] Write synoptiq/utils/tokenization.py (tokenizer wrapper)
[ ] Write synoptiq/utils/io_.py (safe I/O helpers)
[ ] Write synoptiq/training/_config.py (all config dataclasses)
[ ] Create configs/*.yaml files
[ ] Write pyproject.toml with all tool configs
[ ] Write Makefile
[ ] Write .gitignore
[ ] pip install -e ".[dev]"
[ ] make check (must pass: lint + typecheck + test)
[ ] git init && git add -A && git commit -m "feat: Phase 0 — project foundation"
```

---

## 3. Phase 1: Data Pipeline

**Goal:** Download all corpora, parse into canonical format, align tokens across gospels, split into train/val/test.

**Duration:** 2-3 days
**Cost:** $0 (all local)
**GPU:** None
**Depends On:** Phase 0
**Blocks:** Everything

### Files to Create

```
synoptiq/data/__init__.py
synoptiq/data/_download.py
synoptiq/data/_parse_sblgnt.py
synoptiq/data/_parse_morphgnt.py
synoptiq/data/_parse_proiel.py
synoptiq/data/_parse_n1904.py
synoptiq/data/_parse_lxx.py
synoptiq/data/_parse_josephus.py
synoptiq/data/_parse_apostolic.py
synoptiq/data/corpus.py
synoptiq/data/alignment.py
synoptiq/data/pericope.py
synoptiq/data/splits.py
synoptiq/data/augmentation.py
scripts/prepare_data.py
tests/data/test_corpus.py
tests/data/test_alignment.py
tests/data/test_pericope.py
tests/data/test_splits.py
```

### 3.1 Corpus Class Design

The `Corpus` class is the **single entry point** for all data access. Every model, trainer, and evaluation script interacts with data through Corpus — never directly with files.

```python
# synoptiq/data/corpus.py — key interface

class Corpus:
    """The complete Synoptic corpus with morphological annotation.

    This is an immutable, frozen-dataclass-style object. Once built,
    it provides fast token-level access, pericope grouping, and
    tradition-based iteration.

    Construction:
        corpus = Corpus.from_raw()           # First time: download + parse + align
        corpus = Corpus.from_parquet(path)   # Subsequent: load cached
    """

    # ── Construction ──
    @classmethod
    def from_raw(cls, data_dir: Path = Path("data/raw")) -> Corpus: ...
    @classmethod
    def from_parquet(cls, tokens_path: Path, pericopes_path: Path) -> Corpus: ...

    # ── Serialization ──
    def to_parquet(self, tokens_path: Path, pericopes_path: Path) -> None: ...

    # ── Properties ──
    n_tokens: int
    n_pericopes: int
    books: tuple[Book, ...]
    pericope_ids_by_tradition: dict[Tradition, list[str]]

    # ── Token access ──
    def get_tokens(self, book: Book, pericope_id: str) -> list[TokenRecord]: ...
    def get_verse(self, book: Book, chapter: int, verse: int) -> list[TokenRecord]: ...

    # ── Pericope iteration ──
    def iter_pericopes(
        self, *, tradition: Tradition | None = None, books: tuple[Book, ...] | None = None
    ) -> Iterator[PericopeAlignment]: ...

    # ── Direction pair generation ──
    def direction_pairs(
        self, *, tradition: Tradition = "triple", source: Book = "Mark"
    ) -> Iterator[tuple[str, Book, Book, list[TokenRecord], list[TokenRecord]]]: ...
```

### 3.2 Data Download Strategy

```
Download order (sequential, each clones a git repo):
1. SBLGNT     (CC-BY)     — primary Greek NT text
2. MorphGNT   (CC-BY-SA) — morphological tags, lemmas
3. PROIEL     (CC BY-NC-SA) — dependency syntax trees
4. N1904-TF   (Open)      — Aland pericope numbers
5. LXX        (via SWORD) — Septuagint Koine control text
6. Josephus   (via PACE)  — Koine control text
7. Apostolic  (CC-BY-SA)  — Koine control text

Each download checks for existing data first.
If repo exists → git pull
If repo missing → git clone --depth 1
```

### 3.3 Token Alignment

Two alignment strategies, configurable via `DataConfig.alignment_method`:

**Needleman-Wunsch (global)** — default. Forces alignment across the full sequence.
Best when the two passages have similar structure (most triple-tradition pericopes).

**Smith-Waterman (local)** — use when pericopes contain large insertions or omissions.
For example, Luke's "Great Omission" (Mark 6:45-8:26) or Matthew's expanded Sermon on the Mount.
Local alignment finds the best-matching subsequences without forcing alignment of
non-parallel material. Set via `alignment_method: smith_waterman` in config.

**Scoring matrix** (same for both methods):
```
Score matrix for token pair (i in A, j in B):
  Base: lemma_i == lemma_j        → +2.0
  Bonus: surface_i == surface_j   → +1.0  (identical surface forms = stronger match)
  Bonus: pos_i == pos_j           → +0.5  (same POS = less likely coincidental)
  Mismatch: lemma_i != lemma_j   → -1.0
  Gap open:                       → -5.0
  Gap extend:                      → -0.5
```

### 3.4 Pericope Classification Logic

```python
def classify_pericope(books: frozenset[Book]) -> Tradition:
    """Determine tradition type from gospel presence."""
    match books:
        case {"Matthew", "Mark", "Luke"}:
            return "triple"
        case {"Matthew", "Luke"}:
            return "double"
        case {"Mark"}:
            return "mark_unique"
        case {"Matthew"}:
            return "matthean_unique"
        case {"Luke"}:
            return "lukan_unique"
        case _:
            # Contains John, or Mark + only one other (partial parallel)
            msg = f"Unclassified book combination: {books}"
            raise ValueError(msg)
```

### 3.5 Data Splits

**Critical rule:** Pericope-atomic splitting. A pericope must never be split across train/val/test.

```
Stratification dimensions:
1. Tradition type (triple / double / unique)
2. Narrative vs. discourse genre (structural classification)
3. Gospel presence pattern

Split ratios: 60% train / 20% val / 20% test
```

### Phase 1 Subtask Checklist

```
[ ] Write synoptiq/data/_download.py (git clone/pull all repos)
[ ] Write synoptiq/data/_parse_sblgnt.py (XML → token list)
[ ] Write synoptiq/data/_parse_morphgnt.py (TSV → morphology)
[ ] Write synoptiq/data/_parse_proiel.py (XML → dependency trees)
[ ] Write synoptiq/data/_parse_n1904.py (TF → pericope mapping)
[ ] Write synoptiq/data/_parse_lxx.py (SWORD → Koine text)
[ ] Write synoptiq/data/_parse_josephus.py (PACE → Koine text)
[ ] Write synoptiq/data/_parse_apostolic.py (XML → Koine text)
[ ] Write synoptiq/data/alignment.py (Needleman-Wunsch alignment)
[ ] Write synoptiq/data/pericope.py (Aland classification)
[ ] Write synoptiq/data/splits.py (stratified split)
[ ] Write synoptiq/data/augmentation.py (bootstrap, windows, noise)
[ ] Write synoptiq/data/corpus.py (unified Corpus class)
[ ] Write scripts/prepare_data.py (full pipeline entry point)
[ ] Write tests/data/test_corpus.py (smoke test with small fixture)
[ ] Write tests/data/test_alignment.py (alignment correctness)
[ ] Write tests/data/test_pericope.py (classification logic)
[ ] Write tests/data/test_splits.py (stratification correctness)
[ ] Run: python scripts/prepare_data.py
[ ] Validate: corpus.n_tokens ≈ 138,000, corpus.n_pericopes > 300
[ ] Validate: alignment spot-check on 50 random token pairs (>95% correct)
[ ] Run: make check
[ ] Git commit: "feat: Phase 1 — data pipeline"
```

---

## 4. Phase 2: KoineFormer — Domain-Adaptive Language Model

**Goal:** Transform GreTa (classical AG T5) into KoineFormer (Koine Greek T5) via domain-adaptive pre-training and multi-task fine-tuning.

**Duration:** 5-7 days (mostly Modal GPU time + evaluation)
**Cost:** ~$15-18 (Modal A10G for DAPT, T4 for multi-task)
**GPU:** Modal A10G for DAPT, Modal T4 for multi-task
**Depends On:** Phase 1 (needs aligned corpus)
**Blocks:** Phases 3, 4, 5

### Architecture

```
    ┌───────────────────────────────────────────────┐
    │       GreTa (Heidelberg-NLP/gre-ta-base)        │
    │             ALL WEIGHTS FROZEN                   │
    └──────────────────┬────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  LoRA Adapters (r=16,a=32)  │  ← ~4.5M trainable params
        │  W_q, W_v, W_o + MLP        │
        └──────────────┬──────────────┘
                       │
          ┌────────────┴────────────┐
          │ 70% Koine               │ 30% Classical (Replay)
          │ SBLGNT, LXX, Josephus,  │ First1KGreek, Perseus
          │ Apostolic Fathers       │ Plato, Homer, Xenophon
          └────────────┬────────────┘
                       │
                       ▼
                KoineFormer
          (T5 encoder-decoder, DAPT'd)
                       │
          ┌────────────┴────────────┐
          │  Multi-task LoRA (frozen│
          │  base, adapters + heads)│
          ├── MLM (span corruption) │
          ├── POS tagging           │
          ├── Dep parsing (biaffine)│
          ├── Lemmatization         │
          └── Pericope detection    │
          └─────────────────────────┘
```

### 4.1 DAPT Implementation

```python
# synoptiq/training/dapt.py — key training loop

def train_dapt(
    model: KoineFormer,
    train_corpus: Corpus,
    val_corpus: Corpus,
    replay_corpus: Corpus,  # Classical Greek replay buffer (First1KGreek, Perseus)
    config: DAPTConfig,
    *,
    device: str = "cuda",
    output_dir: Path = Path("outputs"),
) -> dict[str, list[float]]:
    """Domain-adaptive pre-training of GreTa on Koine Greek.

    Uses PEFT (LoRA) rather than full fine-tuning to prevent catastrophic
    forgetting on the small (~1M token) Koine corpus:
      - GreTa backbone is FROZEN throughout
      - LoRA adapters (r=16, alpha=32) on W_q, W_v, W_o, and MLP layers
      - ~4.5M trainable params instead of 220M (220:1 token/param ratio)

    A replay buffer of Classical Greek text (First1KGreek, Perseus) is
    interleaved at 30/70 ratio to anchor the model's general Greek
    linguistic knowledge and prevent representation collapse.

    Uses T5-style span corruption: contiguous spans of tokens are
    replaced with sentinel tokens, and the decoder must reconstruct
    the original spans. This is more suitable for encoder-decoder
    architectures than BERT-style single-token masking.

    Args:
        model: The KoineFormer model wrapping a GreTa T5.
        train_corpus: Corpus for training (SBLGNT + LXX + Josephus + Apostolic).
        val_corpus: Held-out Koine corpus for validation.
        config: DAPT configuration.
        device: Training device.
        output_dir: Directory for checkpoints and logs.

    Returns:
        Dictionary of metric histories.
    """
    ...
```

### 4.2 Go/No-Go Benchmark

After DAPT:
1. Train a **from-scratch RoBERTa-small** on the same Koine corpus (6 layers, 384 hidden)
2. Evaluate MLM perplexity of both KoineFormer and the from-scratch model on held-out Koine
3. Ratio = PPL(KoineFormer) / PPL(from-scratch RoBERTa)
4. **Go if ratio ≤ 1.05** (within 5% of from-scratch). If > 1.05, continue DAPT or escalate to from-scratch T5 training.

### 4.2-A DAPT Data Scarcity Mitigations

The Koine corpus is ~1.07M tokens — tiny for pre-training. Two primary mitigations:

**Mitigation 1: PEFT-DAPT (LoRA, not full fine-tune)**
  - GreTa backbone is FROZEN during DAPT
  - LoRA adapters (r=16, alpha=32) on W_q, W_v, W_o, and MLP layers
  - Trainable params: ~4.5M (not 220M) — token/param ratio goes from 5:1 → 220:1
  - This prevents catastrophic forgetting of pre-trained classical Greek representations

**Mitigation 2: Classical Greek replay buffer**
  - Training batches mixed: 70% Koine tokens + 30% Classical/Attic tokens
  - Classical texts from First1KGreek / Perseus (Plato, Homer, Xenophon, etc.)
  - Acts as an architectural anchor, regularizing weights toward general Greek
  - Preserves understanding of classical syntax (optative, particles) that Koine loses

**Fallback (if validation perplexity plateaus despite above):**
  - Expand Koine corpus with Philo of Alexandria (~400K tokens), Polybius (~300K), Plutarch Moralia (~500K) — all public domain, all Hellenistic Koine-adjacent
  - Combined: ~2.27M tokens

### 4.3 Multi-Task LoRA Training

Each task gets its own LoRA adapter (r=16, alpha=32) on the encoder. The base encoder weights are frozen after DAPT.

```python
# synoptiq/models/encoder.py

class MultiTaskEncoder(nn.Module):
    """KoineFormer encoder with task-specific LoRA adapters and heads."""

    def __init__(self, base_encoder: nn.Module, model_config: ModelConfig):
        super().__init__()
        self.encoder = base_encoder  # Frozen DAPT-trained encoder

        # Task-specific LoRA adapters (applied to attention weights)
        self.pos_lora = LoraAdapter(...)
        self.dep_lora = LoraAdapter(...)
        self.lemma_lora = LoraAdapter(...)
        self.pericope_lora = LoraAdapter(...)

        # Task-specific heads
        self.pos_head = nn.Linear(768, n_pos_tags)
        self.dep_head = BiaffineParser(768, n_dep_labels)
        self.lemma_head = nn.Linear(768, n_lemmata)
        self.pericope_head = nn.Linear(768, 2)  # Binary: boundary or not
```

### Phase 2 Subtask Checklist

```
[ ] Write synoptiq/models/_base.py (BaseKoineFormer wrapper)
[ ] Write synoptiq/models/koineformer.py (KoineFormer: GreTa + DAPT management)
[ ] Write synoptiq/models/encoder.py (MultiTaskEncoder with LoRA)
[ ] Write synoptiq/training/_datasets.py (MLMDataset, POSDataset, DepDataset, etc.)
[ ] Write synoptiq/training/_collate.py (span corruption collate, POS collate, etc.)
[ ] Write synoptiq/training/_trainer.py (generic Trainer class)
[ ] Write synoptiq/training/dapt.py (DAPT training loop)
[ ] Write synoptiq/training/multitask.py (multi-task LoRA training loop)
[ ] Write modal/_common.py (shared Modal infrastructure)
[ ] Write modal/app_dapt.py (DAPT Modal app — A10G, long timeout)
[ ] Write modal/app_train.py (general training Modal app — T4)
[ ] Write scripts/train_dapt.py (DAPT CLI entry point)
[ ] Write scripts/train_multitask.py (multi-task CLI entry point)
[ ] Run: modal run modal/app_dapt.py::train_dapt  (DAPT on A10G)
[ ] Evaluate: MLM perplexity benchmark vs. from-scratch baseline
[ ] GO/NO-GO: PPL ratio ≤ 1.05?
[ ] Run: modal run modal/app_train.py::train_multitask  (multi-task on T4)
[ ] Evaluate: POS accuracy, UAS/LAS, lemma accuracy vs. Ancient-Greek-BERT, GreBERTa, Trankit
[ ] GO/NO-GO: KoineFormer beats Ancient-Greek-BERT on ≥3/4 tasks?
[ ] Save: models/koineformer/dapt/ and models/koineformer/multitask/
[ ] Run: make check
[ ] Git commit: "feat: Phase 2 — KoineFormer DAPT + multi-task"
```

---

## 5. Phase 3: Source Detection & Direction Scoring

**Goal:** Train the source detector (tradition classifier) and the direction scorer (cross-attention asymmetry model).

**Duration:** 3-4 days
**Cost:** ~$4 (T4)
**GPU:** Modal T4
**Depends On:** Phase 2 (needs KoineFormer encoder)
**Blocks:** Phases 4, 5, 6

### 5.1 Source Detector

Classifies Luke's pericopes as double tradition (Q) or triple tradition (Mark source). Validates that the two traditions are stylistically distinguishable — a key 2SH prediction.

```python
# synoptiq/models/source_detector.py

class SourceDetector(nn.Module):
    """Classify a pericope as double or triple tradition based on style."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 768):
        super().__init__()
        self.encoder = encoder  # Frozen KoineFormer encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),  # double or triple
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).last_hidden_state
        # Mean pool over non-padding tokens
        pooled = (h * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return self.classifier(pooled)
```

### 5.2 Direction Scorer

The flagship model. See master plan Section 5.3 for the full architecture.

```python
# synoptiq/models/direction.py — key interface

class DirectionScorer(nn.Module):
    """Detect copying direction between parallel passages via cross-attention asymmetry.

    This is the primary technical contribution of SynoptiQ.
    No prior transformer-based model exists for this task.

    Architecture:
    1. Shared frozen encoder → h_A, h_B
    2. Bidirectional cross-attention → asymmetry features h_asym
    3. **Two competing heads trained adversarially:**
       - Direction Head: predict A→B, B→A, independent
       - Authorship Head (via Gradient Reversal Layer): predict which author wrote the text
    4. The GRL forces h_asym to be INFORMATIVE for direction but
       UNINFORMATIVE for authorship — preventing the shortcut of
       "recognize Matthew's style → output default direction"
    """

    def __init__(
        self,
        encoder: nn.Module,          # Frozen KoineFormer encoder
        hidden_dim: int = 768,
        cross_attn_heads: int = 8,
        num_classes: int = 3,        # A→B, B→A, independent
        num_authors: int = 3,        # Matthew, Mark, Luke
        lambda_adv: float = 0.1,     # Adversarial loss weight
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.cross_attn = CrossAttentionEncoder(hidden_dim, cross_attn_heads, dropout)
        self.lambda_adv = lambda_adv

        # Direction head (primary)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Authorship adversary head (via Gradient Reversal Layer)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.authorship_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_authors),
        )

    def forward(
        self,
        input_ids_a: Tensor, attn_mask_a: Tensor,
        input_ids_b: Tensor, attn_mask_b: Tensor,
        author_ids: Tensor | None = None,  # For adversarial training
    ) -> tuple[Tensor, Tensor | None, AsymmetryFeatures]:
        """Forward pass with adversarial de-biasing.

        Returns:
            dir_logits: (batch, 3) direction predictions
            auth_logits: (batch, 3) authorship predictions (None if not adversarial)
            features: AsymmetryFeatures for interpretability
        """
        h_a = self.encoder(input_ids_a, attention_mask=attn_mask_a).last_hidden_state
        h_b = self.encoder(input_ids_b, attention_mask=attn_mask_b).last_hidden_state

        _, _, attn_a_to_b, attn_b_to_a = self.cross_attn(h_a, h_b)
        features = self._compute_asymmetry(attn_a_to_b, attn_b_to_a)
        h_asym = self._pool_asymmetry(features)

        dir_logits = self.direction_head(h_asym)

        auth_logits = None
        if author_ids is not None and self.training:
            h_reversed = self.grl(h_asym)
            auth_logits = self.authorship_head(h_reversed)

        return dir_logits, auth_logits, features

    @torch.no_grad()
    def predict(
        self,
        tokens_a: list[TokenRecord],
        tokens_b: list[TokenRecord],
    ) -> DirectionScores: ...


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for adversarial training (Ganin et al. 2016).

    During forward: passes input through unchanged.
    During backward: multiplies gradients by -alpha, maximising the adversary loss.

    This forces the encoder to produce representations that are
    predictive of direction but NOT predictive of authorship.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFn.apply(x, self.alpha)
```

**Training data composition:**
- Triple tradition (Mark → Matthew): ~280 pericopes
- Triple tradition (Mark → Luke): ~200 pericopes
- **Triple tradition Matthew-Luke pairs (BOTH copy Mark → labeled "independent"):** ~480 pairs
  - *Critical:* These pairs have HIGH lexical overlap (they're parallel Gospel passages)
    but NO direct copying relationship (both independently used Mark).
    This teaches the model that semantic similarity ≠ copying direction —
    exactly the distinction needed for double tradition under 2SH.
- Synthetic (scribal noise applied to known Koine texts): ~500 pairs
- **Total: ~1,460 training examples**

**Evaluation:**
- Accuracy on held-out triple tradition (known direction)
- Per-class F1 (A→B, B→A, independent)
- Expected calibration error (ECE)
- Encoplot baseline comparison
- **Ablation 1 (authorship confound check):** Replace surface tokens with POS-tag+lemma sequences.
  If accuracy drops sharply → model was using authorship style, not direction features.  
  If accuracy holds → model learned genuine copying asymmetry.
- **Ablation 2 (adversarial checkpoint):** Compare direction accuracy with vs. without GRL.
  Is direction accuracy preserved while authorship prediction falls to chance (~33%)?
- OT transfer: train on Chronicles → Sam/Kings (**LXX Greek**, not Hebrew — KoineFormer operates on Greek text; the Septuagint translation of these books contains the same copyist behaviors in the correct language), test on Synoptic (or vice versa)

### Phase 3 Subtask Checklist

```
[ ] Write synoptiq/models/source_detector.py
[ ] Write synoptiq/models/direction.py (DirectionScorer + CrossAttentionEncoder + AsymmetryFeatures)
[ ] Write synoptiq/training/direction_train.py (direction scorer training loop)
[ ] Write synoptiq/evaluation/metrics.py (accuracy, F1, ECE, BLEU, ROUGE, chrF, BERTScore)
[ ] Write synoptiq/evaluation/baselines.py (Encoplot, length heuristic, cosine sim, RF)
[ ] Write synoptiq/evaluation/calibration.py (reliability diagrams, temperature scaling)
[ ] Write scripts/train_source_detector.py
[ ] Write scripts/train_direction.py
[ ] Write scripts/eval_all.py (first version: direction scorer eval)
[ ] Write tests/models/test_direction.py (forward pass shape check, asymmetry feature ranges)
[ ] Write tests/evaluation/test_metrics.py (metric correctness)
[ ] Write tests/evaluation/test_baselines.py (baseline function correctness)
[ ] Run: modal run modal/app_train.py::train_source_detector
[ ] Run: modal run modal/app_train.py::train_direction
[ ] Evaluate: direction scorer accuracy on held-out triple tradition
[ ] GO/NO-GO: accuracy > 80%?
[ ] Evaluate: Encoplot baseline comparison
[ ] Evaluate: OT transfer (Chronicles → Sam/Kings cross-test)
[ ] Save: models/koineformer/direction/
[ ] Run: make check
[ ] Git commit: "feat: Phase 3 — source detection + direction scoring"
```

---

## 6. Phase 4: Editorial Tendency Modeling

**Goal:** Learn Matthean and Lukan editorial tendencies from triple tradition. Implement the editorial fatigue loss. Validate against Goodacre's known fatigue examples.

**Duration:** 3-4 days
**Cost:** ~$3 (T4)
**GPU:** Modal T4
**Depends On:** Phase 3 (uses direction scorer encoder)

### 6.1 Editorial Drift Model

```python
# synoptiq/models/editor.py

class EditorialDrift(nn.Module):
    """Seq2seq model that learns editorial tendencies.

    Trained on (Mark → Matthew) and (Mark → Luke) triple-tradition pairs.
    The model learns what changes each evangelist characteristically makes.
    """

    def __init__(
        self,
        encoder: nn.Module,       # Frozen KoineFormer encoder
        decoder: nn.Module,       # KoineFormer decoder (LoRA-adapted)
    ): ...

    def forward(
        self,
        source_ids: Tensor,      # Mark's text
        target_ids: Tensor,      # Matthew's (or Luke's) version
    ) -> Tensor:                 # Cross-entropy loss
        ...


class FatigueLoss(nn.Module):
    """Position-weighted consistency loss for editorial fatigue detection.

    Formalizes Goodacre's insight: when a copyist makes changes at the
    beginning of a pericope but fails to sustain them, the text shows
    internal inconsistency.

    L_fatigue = (1/N) Σ_i w(i) · D_KL(edit_dist_i || source_dist_i)

    where w(i) = exp(-λ · i/N) gives lower weight to later positions
    (later changes are expected — the copyist has already fatigued).
    """

    def __init__(self, lambda_fatigue: float = 1.0): ...

    def forward(
        self,
        edit_distribution: Tensor,   # (batch, seq, vocab) — predicted edits
        source_distribution: Tensor,  # (batch, seq, vocab) — source tokens
        position: Tensor,             # (batch, seq) — normalized position [0, 1]
    ) -> Tensor:                      # Scalar fatigue loss
        ...
```

### 6.2 Fatigue Detection on Goodacre's Examples

The 9 key pericopes from Goodacre (6 triple + 3 double tradition) serve as the evaluation set. The model should:

1. Correctly detect fatigue in ≥4/6 triple-tradition examples (Mark → Matthew/Luke fatigue)
2. Correctly detect fatigue in ≥2/3 double-tradition examples (Luke fatiguing with Matthew)
3. Identify the approximate position where fatigue begins (within 20% of Goodacre's analysis)

### Phase 4 Subtask Checklist

```
[ ] Write synoptiq/models/editor.py (EditorialDrift + FatigueLoss)
[ ] Write synoptiq/training/editor_train.py (seq2seq editor training + fatigue detection)
[ ] Write scripts/train_editor.py
[ ] Run: modal run modal/app_train.py::train_editor (Mark→Matthew editor)
[ ] Run: modal run modal/app_train.py::train_editor (Mark→Luke editor)
[ ] Evaluate: BLEU/ROUGE on held-out triple tradition (how well does model "edit" Mark?)
[ ] Evaluate: Fatigue detection on Goodacre's 9 examples
[ ] GO/NO-GO: ≥4/6 triple fatigue + ≥2/3 double fatigue correct?
[ ] Analyze: learned tendencies vs. scholarly descriptions (Davies & Allison, Fitzmyer)
[ ] Save: models/koineformer/editor/
[ ] Run: make check
[ ] Git commit: "feat: Phase 4 — editorial tendency + fatigue loss"
```

---

## 7. Phase 5: Proto-Q Reconstruction (FiD)

**Goal:** Train a Fusion-in-Decoder model to reconstruct proto-Q from Matthew + Luke's double tradition passages.

**Duration:** 5-7 days
**Cost:** ~$9 (A10G — needs more VRAM for FiD with long sequences)
**GPU:** Modal A10G
**Depends On:** Phase 3 (uses KoineFormer encoder), Phase 4 (editorial tendencies inform reconstruction)

### 7.1 FiD Architecture

```python
# synoptiq/models/reconstruction.py

class QReconstructor(nn.Module):
    """Fusion-in-Decoder for proto-Q reconstruction.

    Encodes Matthew's and Luke's versions of a pericope independently,
    concatenates their hidden states, and decodes Q's hypothesized text
    via cross-attention over the concatenated representations.

    The decoder can selectively draw from either Matthew's or Luke's
    wording at each generation step — mimicking the IQP's manual
    adjudication process.
    """

    def __init__(
        self,
        encoder: nn.Module,          # Frozen KoineFormer encoder (shared)
        decoder: nn.Module,          # KoineFormer decoder (LoRA-adapted for Q)
        hidden_dim: int = 768,
        num_beams: int = 5,
        max_length: int = 512,
        vocab_size: int = 32000,
    ): ...

    def forward(
        self,
        matthew_ids: Tensor, matthew_mask: Tensor,
        luke_ids: Tensor, luke_mask: Tensor,
        target_ids: Tensor,          # Q target (Mark in Stage 1, IQP in Stage 2)
    ) -> Tensor:                     # Cross-entropy loss
        ...

    @torch.no_grad()
    def generate(
        self,
        matthew_ids: Tensor, matthew_mask: Tensor,
        luke_ids: Tensor, luke_mask: Tensor,
        *,
        num_beams: int = 5,
        temperature: float = 1.0,
        constraint_vocab: set[int] | None = None,  # SBLGNT lexicon
    ) -> tuple[Tensor, Tensor]:       # (generated_ids, generation_scores)
        ...
```

### 7.2 Training Stages

**Stage 1: Mark Reconstruction (triple tradition)**
Input: (Matthew's Mark-based text, Luke's Mark-based text) → Target: Mark
This is our supervised signal. The model learns to factor out editorial changes.

**Stage 2: Q Reconstruction (double tradition)**
Input: (Matthew's Q-based text, Luke's Q-based text) → Target: IQP {A}-rated Q text
Transfer from triple to double tradition.

**Validation:**
1. **Primary:** Mark reconstruction BLEU on held-out triple tradition (>0.40 target)
2. **Secondary:** IQP agreement on held-out {A}-rated pericopes
3. **Tertiary:** Thomas triangulation on ~40 overlapping sayings

### 7.2-A Anti-Degeneracy Safeguards

The decoder's default behavior under FiD is to "average" Matthew's and Luke's wording
into a semantic mush that fits neither text. Three independent guards prevent this:

**Guard 1 — Contrastive loss penalty:**
During training, if the generated text Y is too similar to EITHER Matthew (X_Mt) or
Luke (X_Lk), it receives a penalty:
```
L_total = L_CE - γ · max(Sim(Y, X_Mt), Sim(Y, X_Lk))
```
where Sim is BLEU or cosine similarity, and γ = 0.3.
This forces the model to find the COMMON ANCESTOR, not just copy the closest input.

**Guard 2 — Fluency discriminator:**
After generation, pass the candidate through a frozen KoineFormer MLM head.
If perplexity > threshold (indicating non-grammatical Greek), flag and reject.
See also: Pointer-Generator copy mechanism as an alternative decoder config.

**Guard 3 — Lexical stability constraint:**
Where Matthew and Luke agree verbatim, Q should preserve the agreed wording
with high probability (≥ 0.90). This is the IQP's own editorial principle.
Implemented as a constrained beam search that reduces the probability of
tokens that contradict verbatim agreements.

### Phase 5 Subtask Checklist

```
[ ] Write synoptiq/models/reconstruction.py (QReconstructor with FiD)
[ ] Write synoptiq/training/reconstruction_train.py (2-stage training loop)
[ ] Write scripts/train_reconstruction.py
[ ] Run Stage 1: modal run modal/app_dapt.py::train_reconstruction_stage1 (Mark recon, A10G)
[ ] Evaluate Stage 1: BLEU/ROUGE/chrF on held-out triple tradition
[ ] GO/NO-GO: Mark reconstruction BLEU > 0.40?
[ ] Run Stage 2: Transfer to double tradition, fine-tune on IQP {A}/{B}
[ ] Evaluate Stage 2: IQP agreement, Thomas triangulation
[ ] Run constrained decoding checks: all generated tokens in SBLGNT lexicon?
[ ] Run consistency checks: Q vocabulary/style consistent across pericopes?
[ ] Run divergence analysis: where does model disagree with IQP {C}/{D}?
[ ] Save: models/koineformer/reconstruction/
[ ] Run: make check
[ ] Git commit: "feat: Phase 5 — FiD Q reconstruction"
```

---

## 8. Phase 6: Bayesian Model Comparison

**Goal:** Compute Bayes factors for all four Synoptic hypotheses using direction scorer outputs as data.

**Duration:** 2-3 days
**Cost:** $0 (CPU only, runs locally)
**GPU:** None
**Depends On:** Phase 3 (needs trained direction scorer)

### 8.1 Model Specification

```python
# synoptiq/bayesian/models.py

def build_2sh_model(
    direction_means: np.ndarray,   # μ_i — MC Dropout mean per pericope (n_pericopes,)
    direction_vars: np.ndarray,    # σ²_i — MC Dropout variance per pericope (n_pericopes,)
) -> pm.Model:
    """Two-Source Hypothesis: Matthew and Luke independently use Mark + Q.

    Predicts direction scores ≈ 0.5 for double tradition (no direct dependence).

    UNCERTAINTY-AWARE: The model takes both the mean AND variance from
    MC Dropout (or Laplace approximation) inference. Pericopes where the
    neural network is uncertain (high σ²) contribute less to the posterior,
    because their precision κ_i ∝ 1/σ²_i is low.

    Pipeline: DirectionScorer → MC Dropout (T=20) → (μ_i, σ²_i) → PyMC

    IMPORTANT: `direction_means` must be temperature-calibrated BEFORE
    entering this model (via synoptiq.evaluation.calibration).
    """
    n_pericopes = len(direction_means)
    with pm.Model() as model:
        # Hyperpriors
        alpha_ind = pm.Gamma("alpha_ind", mu=2, sigma=0.5)
        beta_ind = pm.Gamma("beta_ind", mu=2, sigma=0.5)

        # Per-pericope precision: high variance → low influence on posterior
        precision = pm.Deterministic(
            "precision",
            1.0 / (direction_vars + 1e-8),
        )

        # Parameterize Beta using mean + precision (not raw α, β)
        # α_i = μ_i · κ_i,  β_i = (1 - μ_i) · κ_i
        alpha_obs = direction_means * precision
        beta_obs = (1.0 - direction_means) * precision

        # Likelihood: direction scores near 0.5 (no direct dependence)
        scores = pm.Beta(
            "scores",
            alpha=alpha_obs,
            beta=beta_obs,
            observed=direction_means,
        )
    return model


def build_fgh_model(
    direction_means: np.ndarray,
    direction_vars: np.ndarray,
) -> pm.Model:
    """Farrer-Goulder Hypothesis: Matthew is source for Luke.

    Same uncertainty-aware structure as 2SH model.
    Predicts direction means > 0.5 for double tradition (Luke copied Matthew).
    """
    with pm.Model() as model:
        alpha_mt_to_lk = pm.Gamma("alpha_mt_to_lk", mu=4, sigma=1)
        beta_mt_to_lk = pm.Gamma("beta_mt_to_lk", mu=2, sigma=0.5)

        precision = 1.0 / (direction_vars + 1e-8)
        alpha_obs = direction_means * precision
        beta_obs = (1.0 - direction_means) * precision

        scores = pm.Beta(
            "scores",
            alpha=alpha_obs,
            beta=beta_obs,
            observed=direction_means,
        )
    return model


def build_augustinian_model(direction_scores: np.ndarray) -> pm.Model: ...
def build_griesbach_model(direction_scores: np.ndarray) -> pm.Model: ...
```

### 8.2 Bridge Sampling

```python
# synoptiq/bayesian/bridge.py

def compute_bayes_factor(
    model_h1: pm.Model,
    model_h2: pm.Model,
    *,
    method: Literal["bridge", "smc"] = "bridge",
) -> tuple[float, dict[str, float]]:
    """Compute Bayes factor BF_12 = P(Data | H1) / P(Data | H2).

    Uses bridge sampling (Meng & Wong 1996) via R's bridgesampling package
    for the most reliable marginal likelihood estimates with moderate
    computational cost.

    Args:
        model_h1: Fitted PyMC model for hypothesis 1.
        model_h2: Fitted PyMC model for hypothesis 2.
        method: "bridge" for bridge sampling, "smc" for Sequential MC.

    Returns:
        Tuple of (BF_12, diagnostics) where diagnostics includes
        standard error and effective sample size.
    """
```

### Phase 6 Subtask Checklist

```
[ ] STEP 1 — UNCERTAINTY INFERENCE: Run DirectionScorer on all double-tradition
    pericopes with MC Dropout enabled (T=20 forward passes). Collect (μ_i, σ²_i)
    for each pericope. This produces the empirical distribution, not a point estimate.
[ ] STEP 2 — CALIBRATION: Temperature-scale the MC Dropout means μ_i via
    synoptiq.evaluation.calibration. This corrects overconfidence before PyMC.
[ ] Write synoptiq/bayesian/_hypotheses.py (hypothesis model builders)
[ ] Write synoptiq/bayesian/_likelihood.py (Beta-binomial hierarchical likelihood)
[ ] Write synoptiq/bayesian/models.py (PyMC model construction + sampling)
    — Models now accept (μ_i, σ²_i), compute precision κ_i = 1/σ²_i,
      and parameterize Beta(μ·κ, (1-μ)·κ) so uncertain pericopes contribute less.
[ ] Write synoptiq/bayesian/bridge.py (bridge sampling via rpy2)
[ ] Write synoptiq/bayesian/sensitivity.py (prior grid + BF contour plots)
[ ] Write scripts/sample_bayesian.py
[ ] Run locally: python scripts/sample_bayesian.py (4 chains, 500 warmup, 2000 samples)
[ ] Check convergence: R-hat < 1.01 for all params, ESS > 400
[ ] Compute Bayes factors: BF(2SH vs FGH), BF(2SH vs Augustinian), BF(2SH vs Griesbach)
[ ] Run prior sensitivity grid
[ ] Generate BF contour plots for paper
[ ] Run posterior predictive checks (Bayesian p-value)
[ ] Run LOO-CV via arviz.compare()
[ ] GO/NO-GO: BF stable across bridge sampling vs SMC?
[ ] Save: outputs/bayesian/
[ ] Run: make check
[ ] Git commit: "feat: Phase 6 — Bayesian model comparison"
```

---

## 9. Phase 7: Interpretability & Robustness

**Goal:** SHAP feature importance, Hawkins 1899 comparison, BERTViz visualizations, multi-edition sensitivity.

**Duration:** 3-5 days
**Cost:** ~$2 (T4 for SHAP computation — can also run locally on CPU)
**GPU:** Modal T4 (optional, SHAP is CPU-able)
**Depends On:** Phases 3, 5, 6

### 9.1 SHAP + Hawkins Comparison

```python
# synoptiq/interpretability/shap_analysis.py

def compute_shap_direction(
    model: DirectionScorer,
    dataset: DirectionDataset,
    *,
    n_background: int = 100,
    n_samples: int = 50,
) -> pd.DataFrame:
    """Compute SHAP values for the direction scorer.

    Returns a DataFrame with columns: pericope_id, feature, shap_value, hawkins_status.

    Hawkins status: 'convergent' (in both SHAP and Hawkins),
    'divergent' (SHAP-important but not in Hawkins → potential discovery),
    'hawkins_only' (in Hawkins but low SHAP → candidate for re-examination).
    """
    ...


def compare_to_hawkins(
    shap_df: pd.DataFrame,
    hawkins_features: dict[Book, set[str]],
    *,
    top_k: int = 20,
) -> dict[str, list[str]]:
    """Compare SHAP-important features to Hawkins' Horae Synopticae (1899).

    Returns three lists:
    - convergent: Features important in both methods
    - divergent: SHAP-important features NOT in Hawkins
    - hawkins_only: Hawkins features with low SHAP importance
    """
    ...
```

### 9.2 Multi-Edition Sensitivity

```python
# scripts/eval_all.py — extended

def run_multi_edition_sensitivity(
    direction_scorer: DirectionScorer,
    corpus_na28: Corpus,
    corpus_tr: Corpus,        # Textus Receptus
    corpus_majority: Corpus,  # Majority Text
    corpus_wh: Corpus,        # Westcott & Hort
) -> pd.DataFrame:
    """Run direction scorer on all four text editions.

    Returns DataFrame comparing direction scores across editions.
    If conclusions are edition-invariant → robustness.
    If conclusions flip → text-critical dependency (important finding).
    """
    ...
```

### Phase 7 Subtask Checklist

```
[ ] Write synoptiq/interpretability/shap_analysis.py
[ ] Write synoptiq/interpretability/hawkins.py (Hawkins feature loading + comparison)
[ ] Write synoptiq/interpretability/bertviz_viz.py (attention visualization exports)
[ ] Write scripts/run_interpretability.py
[ ] Run SHAP: compute feature importance for direction scorer
[ ] Generate: Top 20 SHAP-important features table with Hawkins status
[ ] Generate: Hawkins Jaccard similarity scores
[ ] Generate: BERTViz attention maps for 10 most discriminative pericopes
[ ] Run multi-edition sensitivity (NA28 vs TR vs Majority Text vs WH)
[ ] Run counterfactual analysis on key pericopes
[ ] Generate all figures for paper
[ ] Save: outputs/figures/
[ ] Run: make check
[ ] Git commit: "feat: Phase 7 — interpretability + robustness"
```

---

## 10. Modal Deployment Design

### 10.1 Image Definition

```python
# modal/_common.py

from modal import App, Image, Secret, Volume

# Single image for all tasks — built once, cached across runs
SYNOPTIQ_IMAGE = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.6",
        "transformers>=4.51",
        "peft>=0.15",
        "datasets>=3.3",
        "accelerate>=1.3",
        "sentencepiece>=0.2",
        "pandas>=2.2",
        "pyyaml>=6.0",
        "biopython>=1.85",
        "scikit-learn>=1.6",
        "wandb>=0.19",
        "tqdm>=4.67",
    )
    .env({
        "WANDB_PROJECT": "synoptiq",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

# Persistent volumes (survive between runs)
DATA_VOLUME = Volume.from_name("synoptiq-data", create_if_missing=True)
MODELS_VOLUME = Volume.from_name("synoptiq-models", create_if_missing=True)
OUTPUTS_VOLUME = Volume.from_name("synoptiq-outputs", create_if_missing=True)

# GPU specifications
GPU_T4 = "t4"         # $0.47/hr — default for most tasks
GPU_A10G = "a10g"     # $1.30/hr — DAPT, FiD reconstruction

app = App("synoptiq", image=SYNOPTIQ_IMAGE)
```

### 10.2 Cost-Optimized Function Deployment

```python
# modal/app_dapt.py — DAPT only (most expensive, needs A10G)
@app.function(
    gpu=GPU_A10G,
    volumes={"/data": DATA_VOLUME, "/models": MODELS_VOLUME},
    timeout=3600 * 12,  # 12h max
    container_idle_timeout=300,
)
def train_dapt(config_path: str = "configs/training.yaml"):
    """KoineFormer DAPT — ~$12-15 on A10G."""
    ...

# modal/app_train.py — All other training (T4, cheaper)
@app.function(
    gpu=GPU_T4,
    volumes={"/data": DATA_VOLUME, "/models": MODELS_VOLUME, "/outputs": OUTPUTS_VOLUME},
    timeout=3600 * 6,
    container_idle_timeout=300,
)
def train_direction(config_path: str = "configs/training.yaml"):
    """Direction scorer training — ~$3-4 on T4."""
    ...
```

### 10.3 Modal Secrets

```toml
# modal/secrets.toml (git-ignored)
[wandb]
api_key = "your-wandb-api-key"

[huggingface]
token = "hf_your_token_here"  # Not needed for GreTa (public), needed for gated models
```

### Phase-by-Phase Modal Cost Breakdown

| Phase | Task | GPU | Est. Hours | Est. Cost |
|-------|------|-----|------------|-----------|
| 2 | DAPT | A10G | 10-14 | $13-18 |
| 2 | Multi-task LoRA | T4 | 4-6 | $2-3 |
| 3 | Source detector | T4 | 1-2 | $0.50-1 |
| 3 | Direction scorer | T4 | 6-8 | $3-4 |
| 4 | Editorial drift | T4 | 4-6 | $2-3 |
| 5 | FiD Stage 1 (Mark recon) | A10G | 4-5 | $5-7 |
| 5 | FiD Stage 2 (Q recon) | A10G | 2-3 | $3-4 |
| 7 | SHAP computation | T4 | 2-4 | $1-2 |
| **Total** | | | **33-48** | **$30-42** |

**Budget risk mitigation:** If we approach the $30 limit:
1. Defer Phase 5 Stage 2 (Q reconstruction, least critical for Paper 2)
2. Run Phase 2 DAPT with smaller batch size on T4 (takes longer but cheaper)
3. Run Phase 7 SHAP locally on CPU (slower but free)

---

## 11. Strategic Timeline: Three Independent Paper Checkpoints

The 7 phases form a chain of dependencies, but they produce **three independent paper checkpoints**. Each checkpoint is a complete, submit-table result that does NOT depend on later phases. If any phase fails, you still have a paper from the checkpoint before it.

### Dependency Map (the actual constraint)

```
M0: Data Pipeline (Phase 1) ─── no paper, just infrastructure
         │
         ├──► Paper A ◄── Phase 2 (KoineFormer DAPT + multi-task)
         │       │
         │       └──► Paper B ◄── Phases 3+4 (Direction Scorer + Editorial Fatigue + Baselines)
         │               │
         │               ├──► Paper C ◄── Phases 5+6+7 (Q Recon + Bayesian + Interpretability)
         │               │
         │               └──► "Failure" pivot: Paper still possible as negative result
```

Each paper:
- Has its own submission-ready output
- Does NOT require later phases to be complete
- Has a "pivot" plan if results are negative (negative results are also publishable)

---

### Paper A: KoineFormer — Parameter-Efficient Adaptation for Ancient Greek

**What:** Phase 2 only (DAPT + multi-task evaluation)
**Constraints:** Phase 1 must be complete (needs aligned Koine corpus)
**Requires:** Only Phase 1. Does NOT need direction scorer, editorial drift, or anything later.
**Submission-ready at:** Week 6-8

**Contribution:**
- PEFT-DAPT strategy (LoRA + replay buffer) for domain-adapting Ancient Greek T5 to Koine
- First encoder-decoder model for Koine Greek specifically
- Benchmarks: POS accuracy, UAS/LAS, lemma accuracy vs. Ancient-Greek-BERT, GreBERTa, Trankit
- Ablation: full fine-tune vs. LoRA-only DAPT on 1M tokens (demonstrates representation collapse risk)

**Venue:** LaTeCH-CLfL (COLING workshop), or *Digital Scholarship in the Humanities* (journal)

**Go deadline:** Week 8

| Week | Milestone | Output | Paper-ready? |
|------|-----------|--------|-------------|
| 5 | DAPT complete + Go/No-go pass | KoineFormer checkpoint | No — need benchmarks |
| 6 | Multi-task LoRA complete | POS/dep/lemma/pericope metrics | No |
| 7 | Baseline comparisons done | Table vs. Ancient-Greek-BERT, GreBERTa, Trankit | Draft tables |
| 8 | Write-up + figures | Complete manuscript draft | ✅ YES — submit |

**Negative pivot:** If DAPT perplexity ratio > 1.05 (domain gap not closing), write as: *"LoRA Domain Adaptation for Ancient Greek: Limitations and Empirical Findings."* The comparison of full fine-tune vs. LoRA on a 1M-token corpus is itself a useful empirical result for the low-resource NLP community.

---

### Paper B: Direction Detection in Parallel Ancient Texts — A Cross-Attention Approach

**What:** Phases 3 + 4 (direction scorer + baselines + editorial fatigue)
**Constraints:** Phase 1 must be complete (needs aligned pericopes).
**Does NOT require:** Paper A to be successful! Paper B can use Ancient-Greek-BERT (encoder-only) as the baseline encoder. GreTa/KoineFormer is beneficial but not essential — the direction scorer works on ANY encoder.
**Submission-ready at:** Week 12-14

**Contribution:**
- First transformer-based direction-of-copying detector (no prior model exists for this task)
- Cross-attention asymmetry features (8 novel feature types)
- Adversarial style de-biasing via Gradient Reversal Layer
- Editorial fatigue formalized as a differentiable loss function
- Validated on two independent ground-truth domains: Synoptic triple tradition (NT) + Chronicles→Sam/Kings (LXX OT)
- Encoplot n-gram baseline and POS-tag+lemma ablation
- Achieves >80% accuracy (target) or provides rigorous analysis of why direction detection fails

**Venue:** CHR (Computational Humanities Research), EMNLP main, or ACL

**Go deadline:** Week 14

| Week | Milestone | Output | Paper-ready? |
|------|-----------|--------|-------------|
| 9 | Encoplot + length baseline computed | Hard baseline scores (N/A — just reference) | Table in paper |
| 10 | Direction scorer trained (no GRL) | Preliminary accuracy | Draft results |
| 11 | GRL adversarial de-biasing added | Final accuracy + authorship ablation | Ablation tables |
| 12 | POS-tag+lemma ablation run | Direction score holds → true direction | Robustness section |
| 13 | Editorial fatigue integrated and validated | ≥4/6 Goodacre examples detected | Fatigue section draft |
| 14 | OT transfer validation + write-up | Cross-domain accuracy | ✅ YES — submit |

**Negative pivot:** If accuracy < 75% (not much better than Encoplot), paper becomes: *"Cross-Attention Asymmetry is Insufficient for Ancient Text Direction Detection: A Systematic Evaluation."* Publishable negative result with (a) comprehensive baseline comparison, (b) ablation of why the transformer approach fails, and (c) recommendations for future work. Important for the field to know this doesn't work.

**Key insight for independence from Paper A:** The direction scorer doesn't need KoineFormer — it needs a frozen encoder that produces token-level representations. Ancient-Greek-BERT works for this. If Paper A DAPT is 1) not done yet or 2) failed, Paper B proceeds with Ancient-Greek-BERT as the encoder. The paper simply says "we used Ancient-Greek-BERT" — it's still novel because no one has built a transformer-based direction detector at all.

---

### Paper C: The Synoptic Problem — A Bayesian Deep Learning Synthesis

**What:** Phases 5 + 6 + 7 (Q reconstruction + Bayesian comparison + interpretability)
**Constraints:** Paper B must succeed (accuracy > 75% — enough signal for the Bayesian model to work with)
**Requires:** Direction scorer from Paper B. Q reconstruction also needs KoineFormer decoder (so Paper A helps).
**Submission-ready at:** Week 24-26

**Contribution:**
- FiD-based proto-Q reconstruction with triple→double transfer learning
- Mark reconstruction proxy validation (the only known-ground-truth evaluation of Q reconstruction)
- Uncertainty-aware Bayesian model comparison (first Bayesian treatment of the Synoptic Problem)
- SHAP + Hawkins 1899 interpretability bridge (a method that humanities scholars can engage with)
- Multi-edition sensitivity analysis across 4 text editions
- Pericope-level decision analysis: which 10-15 pericopes most discriminate the hypotheses?

**Venue:** ACL, *New Testament Studies*, or *Journal of Biblical Literature* (computational section)

**Go deadline:** Week 26

| Week | Milestone | Output | Paper-ready? |
|------|-----------|--------|-------------|
| 16 | FiD Stage 1: Mark reconstruction >0.40 BLEU | Validated reconstruction model | Draft Stage 1 |
| 18 | FiD Stage 2: Q reconstruction + Thomas triangulation | Q text candidate | Draft Stage 2 |
| 20 | Bayesian: all 4 models fitted + converged | Bayes factors + diagnostics | Results tables |
| 22 | Bayesian: prior sensitivity grid complete | BF contour plots | Sensitivity section |
| 24 | SHAP + Hawkins comparison complete | Feature table, Jaccard similarities | Interpretability section |
| 26 | Full manuscript + figures + supplementary | Complete paper | ✅ YES — submit |

**Negative pivot 1:** If Q reconstruction BLEU < 0.30 (can't reconstruct Mark), drop Q reconstruction from Paper C. The paper is: *"A Bayesian Decision Framework for the Synoptic Problem Using Neural Direction Scores,"* relying only on the direction scorer output + Bayesian comparison + SHAP/Hawkins. Still a complete paper.

**Negative pivot 2:** If Bayesian comparison produces BF < 3 for all pairwise comparisons (inconclusive), paper becomes: *"What Deep Learning Reveals About the Limits of Synoptic Source Criticism."* The contribution is the uncertainty quantification itself — showing that the data cannot distinguish the hypotheses given current sample sizes. This is a methodologically honest and publishable finding.

---

### Phase-to-Milestone Map

```
Week 0-4           Week 4-8         Week 8-14           Week 14-20         Week 20-26
┌─────────┐      ┌─────────┐      ┌─────────┐         ┌─────────┐        ┌─────────┐
│  M0     │      │ Paper A │      │ Paper B │         │ Paper C │        │ Paper C │
│ Data    │─ ─ ─►│Koine-   │      │Direction│─ ─ ─ ─ ─►│Q Recon +│─── ─ ─►│Bayesian │
│ Pipeline│      │Former   │      │Scorer   │  (go)    │FiD      │  (go)  │+ Interp │
└─────────┘      └─────────┘      └─────────┘         └─────────┘        └─────────┘
                     │                 │                                              │
                     │                 │ (no-go: <75%)                               │
                     │                 ▼                                              │
                     │        Publish negative pivot                                 │
                     │                                                                 │
                     ▼                                                                 ▼
            Publish Paper A                                               Submit Paper C
            (~Week 8)                                                     (~Week 26)

Timeline notes:
- Paper A and Paper B can OVERLAP (different encoders, different tasks)
- Paper B does NOT need Paper A's DAPT — Ancient-Greek-BERT works as fallback
- Paper C only proceeds if Paper B accuracy > 75% (enough signal for Bayesian model)
- If Paper B < 75%, Paper C is descoped: just Q reconstruction without Bayesian (still valid)

```

### Go/No-Go Decision Tree

```
Data Pipeline (Week 4)
│
├── GO ─────────────────────────────────► Paper A starts (Week 5)
│                                          │
│     ┌────────────────────────────────────┘
│     │                    ┌─────────────────────────────────────────┐
│     │                    │ Paper B starts WITH Ancient-Greek-BERT  │
│     │                    │ (No need to wait for Paper A. If A is  │
│     │                    │  done by week 8, swap encoders.)       │
│     │                    └─────────────────────────────────────────┘
│     ▼
│  Paper A evaluation (Week 8)
│  │
│  ├── GO (PPL ratio ≤ 1.05) ─────────────► KoineFormer usable for Paper C Q recon
│  │
│  └── NO-GO (PPL ratio > 1.05) ─────────► Paper A published as negative result
│                                          Paper B uses Ancient-Greek-BERT (fine)
│                                          Paper C Q recon limited (no decoder DAPT)
│
│  Paper B evaluation (Week 14)
│  │
│  ├── GO (accuracy ≥ 75%) ──────────────► Paper B submitted
│  │                                       Paper C starts (direction signal exists)
│  │
│  └── NO-GO (accuracy < 75%) ───────────► Paper B published as negative result
│                                          Paper C descoped (Q recon only, no Bayesian)
│
│  Paper C evaluation (Week 26)
│  │
│  ├── GO (BF stable, Q BLEU > 0.30) ────► Paper C submitted
│  │
│  └── PARTIAL (one component works) ────► Publish what works, drop what doesn't
```

This means:

**Worst case:** Data pipeline → Paper A negative pivot + Paper B negative pivot → 2 published negative-result papers. That's a solid year's work: two workshop papers on (1) the difficulty of domain-adapting T5 to low-resource ancient Greek, and (2) why transformers can't detect copying direction in ancient texts.

**Best case:** All 3 papers accepted at good venues. Plus the data + models on GitHub. That's a Master's thesis + 2 conference papers in one year.

**Realistic middle:** Paper A works, Paper B works at 78% (strong enough for Bayesian but not stellar), Paper C descoped to direction + Bayesian only (no Q reconstruction). Still 2 publications + a platform for follow-up work.

---

## 12. Development Workflow

### 12.1 Pre-Commit (Every Commit)

```bash
ruff check synoptiq/ tests/ scripts/     # Must pass
ruff format --check synoptiq/ tests/ scripts/  # Must pass
mypy synoptiq/                           # Must pass
python -m pytest tests/ -q --tb=short    # Must pass
```

### 12.2 Pre-Modal (Before Every GPU Run)

```bash
# Verify everything works locally on CPU first
python scripts/prepare_data.py                    # Data is current
python -m pytest tests/ -m "not gpu" -q --tb=short  # Tests pass
python -c "from synoptiq.models.direction import DirectionScorer; print('imports OK')"
modal run modal/_common.py::check_gpu             # Modal GPU is accessible
```

### 12.3 Git Branch Strategy

```
main
├── phase/0-foundation
├── phase/1-data-pipeline
├── phase/2-koineformer
├── phase/3-direction
├── phase/4-editorial
├── phase/5-reconstruction
├── phase/6-bayesian
└── phase/7-interpretability
```

Merge to main only when phase is complete + all tests pass + go/no-go is GO.

### 12.4 Commit Convention

```
feat: Phase N — <description>
fix: <description>
refactor: <description>
docs: <description>
test: <description>
chore: <description>
```

---

## 13. Complete Subtask Master List

### Phase 0: Foundation (17 subtasks)
```
[ ] Create full directory tree with __init__.py files
[ ] Write synoptiq/_about.py
[ ] Write synoptiq/__init__.py
[ ] Write synoptiq/utils/types_.py
[ ] Write synoptiq/utils/greek.py
[ ] Write synoptiq/utils/constants.py
[ ] Write synoptiq/utils/logging_.py
[ ] Write synoptiq/utils/tokenization.py
[ ] Write synoptiq/utils/io_.py
[ ] Write synoptiq/training/_config.py
[ ] Create configs/data.yaml
[ ] Create configs/model.yaml
[ ] Create configs/training.yaml
[ ] Create configs/bayesian.yaml
[ ] Create configs/modal.yaml
[ ] Write pyproject.toml
[ ] Write Makefile + .gitignore
```

### Phase 1: Data Pipeline (19 subtasks)
```
[ ] Write synoptiq/data/_download.py
[ ] Write synoptiq/data/_parse_sblgnt.py
[ ] Write synoptiq/data/_parse_morphgnt.py
[ ] Write synoptiq/data/_parse_proiel.py
[ ] Write synoptiq/data/_parse_n1904.py
[ ] Write synoptiq/data/_parse_lxx.py
[ ] Write synoptiq/data/_parse_josephus.py
[ ] Write synoptiq/data/_parse_apostolic.py
[ ] Write synoptiq/data/alignment.py
[ ] Write synoptiq/data/pericope.py
[ ] Write synoptiq/data/splits.py
[ ] Write synoptiq/data/augmentation.py
[ ] Write synoptiq/data/corpus.py
[ ] Write scripts/prepare_data.py
[ ] Write scripts/_cli_utils.py
[ ] Write tests/conftest.py (fixtures)
[ ] Write tests/data/test_corpus.py
[ ] Write tests/data/test_alignment.py
[ ] Write tests/data/test_pericope.py + test_splits.py
```

### Phase 2: KoineFormer (17 subtasks)
```
[ ] Write synoptiq/models/_base.py
[ ] Write synoptiq/models/koineformer.py
[ ] Write synoptiq/models/encoder.py
[ ] Write synoptiq/training/_datasets.py
[ ] Write synoptiq/training/_collate.py
[ ] Write synoptiq/training/_trainer.py
[ ] Write synoptiq/training/dapt.py
[ ] Write synoptiq/training/multitask.py
[ ] Write modal/_common.py
[ ] Write modal/app_dapt.py
[ ] Write modal/app_train.py
[ ] Write modal/secrets.toml.example
[ ] Write scripts/train_dapt.py
[ ] Write scripts/train_multitask.py
[ ] Write tests/training/test_datasets.py
[ ] Write tests/training/test_collate.py
[ ] Write tests/models/test_koineformer.py
```

### Phase 3: Direction Scoring (14 subtasks)
```
[ ] Write synoptiq/models/source_detector.py
[ ] Write synoptiq/models/direction.py
[ ] Write synoptiq/training/direction_train.py
[ ] Write synoptiq/evaluation/metrics.py
[ ] Write synoptiq/evaluation/baselines.py
[ ] Write synoptiq/evaluation/calibration.py
[ ] Write scripts/train_source_detector.py
[ ] Write scripts/train_direction.py
[ ] Write scripts/eval_all.py
[ ] Write tests/models/test_direction.py
[ ] Write tests/evaluation/test_metrics.py
[ ] Write tests/evaluation/test_baselines.py
[ ] Write tests/models/test_source_detector.py
[ ] Write modal/app_eval.py
```

### Phase 4: Editorial Tendency (8 subtasks)
```
[ ] Write synoptiq/models/editor.py
[ ] Write synoptiq/training/editor_train.py
[ ] Write scripts/train_editor.py
[ ] Write tests/models/test_editor.py
[ ] Train Mark→Matthew editor
[ ] Train Mark→Luke editor
[ ] Evaluate fatigue detection on Goodacre's examples
[ ] Compare learned tendencies to scholarly descriptions
```

### Phase 5: Q Reconstruction (9 subtasks)
```
[ ] Write synoptiq/models/reconstruction.py
[ ] Write synoptiq/training/reconstruction_train.py
[ ] Write scripts/train_reconstruction.py
[ ] Write tests/models/test_reconstruction.py
[ ] Train Stage 1: Mark reconstruction
[ ] Train Stage 2: Q reconstruction
[ ] Evaluate: Mark reconstruction BLEU
[ ] Evaluate: IQP agreement + Thomas triangulation
[ ] Run consistency + divergence analysis
```

### Phase 6: Bayesian Comparison (11 subtasks)
```
[ ] Write synoptiq/bayesian/_hypotheses.py
[ ] Write synoptiq/bayesian/_likelihood.py
[ ] Write synoptiq/bayesian/models.py
[ ] Write synoptiq/bayesian/bridge.py
[ ] Write synoptiq/bayesian/sensitivity.py
[ ] Write scripts/sample_bayesian.py
[ ] Fit all 4 models
[ ] Compute all pairwise Bayes factors
[ ] Run prior sensitivity grid
[ ] Generate BF contour plots
[ ] Run posterior predictive checks + LOO-CV
```

### Phase 7: Interpretability (12 subtasks)
```
[ ] Write synoptiq/interpretability/shap_analysis.py
[ ] Write synoptiq/interpretability/hawkins.py
[ ] Write synoptiq/interpretability/bertviz_viz.py
[ ] Write scripts/run_interpretability.py
[ ] Compute SHAP for direction scorer
[ ] Compare to Hawkins 1899
[ ] Generate BERTViz attention visualizations
[ ] Run multi-edition sensitivity
[ ] Run counterfactual analysis
[ ] Generate all paper figures
[ ] Write scripts/run_full_pipeline.py (orchestrator)
[ ] Final review + cleanup
```

**Total: 107 subtasks across 8 phases (0-7)**

---

*This implementation plan covers the full SynoptiQ system — KoineFormer DAPT, multi-task fine-tuning, source detection, direction scoring, editorial tendency modeling, fatigue detection, FiD Q reconstruction, Bayesian model comparison, and interpretability — all designed to run within a $30/month Modal free tier budget with strategic GPU allocation.*
