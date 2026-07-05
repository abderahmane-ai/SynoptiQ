# CLAUDE.md — SynoptiQ

A multi-task neural source criticism framework for the Synoptic Problem.
Applies transformers, causal direction modeling, and Bayesian inference
to determine the literary relationships among Matthew, Mark, and Luke.

## Project layout

```
SynoptiQ/
├── synoptiq/               # Python package (pip install -e .)
│   ├── data/               # Corpus loading, parsing, alignment, splits, augmentation
│   ├── models/             # KoineFormer, DirectionScorer, MultiTaskEncoder
│   │   ├── koineformer.py  # GreTa + LoRA wrapper, save/load adapters, generate
│   │   ├── direction.py    # Phase 3: CrossAttentionAsymmetry, GRL, DirectionScorer
│   │   └── encoder.py      # Multi-task encoder: POS, biaffine parser, pericope heads
│   ├── training/           # DAPT, multi-task, direction training loops
│   │   ├── _config.py      # Frozen dataclasses: ModelConfig, TrainingConfig, et al.
│   │   ├── dapt.py         # DAPT data loader (70/30 replay) + training loop
│   │   ├── direction.py    # Phase 3: DirectionDataset, DirectionTrainer
│   │   └── multitask.py    # Multi-task LoRA fine-tuning (POS dataset + trainer)
│   ├── evaluation/         # Linear probe POS/lemma, direction metrics
│   ├── bayesian/           # PyMC models, bridge sampling, prior sensitivity
│   ├── interpretability/   # SHAP, Hawkins comparison, BERTViz
│   └── utils/              # Greek text, tokenization, types, constants, logging
├── scripts/                # CLI entry points
│   ├── prepare_data.py     # Phase 1: download → parse → align → split → cache
│   ├── export_hf_dataset.py # Package SynoptiQ Corpus as HuggingFace dataset
│   ├── train_dapt.py       # Phase 2A: KoineFormer DAPT (--smoke-test for quick check)
│   ├── train_multitask.py  # Phase 2B: Multi-task LoRA fine-tuning
│   ├── train_direction.py  # Phase 3: Direction scorer (--smoke-test, full GPU training)
│   ├── eval_baseline.py    # Zero-shot vs DAPT evaluation (--zero-shot, --dapt-checkpoint)
│   └── run_ablation.py     # LoRA vs full fine-tune ablation
├── paper/                  # Paper A: KoineFormer (XeLaTeX)
│   ├── main.tex            # Main manuscript (Poppins + TeX Gyre Pagella, Mediterranean)
│   ├── project_overview.tex # SynoptiQ project brief for Yale/Oxford/Harvard audience
│   └── references.bib      # BibTeX references (6 entries)
├── modal/                  # Modal GPU deployment
│   ├── app_dapt.py         # Phase 2A: DAPT training, ablation, full-FT eval
│   └── app_direction.py    # Phase 3: Direction scorer training (T4 GPU)
├── datasets/               # HuggingFace dataset export (gitignored except README)
│   └── synoptiq-corpus/    # Pushed to ainouche-abderahmane/synoptiq-corpus
├── tests/                  # 84 tests (mirrors synoptiq/ structure)
├── data/                   # Git-ignored: raw corpora, processed Parquet files
├── models/                 # Git-ignored: downloaded HF models, trained adapters
├── outputs/                # Git-ignored: logs, eval results, checkpoints
├── PROJECT_OVERVIEW.md      # Project overview (Markdown)
├── SYNOPTIQ_MASTER_PLAN.md  # Research design, architecture, innovation rationale
└── IMPLEMENTATION_PLAN.md   # Phases, subtasks, file lists, budget, timeline
```

## Current status

| Phase | Status | Key result |
|-------|--------|------------|
| Phase 0 | ✓ Foundation | Types, constants, Greek utils, project skeleton |
| Phase 1 | ✓ Data Pipeline | SynoptiQ Corpus: 49,061 tokens, 170 pericopes, 235 alignments, 84 tests |
| Phase 2A | ✓ DAPT | KoineFormer trained: 96.62% POS, 81.34% lemma, 14 MB |
| Phase 2B | ○ Multi-task | Code ready, not yet trained |
| Phase 3 | ● Direction Scorer | Code ready, smoke-test passed, awaiting Modal GPU training |
| Phase 4 | ○ Editorial Drift | Not started |
| Phase 5 | ○ Q Reconstruction | Not started |
| Phase 6 | ○ Bayesian | Not started |
| Phase 7 | ○ Interpretability | Not started |
| Paper A | ✓ Draft | paper/main.tex — complete manuscript, verified numbers |
| Paper B | ○ Design | Architecture designed, core code implemented |

## Key files to know

- `synoptiq/utils/types_.py` — All shared TypedDicts (`Direction`, `DirectionScores` already defined), Literals, Protocols
- `synoptiq/utils/constants.py` — **Aland pericope table** (bedrock), MorphGNT tagset maps, Goodacre fatigue pericopes, 1,337 lines
- `synoptiq/utils/greek.py` — Greek text normalization (NFD accent stripping, sigma normalization)
- `synoptiq/data/corpus.py` — Central `Corpus` class. `direction_pairs(split=...)` yields (book_a, tokens_a, book_b, tokens_b, alignment) for direction scorer training
- `synoptiq/data/alignment.py` — Needleman-Wunsch token alignment via Bio.Align.PairwiseAligner
- `synoptiq/data/augmentation.py` — Bootstrap resampling, sliding windows, scribal noise injection
- `synoptiq/training/_config.py` — Five frozen dataclasses: `ModelConfig` already has `direction_num_classes`, `cross_attn_num_heads`, `asymmetry_num_features`, `lambda_adversarial`
- `synoptiq/models/koineformer.py` — **KoineFormer**: GreTa + LoRA wrapper, factory, save/load, generate
- `synoptiq/models/direction.py` — **DirectionScorer** (Phase 3): CrossAttentionAsymmetry, GradientReversalLayer, DirectionClassifier, AuthorDiscriminator
- `synoptiq/training/dapt.py` — **DAPT**: data loader + training loop with AMP, SIGTERM handler, crash-safe checkpointing
- `synoptiq/training/direction.py` — **Phase 3 training**: DirectionDataset (wraps Corpus.direction_pairs), DirectionTrainer (AMP, GRL annealing, checkpointing)
- `synoptiq/evaluation/__init__.py` — Linear probe evaluation: POS + lemma accuracy
- `scripts/train_dapt.py` — DAPT CLI: `--smoke-test` (100 steps CPU), full training (20K steps GPU)
- `scripts/train_direction.py` — Direction scorer CLI: `--smoke-test` (100 steps CPU), full training (5K steps GPU)
- `scripts/eval_baseline.py` — Compare zero-shot GreTa vs DAPT KoineFormer on POS + lemma
- `scripts/run_ablation.py` — LoRA vs full fine-tune loss curve comparison
- `modal/app_dapt.py` — Phase 2A Modal: `upload_data`, `start_training`, `run_ablation`, `train_and_eval_full_ft`
- `modal/app_direction.py` — Phase 3 Modal: `upload_data`, `start_training` (T4, 5K steps), `smoke_test`

## Phase 2A results (Paper A)

### POS + Lemma tagging (linear probe, SynoptiQ test set)

| Model | POS Acc. | Lemma Acc. | Params | Checkpoint |
|-------|----------|------------|--------|------------|
| GreTa zero-shot | 95.32% | 82.37% | 0 | 880 MB |
| Full fine-tune (220M) | 96.11% | — | 220M | 880 MB |
| **KoineFormer LoRA** | **96.62%** | 81.34% | **3.7M** | **14 MB** |

Headline: LoRA DAPT eliminates 28% of POS errors vs zero-shot, beats full FT on accuracy,
and produces a 14 MB checkpoint. Lemma is flat — DAPT improves syntax
but not vocabulary.

### DAPT corpus
- Koine (70%): SBLGNT full NT (~773K tokens) + Apostolic Fathers (~732K tokens) ≈ 1.5M tokens
- Classical replay (30%): First1KGreek (Homer, Plato, Xenophon)
- LXX yields 0 chunks (TextFabric `.tf` files are metadata, not Greek text)

### Training config
- GreTa (T5-base, 220M) frozen, LoRA r=16 α=32 targeting `["q", "v", "o", "wi", "wo"]`
- Note: `wi` does NOT actually apply (PEFT uses `endswith` matching; `wi` doesn't match `wi_0`/`wi_1`)
- Actual LoRA targets: W_q, W_v, W_o (attention) + W_o (FFN output) → 3.7M trainable
- Breakdown: attention 2.65M (q+v+o, self+cross) + FFN 1.08M (wo only)
- 20,000 steps, batch 8, seq_len 512, AMP (FP16), AdamW lr=1e-4, cosine to zero
- A10G GPU, 58 minutes, 10 checkpoints + final, crash-safe with SIGTERM handler

## Phase 3: Direction Scorer

### Architecture (v2 — GRL removed)

v1 included a GradientReversalLayer + AuthorDiscriminator for style invariance.
Diagnostic experiments showed the GRL at λ=1.0 destroyed cross-attention gradients:
features 1,2,8 were constant (flat attention maps), the model collapsed to always
predicting "independent" (32.6% accuracy vs 33.3% random). v2 removes the adversarial
component entirely.

```
Pericope pair (A, B) aligned tokens
  → Tokenize each passage → KoineFormer encoder (frozen, DAPT adapters loaded)
  → H_A [L_A, 768], H_B [L_B, 768]
  → CrossAttentionAsymmetry (trainable multi-head cross-attention):
      A→B cross-attn + B→A cross-attn → 8 asymmetry features + pooled states
  → [pooled_A, pooled_B, asym_features] → DirectionClassifier → 3-way softmax
```

### 8 asymmetry features
1-2: mean attention A→B, B→A
3-4: variance A→B, B→A
5-6: entropy A→B, B→A
7: KL-asymmetry (KL(A→B ‖ B→A) - KL(B→A ‖ A→B))
8: position-decay (avg attended position normalized by sequence length)

### Training data
- 65 labeled triple-tradition pericopes (39 train / 12 val / 14 test)
- Labels: (Mark, Matthew) → A_to_B, (Mark, Luke) → A_to_B, (Matthew, Luke) → independent
- Swap augmentation doubles samples (A↔B with flipped label)
- 250 train / 86 val / 92 test samples after augmentation
- Pericope-level split prevents data leakage

### Baselines (built)
- Random: 33.3%, Majority class: 33.7%
- Cosine similarity: 33.7% (direction not detectable from embedding similarity)
- Logistic regression on pooled KoineFormer embeddings: **72.8%**
- Gate: direction scorer must beat 72.8% to justify cross-attention complexity

### Training config
- KoineFormer encoder frozen (DAPT adapters loaded); cross-attention + classifier trained
- Batch 16, 5,000 steps, AdamW lr=1e-4, cosine to zero
- Loss = CrossEntropy(direction) only (no adversarial term)
- AMP (FP16) on CUDA only; T4 GPU, ~15 min
- Same crash-safe checkpointing as DAPT

### Key gotchas
- DAPT adapters MUST be loaded — zero-shot GreTa baseline drops from 72.8% to 60.9%
- Local DAPT path: `models/koineformer/dapt/final/`; Modal: `/outputs/dapt/final/`
- AMP/GradScaler only activates on CUDA (not MPS/CPU)
- GRL removed in v2 — don't add it back without testing on this data size
- GRL is wired correctly: pooled states → GRL → discriminator; classifier sees original (non-GRL'd) states
- Matthew↔Luke pairs labeled "independent" under 2SH — critical negative examples
- Direction types (`A_to_B`, `B_to_A`, `independent`) defined in `types_.py`

## Modal commands

```bash
# ── Phase 2A: DAPT ──────────────────────────────────────────────────
# Upload data (once):
modal run modal/app_dapt.py::upload_data

# Train KoineFormer (auto-resumes):
modal run modal/app_dapt.py::start_training
modal app logs synoptiq-dapt

# Ablation:
modal run modal/app_dapt.py::run_ablation

# Download adapters:
modal volume get synoptiq-outputs dapt/ models/koineformer/dapt/

# ── Phase 3: Direction Scorer ───────────────────────────────────────
# Upload data (once, reuses DAPT volumes):
modal run modal/app_direction.py::upload_data

# Smoke test (100 steps, T4):
modal run modal/app_direction.py::smoke_test

# Full training (5K steps, T4, ~30 min):
modal run modal/app_direction.py::start_training
modal app logs synoptiq-direction

# Download checkpoints:
modal volume get synoptiq-outputs direction/ outputs/direction/
```

Modal volume structure:
- `synoptiq-data` — `/data/raw/` (4007 files) + `/data/processed/` (Parquet files)
- `synoptiq-outputs` — `/outputs/dapt/` (10 checkpoints + final), `/outputs/direction/`, `/outputs/ablation/`, `/outputs/full_ft/`

## HuggingFace

### Model: ainouche-abderahmane/koineformer
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koineformer").merge_and_unload()
```
CC-BY-SA 4.0. 96.62% POS, 81.34% lemma, 14 MB adapter.

### Dataset: ainouche-abderahmane/synoptiq-corpus
```python
from datasets import load_dataset
ds = load_dataset("ainouche-abderahmane/synoptiq-corpus")
# ds["train"] → 27,289 tokens, ds["validation"] → 9,170, ds["test"] → 10,618
```
CC-BY-SA 4.0. 49,061 tokens, 170 pericopes, 235 alignments.

### GitHub: github.com/abderahmane-ai/SynoptiQ

## Tokenizer notes

- GreTa SentencePiece has no pad_token or eos_token — always call:
  `tokenizer.add_special_tokens({"pad_token": "[PAD]"})` then `model.resize_token_embeddings(len(tokenizer))`
- Koine text tokenizes at 1.38 subwords/word (Classical: 1.95) — simpler morphology
- Nomina sacra (Ἰησοῦς, Χριστός, κύριος, θεός) are single tokens
- Subword-to-word alignment uses `▁` prefix (U+2581) for SentencePiece word boundaries

## Paper compilation

```bash
# Compile Paper A:
cd paper
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex

# Compile Project Overview:
xelatex project_overview.tex
```

Paper uses custom fonts (Poppins, TeX Gyre Pagella, FreeSerif for Greek) and XeLaTeX.
Palette: marine (#123C48), terracotta (#B75B36), papyrus (#F7F2E7).
Uses Overleaf if local TeX is unavailable.

## Pre-commit checklist (ALWAYS do these)

```bash
# 1. Lint — must show zero F or E level errors
python3 -m ruff check synoptiq/ tests/ scripts/ --fix

# 2. Full test suite — all 84 must pass
python3 -m pytest tests/ -q --tb=short

# 3. Clean all caches
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# 4. Commit
git add -A && git commit -m "<message>"
```

Ruff TC, UP, RUF, ANN, B90 warnings are cosmetic — only fix F and E.

## Architecture summary

**KoineFormer** = GreTa (T5 encoder-decoder, 220M params, Classical Greek) after
PEFT-DAPT on Koine corpus (SBLGNT + Apostolic Fathers ~1.5M tokens). LoRA adapters
only (~3.7M trainable, r=16 α=32, targeting W_q/W_v/W_o in attention + W_o in FFN).
70/30 Koine/Classical replay buffer. 20K steps, 58 min on A10G. 96.62% POS,
81.34% lemma. 14 MB checkpoint.

**Direction Scorer** (Phase 3, v2) = Frozen KoineFormer encoder (DAPT) →
bidirectional cross-attention → 8 asymmetry features + pooled states →
3-way classifier (A→B, B→A, independent). No GRL (v1 experiment: GRL
destroyed gradients, model collapsed). Trained on 65 labeled triple-tradition
pericopes. Baseline: logistic regression on same encoder = 72.8%. 5K steps T4.

**Editorial Fatigue** (Phase 4) = Position-weighted KL divergence between
editing distribution and source distribution, with exponential decay weight.
Detects copying author reverting to own style over the course of a pericope.

**Q Reconstruction** (Phase 5) = Fusion-in-Decoder: Matthew + Luke encoded
independently → concatenated hidden states → decoder cross-attention. Trained
on triple tradition (Mt+Lk→reconstruct Mark) then transferred to double tradition.

**Bayesian Comparison** (Phase 6) = MC Dropout uncertainty per pericope →
PyMC Beta hierarchical models → bridge sampling Bayes factors. Four hypotheses:
Two-Source (2SH), Farrer–Goulder (FGH), Augustinian, Griesbach.

**Interpretability** (Phase 7) = SHAP feature importance vs Hawkins (1899).
BERTViz attention visualization. Multi-edition sensitivity (NA28/TR/Majority/WH).

## Three-paper strategy

| Paper | Phases | Content | Dependencies |
|-------|--------|---------|-------------|
| **Paper A** | Phase 1+2 | KoineFormer + SynoptiQ Corpus | Self-contained ✓ |
| Paper B | Phase 3+4 | Direction scorer + editorial fatigue | Paper A encoder |
| Paper C | Phase 5+6+7 | Q reconstruction + Bayesian + interpretability | Papers A+B |

Paper A is complete (paper/main.tex). Paper B architecture implemented (Phase 3 code).

## Tech stack

- Python 3.12+, PyTorch 2.6+, HuggingFace transformers 4.51+, PEFT
- Modal (GPU cloud: A10G for DAPT, T4 for direction scorer)
- BioPython (token alignment), PyMC + ArviZ (Bayesian), SHAP + BERTViz
- ruff (linting), pytest (84 tests), XeLaTeX (paper)
- Data: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), Apostolic Fathers, First1KGreek (CC-BY-SA)
- GitHub: [abderahmane-ai/SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ)
- HF model: [ainouche-abderahmane/koineformer](https://huggingface.co/ainouche-abderahmane/koineformer)
- HF dataset: [ainouche-abderahmane/synoptiq-corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus)
