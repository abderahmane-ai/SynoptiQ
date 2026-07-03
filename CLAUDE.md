# CLAUDE.md — SynoptiQ

A multi-task neural source criticism framework for the Synoptic Problem.
Applies transformers, causal direction modeling, and Bayesian inference
to determine the literary relationships among Matthew, Mark, and Luke.

## Project layout

```
SynoptiQ/
├── synoptiq/               # Python package (installable via pip install -e .)
│   ├── data/               # Corpus loading, parsing, alignment, splits
│   ├── models/             # KoineFormer, MultiTaskEncoder, DirectionScorer, Editor, QReconstructor
│   │   ├── koineformer.py  # GreTa + LoRA wrapper, save/load adapters, generate
│   │   └── encoder.py      # Multi-task encoder: POS, biaffine parser, pericope heads
│   ├── training/           # DAPT, multi-task, direction training
│   │   ├── dapt.py         # DAPT data loader (70/30 replay) + training loop with AMP, checkpointing
│   │   └── multitask.py    # Multi-task LoRA fine-tuning (POS dataset + trainer)
│   ├── evaluation/         # Linear probe POS evaluation, baseline comparison
│   ├── bayesian/           # PyMC models, bridge sampling, prior sensitivity
│   ├── interpretability/   # SHAP, Hawkins comparison, BERTViz
│   └── utils/              # Greek text, tokenization, types, constants, logging
├── scripts/                # CLI entry points
│   ├── prepare_data.py     # Phase 1: download → parse → align → split → cache
│   ├── export_hf_dataset.py # Package SynoptiQ Corpus as HuggingFace dataset
│   ├── train_dapt.py       # Phase 2A: KoineFormer DAPT (--smoke-test for quick check)
│   ├── train_multitask.py  # Phase 2B: Multi-task LoRA fine-tuning
│   ├── eval_baseline.py    # Zero-shot vs DAPT evaluation (--zero-shot, --dapt-checkpoint)
│   └── run_ablation.py     # LoRA vs full fine-tune ablation
├── paper/                  # Paper A: KoineFormer (ACL/LaTeCH-CLfL, XeLaTeX)
│   ├── main.tex            # Main manuscript (Poppins + TeX Gyre Pagella, Mediterranean palette)
│   └── references.bib      # BibTeX references
├── modal/                  # Modal GPU deployment
│   └── app_dapt.py         # DAPT training, ablation, full-FT eval — upload, train, monitor
├── configs/                # YAML config files (data, model, training, bayesian, modal)
├── datasets/               # HuggingFace dataset export output (gitignored except README)
│   └── synoptiq-corpus/    # Pushed to ainouche-abderahmane/synoptiq-corpus
├── tests/                  # Mirrors synoptiq/ structure; 84 tests
├── data/                   # Git-ignored: raw corpora, processed Parquet files
├── models/                 # Git-ignored: downloaded HF models, trained adapters
│   └── koineformer/dapt/   # LoRA adapter checkpoints (~14 MB each, 10 checkpoints + final)
├── outputs/                # Git-ignored: logs, eval results, ablation curves
├── SYNOPTIQ_MASTER_PLAN.md  # Research design, architecture, innovation rationale
└── IMPLEMENTATION_PLAN.md   # Phases, subtasks, file lists, budget, timeline
```

## Current status

| Phase | Status | Key result |
|-------|--------|------------|
| Phase 0 | ✓ Foundation | Types, constants, Greek utils, project skeleton |
| Phase 1 | ✓ Data Pipeline | SynoptiQ Corpus: 49,061 tokens, 170 pericopes, 235 alignments, 84 tests |
| Phase 2A | ✓ DAPT | KoineFormer trained: 96.62% POS, 81.34% lemma, $0.35, 14 MB |
| Phase 2B | ○ Multi-task | Code ready, not yet trained |
| Phase 3 | ○ Direction Scorer | Not started |
| Phase 4 | ○ Editorial Drift | Not started |
| Phase 5 | ○ Q Reconstruction | Not started |
| Phase 6 | ○ Bayesian | Not started |
| Phase 7 | ○ Interpretability | Not started |
| Paper A | ✓ Draft | paper/main.tex — complete manuscript, 7/7 refinements applied |

## Key files to know

- `synoptiq/utils/types_.py` — All shared TypedDicts, Literals, Protocols used everywhere
- `synoptiq/utils/constants.py` — **Aland pericope table** (bedrock of the project), MorphGNT tagset maps, Goodacre fatigue pericopes, genre classifications. 1,337 lines, definitional.
- `synoptiq/utils/greek.py` — Greek text normalization (NFD accent stripping, sigma normalization)
- `synoptiq/data/corpus.py` — Central `Corpus` class. Single entry point for all data access.
- `synoptiq/data/alignment.py` — Needleman-Wunsch token alignment via Bio.Align.PairwiseAligner
- `synoptiq/training/_config.py` — Five frozen dataclasses with all training configuration
- `synoptiq/models/koineformer.py` — **KoineFormer**: GreTa + LoRA wrapper, save/load adapters, generate
- `synoptiq/training/dapt.py` — **DAPT**: data loader (70/30 Koine/Classical replay) + training loop with AMP, SIGTERM handler, crash-safe checkpointing
- `synoptiq/evaluation/__init__.py` — **Evaluation**: per-token linear probe for POS accuracy
- `scripts/train_dapt.py` — DAPT CLI: `--smoke-test` (100 steps CPU), full training (20K steps GPU)
- `scripts/eval_baseline.py` — Compare zero-shot GreTa vs DAPT KoineFormer on downstream POS
- `scripts/run_ablation.py` — LoRA vs full fine-tune loss curve comparison
- `modal/app_dapt.py` — Modal GPU deployment: `upload_data`, `start_training`, `run_ablation`, `train_and_eval_full_ft`

## Phase 2A results (Paper A)

### POS + Lemma tagging (linear probe, SynoptiQ test set)

| Model | POS Acc. | Lemma Acc. | Params | Checkpoint | GPU Cost |
|-------|----------|------------|--------|------------|----------|
| GreTa zero-shot | 95.32% | 82.37% | 0 | 880 MB | $0.00 |
| Full fine-tune (220M) | 96.11% | — | 220M | 880 MB | ~$23 |
| **KoineFormer LoRA** | **96.62%** | 81.34% | **3.7M** | **14 MB** | **$0.35** |

Headline: LoRA DAPT eliminates 28% of POS errors vs zero-shot, beats full FT on accuracy,
costs 66× less, and produces a 14 MB checkpoint. Lemma is flat — DAPT improves syntax
but not vocabulary.

### DAPT corpus
- Koine (70%): SBLGNT full NT (~656K tokens) + Apostolic Fathers (~683K tokens) ≈ 1.34M tokens
- Classical replay (30%): First1KGreek (Homer, Plato, Xenophon)
- LXX not yet loaded — TextFabric repo contains converter code but no text data

### Training config
- GreTa (T5-base, 220M) frozen, LoRA r=16 α=32 on W_q, W_v, W_o, W_i, W_o → 3.7M trainable
- 20,000 steps, batch 8, seq_len 512, AMP (FP16), AdamW lr=1e-4, cosine to zero
- A10G GPU, 58 minutes, 10 checkpoints + final, crash-safe with SIGTERM handler

## Modal commands

```bash
# Upload data to Modal volume (once):
modal run modal/app_dapt.py::upload_data

# Train KoineFormer (auto-resumes from checkpoint):
modal run modal/app_dapt.py::start_training

# Monitor training (live logs):
modal app logs synoptiq-dapt

# Ablation — LoRA vs full fine-tune (2K steps, ~10 min):
modal run modal/app_dapt.py::run_ablation

# Full fine-tune DAPT + downstream POS eval (20K steps, ~1 hr):
modal run modal/app_dapt.py::train_and_eval_full_ft

# Download trained adapters:
modal volume get synoptiq-outputs dapt/ models/koineformer/dapt/
```

Modal volume structure:
- `synoptiq-data` — `/data/raw/` (4007 files) + `/data/processed/` (Parquet files)
- `synoptiq-outputs` — `/outputs/dapt/` (10 checkpoints + final), `/outputs/ablation/`, `/outputs/full_ft/`

## HuggingFace dataset

Published at `ainouche-abderahmane/synoptiq-corpus` (CC-BY-SA 4.0):

```python
from datasets import load_dataset
ds = load_dataset("ainouche-abderahmane/synoptiq-corpus")
# ds["train"] → 27,289 tokens, ds["validation"] → 9,170, ds["test"] → 10,618
```

Export/republish: `python3 scripts/export_hf_dataset.py --push --force`

## DAPT data loader behavior

- `DAPTIterableDataset` concatenates short text chunks into full 512-token sequences
- 70% Koine sources (SBLGNT, Apostolic Fathers) interleaved with 30% Classical (First1KGreek)
- Token-level noise: 15% of tokens replaced with mask_token_id=4
- LXX yields 0 chunks (TextFabric `.tf` files are key=value metadata, not Greek text)

## Tokenizer notes

- GreTa SentencePiece has no pad_token or eos_token set — must call `tokenizer.add_special_tokens({"pad_token": "[PAD]"})` and `model.resize_token_embeddings()`
- Koine text tokenizes at 1.38 subwords/word (better than Classical at 1.95) — simpler morphology
- Nomina sacra (Ἰησοῦς, Χριστός, κύριος, θεός) are single tokens — good coverage

## Paper compilation

```bash
# Install TeX (once):
brew install --cask basictex
# Or use Overleaf with XeLaTeX compiler

# Compile:
cd paper
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

Paper uses custom fonts (Poppins, TeX Gyre Pagella, FreeSerif for Greek) and requires XeLaTeX.
Color palette: marine (#123C48), terracotta (#B75B36), papyrus (#F7F2E7).

## Pre-commit checklist (ALWAYS do these)

Before every commit, in this exact order:

```bash
# 1. Lint and auto-fix
python3 -m ruff check synoptiq/ tests/ scripts/ --fix

# 2. Run full test suite
python3 -m pytest tests/ -q --tb=short

# 3. Clean all caches
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# 4. Commit
git add -A
git commit -m "<message>"
```

All tests must pass. Ruff should show zero F or E level errors (TC, UP, RUF, ANN, B90 warnings are cosmetic).

## Architecture summary

**KoineFormer** = GreTa (T5 encoder-decoder, 220M params, Classical Greek) after PEFT-DAPT on Koine corpus (SBLGNT full NT + Apostolic Fathers ~1.34M tokens). LoRA adapters only (~3.7M trainable params, r=16, α=32). 70/30 Koine/Classical replay buffer (First1KGreek). Trained 20K steps, 58 min on A10G, $0.35. 96.62% POS accuracy, 81.34% lemma accuracy. 14 MB checkpoint.

**Direction Scorer** (Phase 3) = Cross-attention asymmetry between parallel passages → 8 asymmetry features → 3-way classification (A→B, B→A, independent). Adversarial GRL head strips authorship style. Trained on triple tradition (known direction: Mark → Matthew, Mark → Luke).

**Editorial Fatigue** (Phase 4) = Position-weighted consistency loss: `L_fatigue = Σ w(i) · D_KL(edit_dist_i || source_dist_i)` where `w(i) = exp(-λ · i/N)`.

**Q Reconstruction** (Phase 5) = Fusion-in-Decoder (FiD): Matthew + Luke encoded independently → concatenated hidden states → decoder with cross-attention. Trained on triple tradition (Matthew+Luke → reconstruct Mark) then transferred to double tradition.

**Bayesian Comparison** (Phase 6) = Direction scorer outputs → MC Dropout (T=20) → (μ_i, σ²_i) per pericope → PyMC Beta models with precision κ_i = 1/σ²_i. Four hypotheses: 2SH, FGH, Augustinian, Griesbach.

**Interpretability** (Phase 7) = SHAP feature importance compared to Hawkins' *Horae Synopticae* (1899). BERTViz attention visualization. Multi-edition sensitivity (NA28 vs TR vs Majority Text vs WH).

## Three-paper strategy

| Paper | Phases | Content | Dependencies |
|-------|--------|---------|-------------|
| **Paper A** | Phase 1+2 | KoineFormer + SynoptiQ Corpus | Self-contained ✓ |
| Paper B | Phase 3+4 | Direction scorer + editorial fatigue | Paper A encoder (or Ancient-Greek-BERT fallback) |
| Paper C | Phase 5+6+7 | Q reconstruction + Bayesian + interpretability | Papers A+B |

Paper A is complete (draft in `paper/main.tex`). Paper B can proceed with Ancient-Greek-BERT as encoder if KoineFormer isn't ready — they're independent.

## Tech stack

- Python 3.12+, PyTorch 2.6+, HuggingFace transformers 4.51+
- PEFT (LoRA adapters), Modal (GPU cloud, A10G, ~$0.45/hr spot)
- BioPython (token alignment), PyMC + ArviZ (Bayesian), SHAP + BERTViz (interpretability)
- ruff (linting), pytest (testing), XeLaTeX (paper)
- Data sources: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), Apostolic Fathers, First1KGreek (CC-BY-SA)
