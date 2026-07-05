# CLAUDE.md — SynoptiQ

A multi-task neural source criticism framework for the Synoptic Problem.
Applies transformers, causal direction modeling, and Bayesian inference
to determine the literary relationships among Matthew, Mark, and Luke.

## graphify (codebase knowledge graph)

This project has a knowledge graph at `graphify-out/` (git-ignored — generated
locally, never committed). It covers `synoptiq/`, `scripts/`, and `modal/`:
664 nodes, 1246 edges, 39 communities. God nodes: `Corpus`, `KoineFormer`,
`DirectionScorer`, `DirectionDataset`, `TokenRecord`.

When the user types `/graphify`, invoke the `graphify` skill before anything else.

Rules:
- For any codebase question, run `graphify query "<question>"` first when
  `graphify-out/graph.json` exists — it returns a scoped subgraph, usually far
  smaller than grep or reading `GRAPH_REPORT.md`. Use `graphify path "<A>" "<B>"`
  for relationships between two symbols and `graphify explain "<concept>"` for a
  focused node.
- Prefer the graph over blind file browsing when tracing call paths, data flow,
  or cross-module dependencies.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review when
  query/path/explain don't surface enough.
- After modifying code, run `graphify update .` to keep the graph current
  (AST-only, no API cost). Dirty `graphify-out/` files are expected and are not a
  reason to skip graphify.

## Session handoff (read me first)

**Last commit:** `chore: clean up dead code, unify device detection, honest
docstrings` — a conservative full-package cleanup. Removed genuinely dead code
(no tested/public API touched), corrected misleading docstrings, consolidated
the 6 duplicated `detect_device()` copies onto `scripts/_cli_utils.detect_device`,
and rewrote `alignment.py` docs to match its actual binary (lemma, POS) matching.
Tree is clean; ruff passes; all 87 tests pass.

**Current focus → the Direction Scorer (Phase 3).** It is *not* simply "awaiting
GPU training" — it is a genuine research plateau. Recent git history is almost
entirely direction-scorer churn (GRL added then removed, swap-equivariance added,
encoder made deterministic). The open question: **do the 10 asymmetry features
carry copying-*direction* signal, or only authorship/style signal?** The scorer
must beat the **72.8%** pooled-embedding logistic-regression gate to justify the
asymmetry probe at all.

**Next concrete step:** run `python scripts/diagnose_direction.py` (CPU, no GPU/Modal
needed; needs DAPT adapters at `models/koineformer/dapt/final/`) and read its 5
experiments *before* spending more GPU time. Experiments 1 (asymmetry-only LR) and
4 (author decodability) directly answer the signal-vs-style question. See the
"Phase 3" section and its "Key gotchas" below for what's already been tried and ruled out.

## Project layout

```
SynoptiQ/
├── synoptiq/               # Python package (pip install -e .)
│   ├── data/               # Corpus loading, parsing, alignment, splits, augmentation
│   ├── models/             # KoineFormer, DirectionScorer, MultiTaskEncoder
│   │   ├── koineformer.py  # GreTa + LoRA wrapper, save/load adapters, generate
│   │   ├── direction.py    # Phase 3: 10-feature swap-equivariant DirectionScorer
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
│   ├── run_ablation.py     # LoRA vs full fine-tune ablation
│   └── diagnose_direction.py # Phase 3: 5 CPU experiments diagnosing the direction plateau
├── paper/                  # Paper A: KoineFormer (XeLaTeX)
│   ├── main.tex            # Main manuscript (Poppins + TeX Gyre Pagella, Mediterranean)
│   ├── project_overview.tex # SynoptiQ project brief for Yale/Oxford/Harvard audience
│   └── references.bib      # BibTeX references (6 entries)
├── modal/                  # Modal GPU deployment
│   ├── app_dapt.py         # Phase 2A: DAPT training, ablation, full-FT eval
│   └── app_direction.py    # Phase 3: Direction scorer training (T4 GPU)
├── datasets/               # HuggingFace dataset export (gitignored except README)
│   └── synoptiq-corpus/    # Pushed to ainouche-abderahmane/synoptiq-corpus
├── tests/                  # 87 tests (mirrors synoptiq/ structure)
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
| Phase 1 | ✓ Data Pipeline | SynoptiQ Corpus: 49,061 tokens, 170 pericopes, 235 alignments, 87 tests |
| Phase 2A | ✓ DAPT | KoineFormer trained: 96.62% POS, 81.34% lemma, 14 MB |
| Phase 2B | ○ Multi-task | Code ready, not yet trained |
| Phase 3 | ● Direction Scorer | Code ready + swap-equivariant; **plateau under investigation** (must beat 72.8% pooled-LR gate — see `diagnose_direction.py`) |
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
- `synoptiq/data/alignment.py` — Needleman-Wunsch token alignment via Bio.Align.PairwiseAligner. Scoring is **binary on the (normalized lemma, POS) key** (each pair → a Private-Use-Area char; identical = `match` +2.5, else `mismatch` -100, forcing a gap). Surface form is reported by `alignment_score` but does not steer the path.
- `synoptiq/data/augmentation.py` — Bootstrap resampling, sliding windows, scribal noise injection
- `synoptiq/training/_config.py` — Five frozen dataclasses: `ModelConfig` already has `direction_num_classes`, `asymmetry_num_features`, `direction_signed_features`, `direction_independence_features`
- `synoptiq/models/koineformer.py` — **KoineFormer**: GreTa + LoRA wrapper, factory, save/load, generate
- `synoptiq/models/direction.py` — **DirectionScorer** (Phase 3): frozen encoder → 10 asymmetry features → swap-equivariant classifier
- `synoptiq/training/dapt.py` — **DAPT**: data loader + training loop with AMP, SIGTERM handler, crash-safe checkpointing
- `synoptiq/training/direction.py` — **Phase 3 training**: DirectionDataset (wraps Corpus.direction_pairs), DirectionTrainer (AMP, feature calibration, checkpointing)
- `synoptiq/evaluation/__init__.py` — Linear probe evaluation: POS + lemma accuracy
- `scripts/train_dapt.py` — DAPT CLI: `--smoke-test` (100 steps CPU), full training (20K steps GPU)
- `scripts/train_direction.py` — Direction scorer CLI: `--smoke-test` (100 steps CPU), full training (5K steps GPU)
- `scripts/eval_baseline.py` — Compare zero-shot GreTa vs DAPT KoineFormer on POS + lemma
- `scripts/run_ablation.py` — LoRA vs full fine-tune loss curve comparison
- `scripts/diagnose_direction.py` — **Phase 3 debugging**: 5 CPU experiments probing why the direction scorer plateaus (asymmetry-only LR, confusion matrix, feature↔label correlation, author decodability, pooled-only). Run this before more GPU training.
- `scripts/_cli_utils.py` — Shared CLI helpers. **Canonical `detect_device()`** lives here; all training/eval scripts import it (do not re-add per-script copies).
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

### Architecture

The current Direction Scorer is a frozen-encoder, feature-calibrated,
swap-equivariant classifier. It extracts 10 deterministic asymmetry features
from KoineFormer token states and emits logits `[d, -d, i]`, so swapping A/B
swaps `A_to_B` and `B_to_A` while leaving `independent` invariant.

```
Pericope pair (A, B) aligned tokens
  → Tokenize each passage → KoineFormer encoder (frozen, DAPT adapters loaded)
  → H_A [L_A, 768], H_B [L_B, 768]
  → Compute cross-similarity matrix S = cos(H_A, H_B)
  → Extract 10 asymmetry features (e.g. BERTScore P/R, entropy, diagonal, length ratio)
  → Train-split z-score calibration
  → DirectionClassifier logits [d, -d, i] → 3-way direction probabilities
```

### 10 asymmetry features
1. BERTScore Recall R(A→B) (mean of row maxes)
2. BERTScore Precision P(B→A) (mean of col maxes)
3. Precision-Recall asymmetry (P - R: >0 means B copies A)
4. Standard deviation of row maxes
5. Standard deviation of col maxes
6. Attention-entropy asymmetry
7. Diagonal alignment strength (order-preserving copy signal)
8. Log length ratio log(len_B / len_A)
9. Symmetric pooled-embedding similarity
10. Coverage asymmetry (fraction of close matches difference)

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
- Gate: direction scorer must beat 72.8% to justify the asymmetry probe over pooled embeddings

### Training config
- KoineFormer encoder frozen (DAPT adapters loaded); only the 17-parameter classifier is trained
- Batch 16, 5,000 steps, AdamW lr=1e-3, cosine to zero
- Loss = CrossEntropy(direction)
- AMP (FP16) on CUDA only; T4 GPU, ~15 min
- Same crash-safe checkpointing as DAPT

### Key gotchas
- DAPT adapters MUST be loaded — zero-shot GreTa baseline drops from 72.8% to 60.9%
- Local DAPT path: `models/koineformer/dapt/final/`; Modal: `/outputs/dapt/final/`
- AMP/GradScaler only activates on CUDA (not MPS/CPU)
- The classifier must preserve swap equivariance: logits for swapped inputs are `[B_to_A, A_to_B, independent]`
- Do not use per-sample LayerNorm on the 10 features; it destroys absolute/sign geometry
- Matthew↔Luke pairs labeled "independent" under 2SH — critical negative examples
- Direction types (`A_to_B`, `B_to_A`, `independent`) defined in `types_.py`
- **GRL (gradient reversal) was removed** — it destroyed cross-attention gradients and hurt the encoder (commit `eb5a0a5`). Do not reintroduce an adversarial author-discriminator head without solving that first.
- **The direction encoder is kept deterministic** (commit `b856943`): dropout off / eval-mode feature extraction, so the 10 asymmetry features are stable across runs. Keep it that way when debugging.
- Open question the plateau hinges on: do the 10 asymmetry features carry *direction* signal, or only *authorship/style* signal? That is exactly what `diagnose_direction.py` experiments 1 & 4 test.

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

# 2. Full test suite — all 87 must pass
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

**Direction Scorer** (Phase 3) = Frozen KoineFormer encoder (DAPT) →
cos-similarity matrix → 10 asymmetry features (P-R, entropy, alignment path) →
swap-equivariant classifier with logits `[d, -d, i]` (A→B, B→A, independent).
Trained on 65 labeled triple-tradition pericopes. Baseline: logistic regression
on pooled encoder states = 72.8%. 5K steps T4.

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
- ruff (linting), pytest (87 tests), XeLaTeX (paper)
- Data: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), Apostolic Fathers, First1KGreek (CC-BY-SA)
- GitHub: [abderahmane-ai/SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ)
- HF model: [ainouche-abderahmane/koineformer](https://huggingface.co/ainouche-abderahmane/koineformer)
- HF dataset: [ainouche-abderahmane/synoptiq-corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus)
