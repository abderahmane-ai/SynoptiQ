# CLAUDE.md — SynoptiQ

A neural source-criticism framework for the Synoptic Problem. Applies transformers
(KoineFormer, a DAPT'd Koine-Greek T5) to the Gospels of Matthew, Mark, and Luke:
a curated parallel corpus + representation learning (Paper A, done), heading toward
Q reconstruction (Phase 5).

> **Copying-direction detection is a CLOSED NEGATIVE RESULT.** Phases 3 (direction scorer)
> and 6 (Bayesian hypothesis comparison) were investigated at length and **removed** on
> 2026-07-07 — inferring the direction of literary copying from the texts alone is not
> achievable (it is mathematically isomorphic to distinguishing a lossy projection from its
> inverse). Read `docs/DIRECTION_NEGATIVE_RESULT.md` before ever proposing anything about
> copying direction, RPM, editorial fatigue as a direction signal, or "scoring" the four source
> hypotheses. Do not re-implement them. Recover detail from git history if needed for a write-up.

## graphify (codebase knowledge graph)

This project has a knowledge graph at `graphify-out/` (git-ignored — generated locally, never
committed). It covers `synoptiq/`, `scripts/`, and `modal/`. God nodes: `Corpus`, `KoineFormer`,
`TokenRecord`. When the user types `/graphify`, invoke the `graphify` skill before anything else.
The graph is stale after the 2026-07-07 direction cleanup — run `graphify update .` to refresh.

Rules:
- For any codebase question, run `graphify query "<question>"` first when `graphify-out/graph.json`
  exists — it returns a scoped subgraph, usually far smaller than grep. Use `graphify path "<A>"
  "<B>"` for relationships and `graphify explain "<concept>"` for a focused node.
- Prefer the graph over blind file browsing when tracing call paths or cross-module dependencies.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review.
- After modifying code, run `graphify update .` (AST-only, no API cost).

## Session handoff (read me first)

**Direction + hypotheses code removed (2026-07-07); tree dirty (uncommitted); ruff F/E clean.**
The whole Phase-3 direction scorer and Phase-6 Bayesian comparison were deleted after the
investigation closed as a negative result (`docs/DIRECTION_NEGATIVE_RESULT.md`). Removed:
`synoptiq/direction/`, `synoptiq/bayesian/`, `synoptiq/legacy/`, `synoptiq/data/external_pairs.py`,
`frequency.py`, `augmentation.py`; all direction/hypothesis scripts + tests; `paper_b/`;
`modal/legacy/`; the `Direction`/`DirectionScores`/`EditorialFatigueScores`/`HypothesisSpec` types;
`BayesianConfig`; the direction fields on `ModelConfig`; the Goodacre/minor-agreement constants.
`Corpus.direction_pairs` → renamed **`Corpus.parallel_pairs`** (kept — generic parallel-pair
iteration, useful for Q reconstruction).

**What stands:** KoineFormer + the SynoptiQ corpus (Paper A) — self-contained and done. The
`evaluation/` linear probes (POS/lemma) and `evaluation/bootstrap.py` (pericope-grouped CIs) remain.

**Next:** **Phase 5 — Q reconstruction** (Fusion-in-Decoder): encode Matthew + Luke, reconstruct
Mark on the triple tradition (ground-truth eval via `Corpus.parallel_pairs(tradition="triple")`),
then transfer to the double tradition to reconstruct proto-Q. A genuine generative task that does
not depend on solving direction. `ModelConfig` already carries the `fid_*` fields.

## Cold start (fresh clone / new machine)

`data/`, `models/`, and `graphify-out/` are git-ignored, so a fresh clone has none of them.
Regenerate in order (skip any step whose output already exists locally):

```bash
# 1. Install the package (Python 3.12+)
pip install -e .

# 2. Regenerate the corpus → data/processed/{tokens,pericopes}.parquet
python scripts/prepare_data.py --validate

# 3. Get DAPT adapters → models/koineformer/dapt/final/ (KoineFormer encoder)
#    Option A (Modal):
modal volume get synoptiq-outputs dapt/ models/koineformer/dapt/
#    Option B (HuggingFace, no Modal):
python -c "from huggingface_hub import snapshot_download; snapshot_download('ainouche-abderahmane/koineformer', local_dir='models/koineformer/dapt/final')"

# 4. Verify everything is wired up (all tests must pass)
python -m pytest tests/ -q

# 5. Rebuild the knowledge graph (git-ignored) — invoke the `graphify` skill.
```

## Project layout

```
SynoptiQ/
├── synoptiq/               # Python package (pip install -e .)
│   ├── data/               # Corpus loading, parsing, alignment, splits, pericope classification
│   ├── models/             # KoineFormer, MultiTaskEncoder
│   │   ├── koineformer.py  # GreTa + LoRA wrapper, save/load adapters, generate
│   │   └── encoder.py      # Multi-task encoder: POS, biaffine parser, pericope heads
│   ├── training/           # DAPT + multi-task training loops (_config, dapt, multitask)
│   ├── evaluation/         # Linear probe POS/lemma; bootstrap.py (pericope-grouped CIs)
│   ├── interpretability/   # SHAP, Hawkins comparison, BERTViz
│   └── utils/              # Greek text, tokenization, shared types, constants, logging
├── scripts/                # CLI entry points
│   ├── prepare_data.py     # Phase 1: download → parse → align → split → cache
│   ├── export_hf_dataset.py # Package SynoptiQ Corpus as a HuggingFace dataset
│   ├── train_dapt.py       # Phase 2A: KoineFormer DAPT (--smoke-test for quick check)
│   ├── train_multitask.py  # Phase 2B: Multi-task LoRA fine-tuning
│   ├── eval_baseline.py    # Zero-shot vs DAPT evaluation (--zero-shot, --dapt-checkpoint)
│   ├── run_ablation.py     # LoRA vs full fine-tune ablation
│   └── _cli_utils.py       # Shared CLI helpers (canonical detect_device())
├── paper/                  # Paper A: KoineFormer (XeLaTeX)
├── modal/                  # Modal GPU deployment (app_dapt.py: DAPT, ablation, full-FT eval)
├── datasets/               # HuggingFace dataset export (gitignored except README)
├── tests/                  # pytest suite (mirrors synoptiq/ structure)
├── data/ models/ outputs/  # All git-ignored (corpora, adapters, logs)
├── docs/DIRECTION_NEGATIVE_RESULT.md  # why copying-direction detection was closed
├── PROJECT_OVERVIEW.md · SYNOPTIQ_MASTER_PLAN.md · IMPLEMENTATION_PLAN.md
```

## Current status

| Phase | Status | Key result |
|-------|--------|------------|
| Phase 0 | ✓ Foundation | Types, constants, Greek utils, project skeleton |
| Phase 1 | ✓ Data Pipeline | SynoptiQ Corpus: 49,061 tokens, 170 pericopes, 235 alignments |
| Phase 2A | ✓ DAPT | KoineFormer trained: 96.62% POS, 81.34% lemma, 14 MB |
| Phase 2B | ○ Multi-task | Code ready, not yet trained |
| Phase 3 | ✗ Removed | Direction detection — **closed negative result** (`docs/DIRECTION_NEGATIVE_RESULT.md`) |
| Phase 5 | ○ Q Reconstruction | **Next.** Fusion-in-Decoder (Mt+Lk → Mark, then → proto-Q) |
| Phase 6 | ✗ Removed | Bayesian hypothesis "scoring" — removed with Phase 3 (depended on it) |
| Phase 7 | ○ Interpretability | Not started |
| Paper A | ✓ Draft | paper/main.tex — complete manuscript, verified numbers |

## Key files to know

- `synoptiq/utils/types_.py` — shared TypedDicts (`TokenRecord`, `PericopeAlignment`, `MorphRecord`,
  `SplitResult`), `Book`/`Tradition`/`Genre` literals, Protocols
- `synoptiq/utils/constants.py` — **Aland pericope table** (bedrock), MorphGNT tagset maps
- `synoptiq/utils/greek.py` — Greek text normalization (NFD accent stripping, sigma normalization)
- `synoptiq/data/corpus.py` — central `Corpus` class. `parallel_pairs(tradition=…, split=…)` yields
  `(book_a, tokens_a, book_b, tokens_b, alignment)`; `iter_parallel_pairs` also yields the pericope_id
- `synoptiq/data/alignment.py` — Needleman-Wunsch token alignment via Bio.Align.PairwiseAligner.
  Scoring is **binary on the (normalized lemma, POS) key** (identical = match +2.5, else −100 → gap).
- `synoptiq/data/{pericope,splits}.py` — tradition classification + pericope-atomic stratified splits
- `synoptiq/models/koineformer.py` — **KoineFormer**: GreTa + LoRA wrapper, factory, save/load, generate
- `synoptiq/models/encoder.py` — MultiTaskEncoder (POS / biaffine parser / pericope heads) for Phase 2B
- `synoptiq/training/_config.py` — frozen dataclasses (`DataConfig`, `ModelConfig` [incl. `fid_*` for
  Q reconstruction], `TrainingConfig`, `DAPTConfig`)
- `synoptiq/training/dapt.py` — **DAPT**: data loader + AMP training loop, SIGTERM-safe checkpointing
- `synoptiq/evaluation/__init__.py` — linear-probe POS + lemma accuracy
- `synoptiq/evaluation/bootstrap.py` — pericope-grouped + paired cluster bootstrap CIs
- `scripts/_cli_utils.py` — canonical `detect_device()`; all training/eval scripts import it
- `scripts/{train_dapt,eval_baseline,run_ablation}.py` — DAPT training, zero-shot vs DAPT eval, ablation
- `modal/app_dapt.py` — Phase 2A Modal: `upload_data`, `start_training`, `run_ablation`, `train_and_eval_full_ft`
- `docs/DIRECTION_NEGATIVE_RESULT.md` — the closed direction investigation (do not re-attempt)

## Phase 2A results (Paper A)

### POS + Lemma tagging (linear probe, SynoptiQ test set)

| Model | POS Acc. | Lemma Acc. | Params | Checkpoint |
|-------|----------|------------|--------|------------|
| GreTa zero-shot | 95.32% | 82.37% | 0 | 880 MB |
| Full fine-tune (220M) | 96.11% | — | 220M | 880 MB |
| **KoineFormer LoRA** | **96.62%** | 81.34% | **3.7M** | **14 MB** |

Headline: LoRA DAPT eliminates 28% of POS errors vs zero-shot, beats full FT on accuracy, and
produces a 14 MB checkpoint. Lemma is flat — DAPT improves syntax but not vocabulary.

### DAPT corpus
- Koine (70%): SBLGNT full NT (~773K tokens) + Apostolic Fathers (~732K tokens) ≈ 1.5M tokens
- Classical replay (30%): First1KGreek (Homer, Plato, Xenophon)
- LXX yields 0 chunks (TextFabric `.tf` files are metadata, not Greek text)

### Training config
- GreTa (T5-base, 220M) frozen, LoRA r=16 α=32 targeting `["q", "v", "o", "wi", "wo"]`
- Note: `wi` does NOT actually apply (PEFT uses `endswith`; `wi` doesn't match `wi_0`/`wi_1`).
  Actual LoRA targets: W_q, W_v, W_o (attention) + W_o (FFN output) → 3.7M trainable
- 20,000 steps, batch 8, seq_len 512, AMP (FP16), AdamW lr=1e-4, cosine to zero
- A10G GPU, 58 minutes, 10 checkpoints + final, crash-safe with SIGTERM handler

## Modal commands

```bash
modal run modal/app_dapt.py::upload_data          # upload data (once)
modal run modal/app_dapt.py::start_training       # train KoineFormer (auto-resumes)
modal app logs synoptiq-dapt
modal run modal/app_dapt.py::run_ablation         # LoRA vs full-FT ablation
modal volume get synoptiq-outputs dapt/ models/koineformer/dapt/   # download adapters
```

Modal volumes: `synoptiq-data` (`/data/raw/` + `/data/processed/`), `synoptiq-outputs`
(`/outputs/dapt/`, `/outputs/ablation/`, `/outputs/full_ft/`).

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
cd paper
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex   # Paper A
xelatex project_overview.tex                                              # Project Overview
```
Custom fonts (Poppins, TeX Gyre Pagella, FreeSerif for Greek), XeLaTeX. Palette: marine (#123C48),
terracotta (#B75B36), papyrus (#F7F2E7). Use Overleaf if local TeX is unavailable.

## Pre-commit checklist (ALWAYS do these)

```bash
python3 -m ruff check synoptiq/ tests/ scripts/ --fix          # zero F or E errors
python3 -m pytest tests/ -q --tb=short                         # all tests pass
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
git add -A && git commit -m "<message>"
```
Ruff TC, UP, RUF, ANN, B90 warnings are cosmetic — only fix F and E.

## Architecture summary

**KoineFormer** = GreTa (T5 encoder-decoder, 220M params, Classical Greek) after PEFT-DAPT on a Koine
corpus (SBLGNT + Apostolic Fathers ~1.5M tokens). LoRA adapters only (~3.7M trainable, r=16 α=32,
W_q/W_v/W_o attention + W_o FFN). 70/30 Koine/Classical replay. 20K steps, 58 min on A10G. 96.62%
POS, 81.34% lemma, 14 MB checkpoint.

**Q Reconstruction** (Phase 5, next) = Fusion-in-Decoder: Matthew + Luke encoded independently →
concatenated hidden states → decoder cross-attention. Train on the triple tradition (Mt+Lk →
reconstruct Mark, ground truth available), then transfer to the double tradition to reconstruct
proto-Q. Uses `Corpus.parallel_pairs` and the `fid_*` fields on `ModelConfig`.

**Interpretability** (Phase 7, later) = SHAP feature importance vs Hawkins (1899); BERTViz attention;
multi-edition sensitivity (NA28/TR/Majority/WH).

## Tech stack

- Python 3.12+, PyTorch 2.6+, HuggingFace transformers, PEFT
- Modal (GPU cloud: A10G for DAPT)
- BioPython (token alignment), SHAP + BERTViz (interpretability)
- ruff (linting), pytest, XeLaTeX (paper)
- Data: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), Apostolic Fathers, First1KGreek (CC-BY-SA)
- GitHub: [abderahmane-ai/SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ)
- HF model: [ainouche-abderahmane/koineformer](https://huggingface.co/ainouche-abderahmane/koineformer)
- HF dataset: [ainouche-abderahmane/synoptiq-corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus)
