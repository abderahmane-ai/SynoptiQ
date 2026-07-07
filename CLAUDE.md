# CLAUDE.md — SynoptiQ

A multi-task neural source criticism framework for the Synoptic Problem.
Applies transformers, causal direction modeling, and Bayesian inference
to determine the literary relationships among Matthew, Mark, and Luke.

## graphify (codebase knowledge graph)

This project has a knowledge graph at `graphify-out/` (git-ignored — generated
locally, never committed). It covers `synoptiq/`, `scripts/`, and `modal/`:
1326 nodes, 2439 edges, 77 communities. God nodes: `Corpus`, `KoineFormer`,
`TokenRecord` (live) and `DirectionDataset`, `DirectionScorer` (now under
`synoptiq/legacy/`; the live direction path is `synoptiq/direction/` + `bayesian/rooting`).

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

**Last commit:** `refactor(phase3): unify direction scorer into synoptiq/direction/ + wire
Phase 6`. Tree clean; ruff F/E passes; 133 tests pass. Full story in
`docs/DIRECTION_SCORER_FINDINGS.md` (read it first — it LEADS with the delivered two-regime
`DirectionScorer`, then the RPM baseline, then the negatives).

**Phase 3 DELIVERED as a real component (`synoptiq/direction/`).** The objective (master plan
§4.1) — a per-pericope direction-probability sensor feeding Phase 6 — now exists:
- `synoptiq/direction/scorer.py` — **DirectionScorer**: per pericope, per pair, emits calibrated
  `[A→B, B→A, independent]` + confidence + abstention (`DirectionScores` in `utils/types_.py`).
- **Two regimes** (the robust signal needs a 3rd witness): **triangulated** (agreement-structure
  `centrality` — theory-neutral: the text agreeing more with the third gospel is more primitive;
  **G1 0.79 [0.72,0.86], 0.985 on the confident half, length-decorrelated corr 0.25**) and
  **pair-only** (weak connective+fatigue, abstains heavily; both-polarity honest on external).
- `synoptiq/direction/{alignment3,features}.py` — 3-way multiple alignment + all signed features
  (agreement spectrum, centrality, connective [demoted RPM], fatigue). `data/frequency.py` kept.
- **Phase 6 wired**: `bayesian/rooting.py` now CONSUMES `DirectionScores` (`relationship_counts`);
  `scripts/compare_hypotheses.py` runs corpus→scorer→posterior end-to-end and reproduces
  **2SH 0.64 / Farrer 0.36 / Griesbach ≈0 / Augustinian ≈0** from the robust signal.
- **Phase 4 (fatigue) merged into Phase 3** (one feature). RPM connective kept as a documented
  feature. Scripts: `train_direction_scorer.py` (calibrate+G1-G3+emit scores.json),
  `compare_hypotheses.py` (Phase 6), `direction_report.py` (findings). Old RPM scripts/modules
  folded/deleted; dead-end code stays under `synoptiq/legacy/`.

**Next up:** update Paper B to headline the two-regime DirectionScorer (agreement structure
primary, RPM the baseline). The `paper_b/` draft still describes RPM as primary — revise.

**The scientific history (settled — full detail in `docs/DIRECTION_SCORER_FINDINGS.md`):**

- **Negatives (do NOT re-try):** all *global passage scores* — similarity features (chance),
  NLL/MDL compression (Markan style + length; sign flips with length polarity), a synthetic-
  trained head (generator artifact + length prior). Proven by two known-direction sets of
  opposite length polarity (Jude→2 Peter copy-longer; LXX Chronicles copy-shorter): global
  scores track *which text is longer*, not the source.
- **RPM baseline (now the demoted connective feature):** variant polarization via the
  connective-smoothing canon (καί→δέ) reached 0.78 synoptic / 0.876 on evidenced pericopes /
  0.97 @25% cov, length-controlled (R6: survives matching καί-density, matched stratum 0.784
  [0.64,0.90], per-edit ratio 3.5:1). Fatigue does not add a second canon on the synoptics
  (chance there; genre-limited). Superseded as the *primary* signal by agreement structure.
- **Robust signal (now primary):** the agreement structure across three witnesses — Markan-hub
  15:1, Griesbach excluded (singular rate 0.22), Mark-Q overlaps rediscovered; theory-neutral,
  uses all shared words. This is the DirectionScorer's triangulated `centrality` feature.

**NEXT (open leads):** revise Paper B to the two-regime scorer; Farrer-vs-Q still needs more
Q-material data or a sharper canon (double tradition too sparse); Phase 6 can be extended to a
hierarchical PyMC model with prior-sensitivity (it consumes the scorer non-circularly).

Do-not-repeat rules: global passage scores are dead; `local_brevior` is a confirmed length
proxy (excluded by design); aggregate `intro_lateness` fatigue is chance on the synoptics
(H4 — do NOT re-add it as a synoptic canon); always test on BOTH length polarities and use
pericope-grouped bootstrap CIs (`synoptiq/evaluation/bootstrap.py`). The plan file
`~/.claude/plans/buzzing-squishing-fairy.md` (R0–R5) is now fully executed.

## Cold start (fresh clone / new machine)

`data/`, `models/`, and `graphify-out/` are all git-ignored, so a fresh clone
has none of them. Regenerate in this order (skip any step whose output already
exists locally):

```bash
# 1. Install the package (Python 3.12+)
pip install -e .

# 2. Regenerate the corpus → data/processed/{tokens,pericopes}.parquet
python scripts/prepare_data.py --validate

# 3. Get DAPT adapters → models/koineformer/dapt/final/ (needed for Phase 3)
#    Option A (Modal):
modal volume get synoptiq-outputs dapt/ models/koineformer/dapt/
#    Option B (HuggingFace, no Modal):
python -c "from huggingface_hub import snapshot_download; snapshot_download('ainouche-abderahmane/koineformer', local_dir='models/koineformer/dapt/final')"

# 4. Verify everything is wired up (all 133 must pass)
python -m pytest tests/ -q

# 5. Rebuild the knowledge graph (git-ignored) — invoke the `graphify` skill,
#    or from the CLI over the source dirs. Then `graphify query "..."` works again.
```

## Project layout

```
SynoptiQ/
├── synoptiq/               # Python package (pip install -e .)
│   ├── data/               # Corpus loading, parsing, alignment, splits, frequency, external pairs
│   ├── direction/          # Phase 3 DirectionScorer (the sensor; absorbs Phase 4 fatigue)
│   │   ├── alignment3.py   # Mark-anchored 3-way multiple alignment (align_three, Column)
│   │   ├── features.py     # agreement spectrum + centrality (triangulation) + connective + fatigue
│   │   └── scorer.py       # DirectionScorer: two-regime → calibrated DirectionScores + abstention
│   ├── models/             # KoineFormer, MultiTaskEncoder
│   │   ├── koineformer.py  # GreTa + LoRA wrapper, save/load adapters, generate
│   │   └── encoder.py      # Multi-task encoder: POS, biaffine parser, pericope heads
│   ├── training/           # DAPT, multi-task training loops (_config, dapt, multitask)
│   ├── evaluation/         # Linear probe POS/lemma; bootstrap.py (pericope-grouped CIs)
│   ├── bayesian/           # Phase 6: rooting.py — CONSUMES DirectionScores → stemma posterior
│   ├── legacy/             # ARCHIVED Phase-3 dead-ends (global scores/NLL/redaction) — see __init__
│   ├── interpretability/   # SHAP, Hawkins comparison, BERTViz
│   └── utils/              # Greek text, tokenization, types (DirectionScores), constants, logging
├── scripts/                # CLI entry points
│   ├── prepare_data.py     # Phase 1: download → parse → align → split → cache
│   ├── export_hf_dataset.py # Package SynoptiQ Corpus as HuggingFace dataset
│   ├── train_dapt.py       # Phase 2A: KoineFormer DAPT (--smoke-test for quick check)
│   ├── train_multitask.py  # Phase 2B: Multi-task LoRA fine-tuning
│   ├── train_direction_scorer.py # Phase 3: calibrate DirectionScorer + G1-G3 + emit scores.json
│   ├── compare_hypotheses.py     # Phase 6: scores.json → stemma posterior + Farrer/Q
│   ├── direction_report.py       # Phase 3: agreement structure + style-confound control (findings)
│   ├── build_external_pairs.py / build_lxx_pairs.py  # known-direction validation sets
│   ├── eval_baseline.py    # Zero-shot vs DAPT evaluation (--zero-shot, --dapt-checkpoint)
│   ├── run_ablation.py     # LoRA vs full fine-tune ablation
│   └── legacy/             # ARCHIVED Phase-3 dead-end scripts (diagnose/train_direction/redaction…)
├── paper/                  # Paper A: KoineFormer (XeLaTeX)
│   ├── main.tex            # Main manuscript (Poppins + TeX Gyre Pagella, Mediterranean)
│   ├── project_overview.tex # SynoptiQ project brief for Yale/Oxford/Harvard audience
│   └── references.bib      # BibTeX references (6 entries)
├── modal/                  # Modal GPU deployment
│   ├── app_dapt.py         # Phase 2A: DAPT training, ablation, full-FT eval
│   └── legacy/app_direction.py # ARCHIVED Phase 3: dead-end direction-scorer training (T4)
├── datasets/               # HuggingFace dataset export (gitignored except README)
│   └── synoptiq-corpus/    # Pushed to ainouche-abderahmane/synoptiq-corpus
├── tests/                  # 133 tests (mirrors synoptiq/ structure)
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
| Phase 3 | ✓ Direction Scorer | **DirectionScorer delivered** (`synoptiq/direction/`) — two-regime, per-pericope calibrated `[A→B,B→A,indep]` + abstention. Triangulated (agreement-structure `centrality`): **G1 0.79 [0.72,0.86], 0.985 confident half, length-decorrelated (0.25)**; pair-only (connective+fatigue) abstains. RPM connective = documented feature. See `docs/DIRECTION_SCORER_FINDINGS.md` |
| Phase 4 | ✓ merged into Phase 3 | Editorial fatigue (`intro_lateness`) is one pair-only feature of the DirectionScorer; genre-limited (chance on synoptics) |
| Phase 5 | ○ Q Reconstruction | Not started |
| Phase 6 | ◐ Bayesian | `bayesian/rooting.py` CONSUMES DirectionScores → posterior **2SH 0.64 / Farrer 0.36 / Griesbach ≈0 / Aug ≈0** (`compare_hypotheses.py`); hierarchical PyMC + prior-sensitivity still to do |
| Phase 7 | ○ Interpretability | Not started |
| Paper A | ✓ Draft | paper/main.tex — complete manuscript, verified numbers |
| Paper B | ◐ Draft | paper_b/main.tex — describes RPM as primary; revise to headline the two-regime DirectionScorer |

## Key files to know

- `synoptiq/utils/types_.py` — All shared TypedDicts (`Direction`, `DirectionScores` already defined), Literals, Protocols
- `synoptiq/utils/constants.py` — **Aland pericope table** (bedrock), MorphGNT tagset maps, Goodacre fatigue pericopes, 1,337 lines
- `synoptiq/utils/greek.py` — Greek text normalization (NFD accent stripping, sigma normalization)
- `synoptiq/data/corpus.py` — Central `Corpus` class. `direction_pairs(split=...)` yields (book_a, tokens_a, book_b, tokens_b, alignment) for direction scorer training
- `synoptiq/data/alignment.py` — Needleman-Wunsch token alignment via Bio.Align.PairwiseAligner. Scoring is **binary on the (normalized lemma, POS) key** (each pair → a Private-Use-Area char; identical = `match` +2.5, else `mismatch` -100, forcing a gap). Surface form is reported by `alignment_score` but does not steer the path.
- `synoptiq/data/augmentation.py` — Bootstrap resampling, sliding windows, scribal noise injection
- `synoptiq/training/_config.py` — Five frozen dataclasses: `ModelConfig` already has `direction_num_classes`, `asymmetry_num_features`, `direction_signed_features`, `direction_independence_features`
- `synoptiq/models/koineformer.py` — **KoineFormer**: GreTa + LoRA wrapper, factory, save/load, generate
- `synoptiq/direction/scorer.py` — **DirectionScorer**: two-regime, swap-antisymmetric; `score_pair` → `DirectionScores`. Weights fit on external pairs only (`fit`); centrality is a fixed prior. `with_priors()` for the unfit centrality-led scorer.
- `synoptiq/direction/features.py` — all signed features (positive ⇒ A source): `centrality_asym` (triangulation, the robust one), `connective_vote` (demoted RPM canon), `intro_lateness` (fatigue), plus `agreement_spectrum` + `align_three` re-export
- `synoptiq/direction/alignment3.py` — Mark-anchored 3-way multiple alignment (`align_three`, `Column`)
- `synoptiq/data/frequency.py` — Koine lemma-rarity markedness table (feature resource)
- `synoptiq/bayesian/rooting.py` — **Phase 6**: `relationship_counts(DirectionScores)` → Beta-Bernoulli marginal likelihood per relationship → posterior over the 4 stemmata + Farrer/Q Bayes factor
- `synoptiq/evaluation/bootstrap.py` — pericope-grouped + paired cluster bootstrap (use for ALL direction comparisons)
- `synoptiq/data/external_pairs.py` — loader for known-direction pairs (Jude→2Pet, LXX Chronicles); no Corpus needed
- `docs/DIRECTION_SCORER_FINDINGS.md` — **the Phase-3 conclusion**: full evidence + reproduction commands
- `synoptiq/training/dapt.py` — **DAPT**: data loader + training loop with AMP, SIGTERM handler, crash-safe checkpointing
- `synoptiq/evaluation/__init__.py` — Linear probe evaluation: POS + lemma accuracy
- `scripts/train_dapt.py` — DAPT CLI: `--smoke-test` (100 steps CPU), full training (20K steps GPU)
- `scripts/train_direction_scorer.py` — calibrate the DirectionScorer + gated validation G1-G3 + emit `outputs/direction/scores.json`; `scripts/compare_hypotheses.py` — Phase 6 consumer (scores → posterior); `scripts/direction_report.py` — agreement structure + style-confound control
- `scripts/build_external_pairs.py` / `build_lxx_pairs.py` — Jude→2Pet (copy longer) and LXX Chronicles (copy shorter) known-direction sets for RPM validation
- `scripts/eval_baseline.py` — Compare zero-shot GreTa vs DAPT KoineFormer on POS + lemma
- `scripts/run_ablation.py` — LoRA vs full fine-tune loss curve comparison
- `synoptiq/legacy/` + `scripts/legacy/` — **ARCHIVED Phase-3 dead-ends** (10-feature `DirectionScorer`, `MDLDirectionHead`, NLL/MDL scoring, synthetic redaction generator + their trainers/scripts). Confounded by length/Markan style — kept only for Paper-B reproducibility; nothing live imports them. See `synoptiq/legacy/__init__.py`.
- `scripts/_cli_utils.py` — Shared CLI helpers. **Canonical `detect_device()`** lives here; all training/eval scripts import it (do not re-add per-script copies).
- `modal/app_dapt.py` — Phase 2A Modal: `upload_data`, `start_training`, `run_ablation`, `train_and_eval_full_ft`
- `modal/legacy/app_direction.py` — ARCHIVED Phase 3 Modal (dead-end scorer): `upload_data`, `start_training` (T4, 5K steps), `smoke_test`

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

> **Read `docs/DIRECTION_SCORER_FINDINGS.md` first.** The live component is the two-regime
> **`DirectionScorer`** (`synoptiq/direction/`): per pericope it emits calibrated
> `[A→B, B→A, independent]` + abstention. The robust signal is the agreement-structure
> `centrality` feature (triangulated regime); the connective canon (RPM) and fatigue are
> demoted pair-only features. Global passage scores (the 10-feature encoder scorer, NLL/MDL,
> synthetic-trained head) are DEAD (confounded with length + Markan style, proven on both
> length polarities) and archived under `synoptiq/legacy/`. The section below documents the
> old encoder-probe baseline for reference only.

### Architecture

The Direction Scorer is a frozen-encoder, feature-calibrated,
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
- The "beat 72.8%" gate is superseded: that 72.8% is itself confounded (pooled
  embeddings detect Markan authorship), and on ~14 test pericopes it carries a
  ±13-pt bootstrap CI. Use pericope-grouped bootstrap CIs (`synoptiq/evaluation/bootstrap.py`),
  not point accuracy, for any direction comparison.

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
- Open question — now ANSWERED: the 10 asymmetry features carry neither (chance, and author only weakly decodable). The direction signal that *does* move (conditional-NLL asymmetry) is Markan style + length, not direction. Full evidence and the two-length-polarity proof are in `docs/DIRECTION_SCORER_FINDINGS.md`.

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
# The live RPM pipeline is CPU-only and runs locally (no Modal): see
#   scripts/train_direction_scorer.py, compare_hypotheses.py, direction_report.py
# The GPU direction-scorer training app is an ARCHIVED dead-end (kept for
# reproducibility only):
modal run modal/legacy/app_direction.py::smoke_test        # archived, dead-end
modal run modal/legacy/app_direction.py::start_training    # archived, dead-end
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

# 2. Full test suite — all 133 must pass
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

**Direction Scorer** (Phase 3) = the **Redactional Polarization Model (RPM)** (see
`docs/DIRECTION_SCORER_FINDINGS.md`). Direction is the stemmatology *rooting* problem:
align a pair → extract typed variants → score each with signed textual-criticism canon
features (connective smoothing is the survivor; positive ⇒ X is the source) → aggregate
with abstention → pool per-pericope votes into a posterior over the four stemmata. Reaches
0.78 synoptic directed accuracy (0.876 on evidenced pericopes) and recovers Markan priority
unsupervised (2SH 0.64 / Farrer 0.36 / Griesbach ≈ 0 / Augustinian ≈ 0). The earlier
*global-score* heads (10-feature `DirectionScorer`, NLL `MDLDirectionHead`, synthetic
redaction corpus) are a documented negative result — confounded with length and Markan
style — now archived under `synoptiq/legacy/`.

**Editorial Fatigue** (Phase 4) = Position-weighted KL divergence between
editing distribution and source distribution, with exponential decay weight.
Detects copying author reverting to own style over the course of a pericope. The
feature (`synoptiq/direction/features.py`, `intro_lateness`) is a
genre-limited canon — chance on the synoptics (RPM H4), real but weak on LXX narrative.

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
- ruff (linting), pytest (133 tests), XeLaTeX (paper)
- Data: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), Apostolic Fathers, First1KGreek (CC-BY-SA)
- GitHub: [abderahmane-ai/SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ)
- HF model: [ainouche-abderahmane/koineformer](https://huggingface.co/ainouche-abderahmane/koineformer)
- HF dataset: [ainouche-abderahmane/synoptiq-corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus)
