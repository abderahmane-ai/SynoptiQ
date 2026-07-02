# CLAUDE.md — SynoptiQ

A multi-task neural source criticism framework for the Synoptic Problem.
Applies transformers, causal direction modeling, and Bayesian inference
to determine the literary relationships among Matthew, Mark, and Luke.

## Project layout

```
SynoptiQ/
├── synoptiq/               # Python package (installable via pip install -e .)
│   ├── data/               # Corpus loading, parsing, alignment, splits
│   ├── models/             # KoineFormer, DirectionScorer, Editor, QReconstructor
│   ├── training/           # Trainer, configs, DAPT, multi-task, direction training
│   ├── evaluation/         # Metrics, baselines, calibration
│   ├── bayesian/           # PyMC models, bridge sampling, prior sensitivity
│   ├── interpretability/   # SHAP, Hawkins comparison, BERTViz
│   └── utils/              # Greek text, tokenization, types, constants, logging
├── scripts/                # CLI entry points (prepare_data.py, train_*.py, eval_*.py)
├── configs/                # YAML config files (data, model, training, bayesian, modal)
├── modal/                  # Modal GPU deployment (app definitions, volumes, secrets)
├── tests/                  # Mirrors synoptiq/ structure; uses conftest.py fixtures
├── notebooks/              # Exploratory notebooks (not in package)
├── data/                   # Git-ignored: raw corpora, processed Parquet files
├── models/                 # Git-ignored: downloaded HF models, trained checkpoints
├── outputs/                # Git-ignored: logs, predictions, figures
├── SYNOPTIQ_MASTER_PLAN.md  # Research design, architecture, innovation rationale
└── IMPLEMENTATION_PLAN.md   # Phases, subtasks, file lists, budget, timeline
```

## Key files to know

- `synoptiq/utils/types_.py` — All shared TypedDicts, Literals, Protocols used everywhere
- `synoptiq/utils/constants.py` — **Aland pericope table** (bedrock of the project), MorphGNT tagset maps, Goodacre fatigue pericopes, genre classifications. 1,337 lines, definitional.
- `synoptiq/utils/greek.py` — Greek text normalization (NFD accent stripping, sigma normalization)
- `synoptiq/data/corpus.py` — Central `Corpus` class. Single entry point for all data access.
- `synoptiq/data/alignment.py` — Needleman-Wunsch token alignment via Bio.Align.PairwiseAligner
- `synoptiq/training/_config.py` — Five frozen dataclasses with all training configuration

## Pre-commit checklist (ALWAYS do these)

Before every commit, in this exact order:

```bash
# 1. Lint and auto-fix
python3 -m ruff check synoptiq/ tests/ scripts/ --fix
python3 -m ruff format synoptiq/ tests/ scripts/

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
Tests run FIRST, caches cleaned AFTER (so cached .pyc files don't pollute the commit).

Or use the Makefile shortcut:
```bash
make check   # ruff format + ruff check + mypy + pytest
```

## Architecture summary

**KoineFormer** = GreTa (T5 encoder-decoder, monolingual Ancient Greek) after PEFT-DAPT on Koine corpus (SBLGNT + LXX + Josephus + Apostolic Fathers). LoRA adapters only (~4.5M trainable params). 70/30 Koine/Classical replay buffer.

**Direction Scorer** = Cross-attention asymmetry between parallel passages → 8 asymmetry features → 3-way classification (A→B, B→A, independent). Adversarial GRL head strips authorship style. Trained on triple tradition (known direction: Mark → Matthew, Mark → Luke).

**Editorial Fatigue** = Position-weighted consistency loss: `L_fatigue = Σ w(i) · D_KL(edit_dist_i || source_dist_i)` where `w(i) = exp(-λ · i/N)`.

**Q Reconstruction** = Fusion-in-Decoder (FiD): Matthew + Luke encoded independently → concatenated hidden states → decoder with cross-attention. Trained on triple tradition (Matthew+Luke → reconstruct Mark) then transferred to double tradition.

**Bayesian Comparison** = Direction scorer outputs → MC Dropout (T=20) → (μ_i, σ²_i) per pericope → PyMC Beta models with precision κ_i = 1/σ²_i. Four hypotheses: 2SH, FGH, Augustinian, Griesbach.

**Interpretability** = SHAP feature importance compared to Hawkins' *Horae Synopticae* (1899). BERTViz attention visualization. Multi-edition sensitivity (NA28 vs TR vs Majority Text vs WH).

## Current phase

Phase A (data pipeline) is complete. Ready for Phase 2 (KoineFormer DAPT + multi-task).

## Tech stack

- Python 3.12+, PyTorch 2.6+, HuggingFace transformers 4.51+
- PEFT (LoRA adapters), Modal (GPU cloud), Weights & Biases (experiment tracking)
- BioPython (token alignment), PyMC + ArviZ (Bayesian), SHAP + BERTViz (interpretability)
- ruff (linting + formatting), mypy (strict type checking), pytest (testing)
- Data sources: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), PROIEL (CC BY-NC-SA), First1KGreek (CC-BY-SA)
