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

### 2026-07-10 — Koine-T5 delivered + bug-fixed; PAPER PIVOT to the honest paper

**Branch renamed `cleanup/remove-direction-hypotheses` → `feat/phase5-koine-t5`** (not `main`,
not pushed). **Two commits landed:** `f0115aa` (Phase-5 M0–M4 code + tests + preregistration +
DAPT-decontamination) and `8cbc8a1` (Koine-T5 trainer + docs). 186 tests pass; repo is pristine
(zero cache/`.DS_Store`/junk); all `.md` files correctly tracked (the two root `*_PLAN.md` are
intentionally git-ignored — do NOT archive them).

**Koine-T5** = the standalone multitask model (was root `train_greta_ultimate.py`, renamed +
relocated to **`modal/app_koine_t5.py`**, sibling of `app_dapt.py` which trained KoineFormer #1).
Model name `koine-t5`; entrypoints `train`/`generate`/`build_tokenizer`; outputs to the
`koine-t5-outputs` volume. See the "Koine-T5" section below.

**⚠️ UNCOMMITTED fixes in `modal/app_koine_t5.py` (this session, AFTER `8cbc8a1`) — commit them
or they live only locally.** Found while debugging the first GPU runs (POS-EM was stuck at 0.000):
- **LR-schedule unit bug (the cause):** `scheduler.step()` fires per *optimizer* update, so
  `LambdaLR` got optimizer-steps while `WARMUP_STEPS`/`MAX_STEPS` were micro-steps → warmup ran
  8× too long (4,800 micro-steps = 40% of training) and cosine never annealed. `lr_lambda` now
  converts (`warmup_opt = WARMUP_STEPS // GRAD_ACCUM`, etc.). Verified numerically.
- **T5 eval used left-padding** (a decoder-only requirement) → corrupted the encoder → garbage
  eval. Now right-padded (T5 default).
- **"6.4 epochs" was a unit error → really ~0.8 epochs** (~55% of the 15,041 POS pool). Comment
  + startup print corrected.
- **POS-EM eval false 0.000 (case-fold) — the bug that hid everything:** GreTa's tokenizer
  LOWERCASES all text (`N-`→`n-`), so the model emits lowercase tags while gold is upper-case
  MorphGNT → a naive `==` pins EM/token-acc to exactly 0.000 even though the model IS learning.
  Fixed: `_batch_pos_predict` upper-cases preds (MorphGNT codes are case-unique → lossless).
  See memory `greta-tokenizer-lowercases-pos-eval`.
- Added an **eval diagnostic** (prints `gold:`/`pred:` — this is what caught the case-fold);
  **bumped the run fingerprint to `rev: 3`** (rev 2 = LR + T5-padding fixes; rev 3 = case-fold
  fix) so superseded checkpoints are auto-ignored and a rerun starts clean.
- **Prose cleanup:** dropped `Bulletproof`/`maximalist`/`SOTA`/`Phase N:` slop; fixed the
  synoptic-pool count (`~401` → **155**; 401 was the old *total*).
- **Also uncommitted:** the resume path restores only adapter weights, not optimizer/scheduler
  state — fine for a fresh run, suboptimal on crash-resume (flagged, not fixed).

**First run outcome (2026-07-10, `MAX_STEPS=12000`, abandoned):** LR fix CONFIRMED (lr peaks at
step 600; loss 5.4→1.3) and the diagnostic showed the model genuinely learning POS (degenerate
`n- n- n-` → varied `rp d- rp v-`) — but POS-EM read 0.000 throughout, entirely the **case-fold**
bug above. Superseded by the rev-3 rerun below. graphify graph was refreshed this session
(code-only AST update; koine-t5 + Phase-5 now in it).

**Rev-3 rerun DONE (2026-07-10, `MAX_STEPS=30000`): Koine-T5 is trained and healthy.** All three
fixes confirmed in the wild (lr annealed to 0 exactly at 30000; POS-EM real, not 0.000; `best/`
selection worked). **Best checkpoint = step 28000: pooled PROIEL-dev POS token-acc 0.9169 / EM
0.686; NT tok 0.966 / EM 0.852; Classical tok 0.877 / EM 0.520** (n=250 each). NT token-acc 96.6%
matches the KoineFormer linear probe's 96.62% — but via free seq2seq *generation* and on a
different eval set (PROIEL dev vs SynoptiQ test), so cite as comparable, not identical. Curve was
plateauing (26k 0.913 → 28k 0.917 → 30k 0.914): 30k steps ≈ converged, no rerun needed. The
earlier "repetitive free-generation POS" worry is resolved (eval samples show varied, input-aligned
tags). Adapters downloaded to `models/koine_t5/{best,final}/` (fetched file-by-file — the
`modal volume get` directory quirk applies to this volume too); `best/metrics.json` has the
numbers; run fingerprint `526985e86dbf`. Volume also holds `step-29000`/`step-30000`.

**PAPER PIVOT (the next writing deliverable): the HONEST paper — see `docs/PAPER_PLAN.md`.**
A collaborator's outline proposed conditional-NLL / perplexity-ratio **direction detection** as a
"novel framework" — that is this project's CLOSED negative result (invalidated, not "unvalidated";
`docs/DIRECTION_NEGATIVE_RESULT.md`). The honest paper = **negative result** (direction
unrecoverable from text; argument + sign-flip demonstration) + **SynoptiQ corpus** + **Koine-T5** +
the **preregistered E2 null** (minor agreements = text-critical noise). `docs/PAPER_PLAN.md` has the
full section-by-section outline, evidence inventory, DONE-vs-NEEDED, and honest framing rules
(no "theorem" overclaim; NLL ratios presented as *failed*; the null reported as a null).

---

**Direction + hypotheses code removed (2026-07-07); committed on branch
`cleanup/remove-direction-hypotheses` (not yet merged to `main`); ruff F/E clean; 91 tests pass.**
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

**Next:** **Phase 5** — two tracks (see `docs/SOURCE_CRITICISM_STUDY.md`, a frozen preregistration).
**Track A** = Q reconstruction (Fusion-in-Decoder): reconstruct Mark on the triple tradition
(ground truth), then transfer to the double tradition for proto-Q. **Track B** = source
identification *without* re-opening direction: compare generative models of the double tradition
(2SH bottleneck vs Farrer/MPH direct) trained only on hypothesis-neutral triple-tradition
supervision (Mk→Mt, Mk→Lk, which every live hypothesis accepts). Neither depends on solving
per-pair direction; `docs/DIRECTION_NEGATIVE_RESULT.md` still stands. `ModelConfig` carries `fid_*`.

**Phase-5 M0 delivered (2026-07-08), tests green (132 pass):** `synoptiq/data/study_design.py`
(full-triple folds, overlap partition, census, prereg hashing), `StudyConfig` in `_config.py`,
`MARK_Q_OVERLAP_{CORE,EXTENDED}` in `constants.py`, `synoptiq/evaluation/model_comparison.py`
(lift/DiD/MDE power sim) + `statistic_ci`/`difference_in_differences` in `bootstrap.py`,
`synoptiq/evaluation/scoring.py` (NLL backbone; the M2 redactor/FiD models go in `synoptiq/models/`).
Script: `scripts/prepare_study.py` (`freeze` + `power` subcommands) → `outputs/study/`.
**Key data-driven corrections:** effective triple-tradition N is **65 full
triples, not 88** (23 lack a book); the DT `wisdom` stratum (4/17) has **no triple analog** →
unidentifiable a priori. Power analysis: DiD contrast (5 overlap pericopes) is the binding
bottleneck (MDE ~3× the whole-TT lift).

**M1 code delivered (2026-07-08), 148 tests pass:** DAPT decontamination (`exclude_books` on the
operative `DAPTConfig` in `dapt.py`; SBLGNT stem filter in `_extract_text_from_dir`, verified on
real data — drops 12 gospel files), `--no-synoptics` flag on `scripts/train_dapt.py`, Modal
`start_training_ns` (→ `/outputs/dapt_ns`), audit stats `synoptiq/evaluation/contamination.py`
(perplexity, DiD memorization gap, exact-match) + `scripts/audit_contamination.py`. **M1 GPU run
still pending** — `modal run modal/app_dapt.py::start_training_ns` (~1 hr A10G), then
`audit_contamination.py --compare-adapters models/koineformer/dapt/final models/koineformer_ns/final`.
Current KoineFormer DAPT includes SBLGNT (= the eval gospels), so likelihood verdicts need the NS
adapters.

**M1 GPU run DONE (2026-07-08): contamination is negligible.** NS adapters trained
(`modal .../start_training_ns`, `excluded_files=12`) and audited: memorization gap **0.016**
(threshold 0.25), verse-completion exact-match **0%**, gospel vs control perplexity 2.08/1.66
(original) ≈ 2.12/1.66 (NS). LoRA's 1.5% capacity can't memorize verbatim — the likelihood
approach is safe. NS model at `models/koineformer_ns/final`; audit in `outputs/study/audit.md`.
(Download quirk: `modal volume get` mangles directory downloads — fetch the two adapter files
individually; `audit_contamination.py` auto-resolves nested adapter paths.)

**M2 fold-0 GPU run DONE (2026-07-08). Two-part result — see `docs/SOURCE_CRITICISM_STUDY.md` §3b.**
Modules: `synoptiq/models/{redactor,fid,_seq2seq_base}.py`, `synoptiq/data/redaction.py`,
`synoptiq/evaluation/reconstruction.py`, `scripts/train_{redactors,fid}.py`, `modal/app_fid.py`.
- **Loader bug fixed:** `_seq2seq_base.load_greta_seq2seq` must NOT add a `[PAD]` token / resize
  embeddings — GreTa ships `<pad>`=0 and `</s>`=1; adding a new pad desyncs the tokenizer pad id
  from the model's decoder-start id and collapses generation to empty (and inflates NLL ~4×).
- **Operators: strong PASS.** Held-out NLL ≈ 2 nats; real source beats a *mismatched* source by
  ~1 nat (R_Lk 1.995 vs 3.094) — they use pericope-specific content, not just style. This is the
  validated signal the E2/E1 verdicts build on (scoring path, no generation).
- **Track A reconstruction: bounded negative.** FiD (Mt+Lk→Mark) F1 0.31 vs nearest-witness 0.56;
  overfits (train loss 2.17→1.28, held-out F1 flat). Demoted from headline to a reported limitation
  — witnesses are too close to Mark for abstractive fusion to beat copying at 52-pericope scale.

**M3/M4 code DONE (2026-07-09), 186 tests pass. M4 E2 (Lk-target) verdict DONE (pooled CV below);
M3 G1/G2/G4 gates still need GPU.** `synoptiq/evaluation/verdict.py`
(`minor_agreement_test` = excess-lift E2, `did_contrast`, `null_threshold` = G3 floor,
`channel_recovery_gate` = G1/G2; all unit-tested). `scripts/run_mai_test.py` runs the E2 verdict
(per-pericope excess lift of Mk+Mt→Lk vs a mismatched-Matthew control + overlap-vs-rest DiD + G3
floor) — CPU-smoke-tested end-to-end on real GreTa. `train_fid.py` now takes `--witnesses/--target`
so it also trains the E2 model. Modal: `train_fid_mai`, `run_mai`, `mai_cv`; `scripts/pool_mai.py`
pools per-fold rows into the CV verdict.

**E2 pooled CV verdict DONE (2026-07-09): NULL-after-calibration on both axes.** `mai_cv` trained
+ scored all 5 folds → `outputs/study/mai/mai_pooled.json` (65 held-out pericopes, 5 overlap in one
DiD). **Pooled excess lift +0.169 [CI 0.127,0.213] P(>0)=1.00 but does NOT clear the G3 null floor
0.194**; **DiD overlap(5) vs rest(60) +0.096 [CI −0.049,+0.296] spans zero (not overlap-concentrated).**
So the Lk-target minor-agreement signal is null once calibrated against the mismatched-Matthew control —
the prereg floor + DiD kill what a naive test (CI excludes 0) would have published as Farrer support.
Reads as §4's **strict-independence-beyond-Mark** row (MAs = text-critical noise; a publishable null).
Confirms + generalizes fold 0 across the full CV. Results folded into `docs/SOURCE_CRITICISM_STUDY.md`
§3b + M4. **Then:** symmetric **Mt-target** E2 model (distinguishes Farrer vs MPH + the T4 symmetry
check) and edition-swap ablation (T8) — both flip this from "null" to "null and robust"; M3 G1/G2 + G4;
M5 E1 (only if K2 powered); M6 write-up.

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
│   │   ├── study_design.py # Phase 5: full-triple folds, overlap partition, census, prereg hashing
│   │   └── redaction.py    # Phase 5: source→target pairs, fusion examples, source-dropout
│   ├── models/             # KoineFormer, MultiTaskEncoder, + Phase-5 redactor/FiD
│   │   ├── koineformer.py  # GreTa + LoRA wrapper, save/load adapters, generate
│   │   ├── encoder.py      # Multi-task encoder: POS, biaffine parser, pericope heads
│   │   ├── redactor.py     # Phase 5: R_Lk/R_Mt/G_Mt/G_Lk seq2seq LoRA operators + NLL scoring
│   │   ├── fid.py          # Phase 5: Fusion-in-Decoder (witnesses fused in the decoder) — Track A
│   │   └── _seq2seq_base.py# shared GreTa(+NS)+LoRA loader for redactor/FiD
│   ├── training/           # DAPT + multi-task training (_config incl. StudyConfig, dapt, multitask)
│   ├── evaluation/         # bootstrap CIs, scoring (NLL), model_comparison (power/DiD),
│   │                       #   contamination audit, reconstruction F1, linear probes
│   ├── interpretability/   # SHAP, Hawkins comparison, BERTViz
│   └── utils/              # Greek text, tokenization, shared types, constants, logging
├── scripts/                # CLI entry points
│   ├── prepare_data.py     # Phase 1: download → parse → align → split → cache
│   ├── train_dapt.py       # Phase 2A: KoineFormer DAPT (--no-synoptics → KoineFormer-NS)
│   ├── eval_baseline.py    # Zero-shot vs DAPT evaluation (--zero-shot, --dapt-checkpoint)
│   ├── prepare_study.py    # Phase 5 M0: `freeze` (census/folds/hashes) + `power` subcommands
│   ├── audit_contamination.py # Phase 5 M1: memorization audit (original vs KoineFormer-NS)
│   ├── train_redactors.py  # Phase 5 M2: train R_*/G_* operators with TT CV
│   ├── train_fid.py        # Phase 5 M2: train FiD, any --witnesses/--target (Track A or E2)
│   ├── run_mai_test.py     # Phase 5 M4: E2 excess-lift verdict + DiD + G3 floor
│   └── _cli_utils.py       # Shared CLI helpers (canonical detect_device())
├── paper/                  # Paper A: KoineFormer (XeLaTeX)
├── modal/                  # Modal GPU: app_dapt.py (DAPT), app_fid.py (M2), app_koine_t5.py (Koine-T5)
├── tests/                  # pytest suite (mirrors synoptiq/ structure) — 176 pass
├── data/ models/ outputs/  # All git-ignored (corpora, adapters, logs)
├── docs/DIRECTION_NEGATIVE_RESULT.md   # why copying-direction detection was closed
├── docs/SOURCE_CRITICISM_STUDY.md      # Phase 5 preregistration (Q reconstruction + source ID)
├── docs/PAPER_PLAN.md                  # the honest paper plan (negative result + corpus + Koine-T5 + E2 null)
├── paper/project_overview.tex · {SYNOPTIQ_MASTER,IMPLEMENTATION}_PLAN.md (local planning, git-ignored)
```

## Koine-T5 — standalone general-purpose model (`modal/app_koine_t5.py`)

A second, self-contained model line, **distinct from the KoineFormer encoder work above**.
`modal/app_koine_t5.py` trains **Koine-T5**: a general-purpose multitask Ancient Greek seq2seq
model (GreTa + LoRA r=32), sibling to `app_dapt.py` (which trained KoineFormer #1). One model,
**four balanced task pools** sampled every batch so none is starved:

1. `denoise` — online T5 span corruption on **raw** Greek prose
2. `pos` — POS tagging (MorphGNT tagset)
3. `lemma` — lemmatization
4. `synoptic` — Mark→Matthew / Mark→Luke style transfer (the tiny curated pool, upsampled)

`pos`/`lemma`/`denoise` are fed by the Gospel corpus **plus the UD_Ancient_Greek-PROIEL treebank**
(NT Koine + Herodotus Classical, ~214K tokens). PROIEL's granular **XPOS** column is mapped to the
13-code MorphGNT tagset (`S-`→`RA` etc. — *not* UPOS/FEATS, whose `PronType=Dem` on the article is a
trap); this cures the POS "task collapse" that afflicted the 401-example version. Validation is POS
**Exact-Match on PROIEL dev, reported NT vs Classical**, with the best-EM adapter saved to `best/`.
Config-fingerprinted checkpoints prevent a stale run from resuming into a changed config. Standalone
(no `synoptiq` imports). Run: `modal run modal/app_koine_t5.py::train` ·
`python modal/app_koine_t5.py demo` · outputs to the `koine-t5-outputs` volume (`best/` + `final/`).

**Status (2026-07-10): TRAINED + PUBLISHED** at
[huggingface.co/ainouche-abderahmane/koine-t5](https://huggingface.co/ainouche-abderahmane/koine-t5)
(CC BY-NC-SA 4.0; see the HuggingFace section for loading + task-prefix details).
Rev-3 run (`MAX_STEPS=30000`) complete — best checkpoint
(step 28000) at `models/koine_t5/best/`: PROIEL-dev POS token-acc **0.917 pooled / 0.966 NT /
0.877 Classical** (EM 0.686 / 0.852 / 0.520). See the 2026-07-10 handoff for run details.
The post-commit training-bug fixes (LR-schedule units, T5 eval right-padding, case-fold POS eval)
are validated by this run but **still UNCOMMITTED** in `modal/app_koine_t5.py`. Koine-T5 is the
honest paper's R2 (`docs/PAPER_PLAN.md` §5); next: `python modal/app_koine_t5.py demo
models/koine_t5/best` for qualitative checks, and a HF model card if it ships with the paper.

## Current status

| Phase | Status | Key result |
|-------|--------|------------|
| Phase 0 | ✓ Foundation | Types, constants, Greek utils, project skeleton |
| Phase 1 | ✓ Data Pipeline | SynoptiQ Corpus: 49,061 tokens, 170 pericopes, 235 alignments |
| Phase 2A | ✓ DAPT | KoineFormer trained: 96.62% POS, 81.34% lemma, 14 MB |
| Phase 2B | ○ Multi-task | Code ready, not yet trained |
| Phase 3 | ✗ Removed | Direction detection — **closed negative result** (`docs/DIRECTION_NEGATIVE_RESULT.md`) |
| Phase 5 | ◐ In progress | **M0–M2 done; M3/M4 code done; M4 E2 (Lk-target) pooled CV DONE** (186 tests). Operators strong PASS; Track A reconstruction = bounded negative (demoted). **E2 verdict: null-after-calibration** (lift +0.169 < G3 floor 0.194; DiD spans 0) → strict-independence-beyond-Mark. See `docs/SOURCE_CRITICISM_STUDY.md` |
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
- `synoptiq/models/redactor.py` — **Redactor** (Phase 5): source→target seq2seq LoRA operator; the four
  instances R_Lk/R_Mt/G_Mt/G_Lk differ only in training data. `score()` = teacher-forced conditional NLL
- `synoptiq/models/fid.py` — **FusionInDecoder** (Phase 5, Track A): encodes witnesses separately, `torch.cat`
  their encoder states, decoder cross-attends over the fusion → reconstruct Mark from Mt+Lk (then proto-Q)
- `synoptiq/training/_config.py` — frozen dataclasses (`DataConfig`, `ModelConfig` [incl. `fid_*`],
  `TrainingConfig`, `DAPTConfig`, **`StudyConfig`** [the hashed Phase-5 prereg artifact])
- `synoptiq/training/dapt.py` — **DAPT**: data loader (+ `exclude_books` decontamination) + AMP loop
- `synoptiq/data/study_design.py` — Phase-5 folds/census/overlap partition/prereg hashing
- `synoptiq/data/redaction.py` — Phase-5 training-data builder (redaction pairs, fusion examples, dropout)
- `synoptiq/evaluation/scoring.py` — teacher-forced per-token NLL + log-mean-exp bottleneck estimator
- `synoptiq/evaluation/verdict.py` — **M3/M4 decision core**: `minor_agreement_test` (E2 excess-lift),
  `did_contrast`, `null_threshold` (G3 floor), `channel_recovery_gate` (G1/G2) — pure, unit-tested
- `synoptiq/evaluation/model_comparison.py` — lift/DiD statistics + MDE/power simulation (Track B)
- `synoptiq/evaluation/contamination.py` — memorization audit (perplexity gap, exact-match)
- `synoptiq/evaluation/reconstruction.py` — bag-of-tokens F1 vs nearest-witness (Track A grading)
- `synoptiq/evaluation/bootstrap.py` — pericope-grouped + paired cluster bootstrap CIs; `statistic_ci`,
  `difference_in_differences` (real-valued, for NLL-lift verdicts)
- `scripts/_cli_utils.py` — canonical `detect_device()`; all training/eval scripts import it
- `scripts/{train_dapt,eval_baseline}.py` — DAPT (+`--no-synoptics`), zero-shot vs DAPT eval
- `scripts/{prepare_study,audit_contamination,train_redactors,train_fid,run_mai_test}.py` — Phase-5 CLIs
- `modal/app_dapt.py` — Modal: `start_training`, `start_training_ns`; `modal/app_fid.py` —
  `train_redactors`, `train_fid`, `train_fid_mai` (E2 model), `run_mai` (E2 verdict)
- `docs/DIRECTION_NEGATIVE_RESULT.md` — the closed direction investigation (do not re-attempt)
- `docs/SOURCE_CRITICISM_STUDY.md` — Phase-5 preregistration (design, gates, kill criteria, freeze block)
- `docs/PAPER_PLAN.md` — the honest paper plan (supersedes a collaborator's NLL-direction outline; §5 framing rules)

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

### Model: ainouche-abderahmane/koine-t5 (published 2026-07-10)
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
tokenizer.pad_token = "<pad>"; tokenizer.eos_token = "</s>"   # do NOT add [PAD]/resize
tokenizer.add_special_tokens({"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]})
base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koine-t5")
```
**CC BY-NC-SA 4.0** (NC mirrors PROIEL's CC BY-NC-SA 3.0 — deliberate, decided 2026-07-10).
Best checkpoint (step 28000), LoRA r=64, 27.1M params, 104 MB. Card examples are all
locally verified real outputs. Task prefixes: `pos: ` / `lemma: ` / `synoptic mark_to_matt: ` /
`synoptic mark_to_luke: ` / **denoise = NO prefix** (training corrupts raw text unprefixed;
the old `denoise: ` prefix in `generate()` was a mismatch, fixed 2026-07-10). POS preds must
be `.upper()`'d (GreTa lowercases). Note: HF dropped `text2text-generation` from official
pipeline tags — the card uses `pipeline_tag: text-generation`.

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

**Source-criticism study** (Phase 5, in progress — `docs/SOURCE_CRITICISM_STUDY.md`). Two tracks
share one primitive: the **teacher-forced conditional NLL** `−log p(target | context)` (`synoptiq/
evaluation/scoring.py`); every verdict is a paired difference of these on the *same* target, so
length/style cancel (threat T1). **Redaction operators** (`models/redactor.py`): GreTa+NS+LoRA
seq2seq models `p(target|source)` — R_Lk/R_Mt (forward) and G_Mt/G_Lk (inverse) — each a learned
approximation of one evangelist's editing, fit on hypothesis-neutral triple-tradition pairs.
**Track A = Fusion-in-Decoder** (`models/fid.py`): witnesses encoded independently → encoder states
`torch.cat` along the sequence axis → one decoder cross-attends over the fusion. Trained on
(Mt+Lk)→Mark (ground truth), transferred to the double tradition for proto-Q. **Source-dropout**
training keeps one- and two-witness conditionals in the *same* weights (capacity-fairness rule).
**Track B** compares those conditionals (E2 minor-agreement lift; E1 direct-vs-bottleneck channel)
to weigh 2SH vs Farrer — never per-pair direction (`docs/DIRECTION_NEGATIVE_RESULT.md` stands).
The E2 verdict (`evaluation/verdict.py` + `scripts/run_mai_test.py`) is a paired excess-lift:
`NLL(Lk|Mk,control) − NLL(Lk|Mk,Mt)`, clustered over pericopes, plus the overlap-vs-rest DiD and
the G3 null floor — all from the scoring path, no generation. **Fold-0 result (§3b): the operators
learn strongly (~1-nat real-vs-mismatch gap), but Track A reconstruction is a bounded negative**
(FiD F1 0.31 < nearest-witness 0.56; witnesses too close to Mark) → reconstruction demoted from
headline to a reported limitation; the study leans on the operators + scoring verdicts.
M0–M2 + M3/M4 code delivered (186 tests); remaining: GPU E2 verdict + CV, G1/G2/G4, M5 E1, M6 paper.

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
