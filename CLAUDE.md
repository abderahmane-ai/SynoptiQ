# CLAUDE.md — SynoptiQ

A neural source-criticism framework for the Synoptic Problem. Applies transformers
(KoineFormer, a DAPT'd Koine-Greek T5) to the Gospels of Matthew, Mark, and Luke: a curated
parallel corpus + representation learning (Paper A, done), a general-purpose Koine model
(Koine-T5, published), a preregistered source-criticism study (Phase 5), and an honest
write-up (`paper_limit/`, drafted).

> **Copying-direction detection is a CLOSED NEGATIVE RESULT.** Phases 3 (direction scorer)
> and 6 (Bayesian hypothesis comparison) were investigated at length and **removed** on
> 2026-07-07 — inferring the direction of literary copying from the texts alone is not
> achievable (isomorphic to distinguishing a lossy projection from its inverse). Read
> `docs/DIRECTION_NEGATIVE_RESULT.md` before ever proposing anything about copying direction,
> RPM, editorial fatigue as a direction signal, or "scoring" the four source hypotheses. Do
> not re-implement them. Recover detail from git history if needed for a write-up.

## graphify (codebase knowledge graph)

This project has a knowledge graph at `graphify-out/` (git-ignored — generated locally, never
committed). It covers `synoptiq/`, `scripts/`, and `modal/`. God nodes: `Corpus`, `KoineFormer`,
`TokenRecord`. When the user types `/graphify`, invoke the `graphify` skill before anything else.

Rules:
- For any codebase question, run `graphify query "<question>"` first when `graphify-out/graph.json`
  exists — it returns a scoped subgraph, usually far smaller than grep. Use `graphify path "<A>"
  "<B>"` for relationships and `graphify explain "<concept>"` for a focused node.
- Prefer the graph over blind file browsing when tracing call paths or cross-module dependencies.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review.
- After modifying code, run `graphify update .` (AST-only, no API cost) to refresh it.

## Where things stand (read me first)

**Branch `feat/phase5-koine-t5`** — pushed to origin and merged into `main` (both at `a4d5d9a`).
**224 tests pass; repo pristine** (zero cache/`.DS_Store`; `data/ models/ outputs/ graphify-out/`
git-ignored). Four research deliverables (two published, one code-complete, one drafted) + the
**Koine Reader** tool (`synoptiq/reader/` + `spaces/koine-reader/`, built + locally verified).

**Done + published**
- **Paper A — KoineFormer + SynoptiQ corpus** (`paper/main.tex`): GreTa DAPT'd to Koine, 96.62%
  POS. Corpus = 49,061 tokens / 170 pericopes / 235 alignments. Both on HuggingFace.
- **Koine-T5** (`modal/app_koine_t5.py` · HF `ainouche-abderahmane/koine-t5` · CC BY-NC-SA 4.0):
  multitask Ancient-Greek seq2seq. Best checkpoint step 28000 → PROIEL-dev POS **96.6 NT / 87.7
  Classical / 91.7 pooled** (EM 85.2 / 52.0 / 68.6). See the Koine-T5 section.

**Code-complete (Phase 5 source-criticism study — `docs/SOURCE_CRITICISM_STUDY.md`, preregistered)**
- Operators **strong PASS** (~1-nat real-vs-mismatched-source gap; validity signal for all verdicts).
- **E2 minor-agreement verdict = SYMMETRIC NULL-after-calibration** (pooled 5-fold, N=65, both axes).
  Lk-target (Farrer axis): lift **+0.169** [0.127, 0.213] < G3 floor **0.194**; DiD +0.096 [−0.049, +0.296]
  spans 0. Mt-target (MPH axis, `mai_cv_mt` — **DONE**, results on `synoptiq-outputs:study/mai_mt/`):
  lift **+0.135** [0.097, 0.176] < floor **0.173**; DiD +0.091 [−0.063, +0.293] spans 0. Both axes below
  floor → strict independence beyond Mark; **T4 symmetry check passes** (a style artefact would clear
  both floors, a real one-way dependence one of them; neither happens) → the null is **robust-in-fact**,
  not just robust-by-design. Does NOT separate Farrer/MPH (both underpowered-null). Paper table `tab:symmetry`.
- Track A reconstruction = **bounded negative** (FiD Mt+Lk→Mark F1 0.31 < nearest-witness 0.56).
- Contamination negligible (memorization gap 0.016 ≪ 0.25; 0% verse-completion exact-match).

**The honest paper — DRAFTED (2026-07-11) at `paper_limit/main.tex`** (+ `references.bib`).
Full ~15-page manuscript reusing `paper/main.tex` styling. **Overleaf-only compile** (local TeX
lacks `pdfcol` + the fonts). Contents: **C1** negative result (Theorem 1 = direction unidentifiable
without a prior over editing operators; sign-flip demonstration) · **R1** SynoptiQ corpus · **R2**
Koine-T5 · **C2** the E2 null. Every number traced to source; **all 33 references web-verified**;
Appendix XPOS table checked against the code. Title: *Who Copied Whom? Why the Texts Cannot Tell
Us — and What They Can*. Framing rules in `docs/PAPER_PLAN.md`; memory `honest-paper-not-nll-direction`.

**Next (finish the paper)**
- **Mt-target E2 — DONE** (`mai_cv_mt`, symmetric null, `tab:symmetry`); results already on the volume,
  downloaded to `outputs/study/mai_mt/`. No re-run needed.
- **Edition-swap ablation** (T8) — Westcott–Hort / Robinson–Pierpont; the one designed robustness check left.
- M3 gates G1/G2/G4; M5 E1 is likely **not run** (underpowered at N=17 — itself a prereg outcome).
- Paper: affiliation + contact done (ENSIA); Mt-target landed; remaining is an optional figure, then Overleaf compile.

**Changelog (newest first)**
- 2026-07-13 — **Koine Reader built — the published models become a usable tool** (`synoptiq/reader/` +
  `spaces/koine-reader/`). A two-engine interlinear reading assistant: **gold mode** surfaces the on-disk
  Nestle-1904 GNT Text-Fabric data (per word: lemma, full morphology, BGVB gloss, Strong's) by reference
  lookup + **synoptic parallels** via `ALAND_PERICOPES`; **neural mode** analyses arbitrary Koine with the
  published Koine-T5 (POS+lemma, glossed against the GNT lexicon). New pkg `synoptiq/reader/`: a
  dependency-free TF reader that correctly **expands TF compact encoding** (the naive `read_tf_column`
  misaligns sparse morphology after the first caseless word) + gold reader + morphology formatting +
  parallels + index serializer + lazy neural engine. `scripts/build_reader_index.py` → a **1.9 MB** gzipped
  artifact (`spaces/koine-reader/data/n1904_index.json.gz`, 137,779 words / 5,379 lemmas) the Gradio Space
  bundles. **Space verified live** (Read tab + Mark 1:9-11 → Matt 3:13-17 / Luke 3:21-22 parallels). **30
  new tests (224 pass)**, ruff clean. Louw-Nida domains deliberately NOT surfaced (UBS copyright); confirm
  MACULA/BOL gloss redistribution license before making the Space public. Noted: Aland §011 pairs John
  1:1-18 with Matt 1:1-18 in `ALAND_PERICOPES` (surfaced faithfully; verify vs the print Synopsis).
- 2026-07-12 — **Koine-T5-Hexapla SHELVED — the generation-MAX strategy is a negative result.**
  Two GPU revs, both fail to beat the published Koine-T5. **rev 1** (two-stage curriculum, a
  generation-heavy Stage A with POS at 10% weight) collapsed POS into a prose-output basin (~**0.38**
  vs 0.966) that Stage B could not climb back out of at the decayed LR. **rev 2** (uncommitted:
  single-stage POS-favorable `TASK_WEIGHTS`, lemma gate 0.80→**0.76** = Koine-T5's measured dev-acc,
  gold/pred POS eval samples, log-quieting) fixed the collapse but **learned too slowly** to justify
  the ~3–4× compute of 512-ctx/r=128 — stopped mid-run. Conclusion: r=128 + 512-ctx + 16.8M-word diet
  + continuation task on the **GreTa-220M backbone** does not yield a better generation model — the
  ceiling is the backbone, not the diet. Koine-T5 stays the published generation story. **Salvage:**
  the dependency-free LXX Text-Fabric reader in `synoptiq/data/koine_corpus.py` (recovers **623,693**
  LXX words, fixes the "0 chunks" bug) is reusable for any future Koine work. Code kept (tested, 194
  pass), not deleted; `docs/GENERATION_PLAN.md` marked shelved. Door left open for a
  **different-backbone** attempt — not this strategy.
- 2026-07-11 — **Koine-T5-Hexapla (the MAX edition) — code-complete + CPU-validated.** New
  generation-focused model line (`modal/app_koine_hexapla.py`) that lifts free-generation quality
  while holding POS/lemma/synoptic at/above Koine-T5 via a **regression gate**. Adds: a self-contained
  **Text-Fabric reader** recovering the **623,693-word LXX** (the documented "0 chunks" bug — on-disk LXX
  is the eliranwong TF edition, not the biblicalhumanities plaintext `_parse_lxx` expects) in
  `synoptiq/data/koine_corpus.py`; `scripts/prepare_koine_maxi_corpus.py` → a **16.8M-word**
  decontaminated corpus (`data/processed/koine_maxi/`, git-ignored); a **continuation (prefix-LM)** task;
  LoRA **r=128**, **512-tok** context; two-stage curriculum + 70/30 Koine/Classical sampling;
  `evaluate_all` (pos/lemma/perplexity/token-F1/**morphological self-consistency**) with gated
  best-selection. 8 new tests (**194 pass**). Prereg `docs/GENERATION_PLAN.md`. Awaiting the GPU run.
  Housekeeping: **`paper/` + `paper_limit/` now git-ignored (local-only, removed from GitHub remote)**;
  **`AGENTS.md` deleted**; `README.md` rewritten.
- 2026-07-11 — **Mt-target E2 landed** (symmetric null): `mai_cv_mt` was already complete on
  `synoptiq-outputs:study/mai_mt/` (all 5 folds + pooled); no GPU spend. Pooled Mt-target lift
  **+0.135** [0.097, 0.176] < floor **0.173**, DiD +0.091 [−0.063, +0.293] spans 0 — mirrors the
  Lk-target null. **T4 symmetry check passes** → null is robust-in-fact. Written into `paper_limit/main.tex`
  §7.5 (new `tab:symmetry` + prose), T4 threat, Limitations, abstract, C2 bullet all updated from
  "pending" to the result. Only edition-swap (T8) robustness check remains.
- 2026-07-11 — honest paper **hardened after a full self-audit** (`paper_limit/main.tex`, prose-only,
  no fabricated numbers): E2 null reframed as *indistinguishable from the 95th-pct G3 noise band*
  (CI straddles floor) + "null ≠ positive 2SH support"; Koine-T5 clarified as a **sibling resource,
  not the reframe's engine**; its POS numbers labelled dev-set-for-selection; theorem-triviality
  defused (theorem = easy half; binding claim = no *estimable* prior); C1↔C2 seam made explicit;
  null-by-construction / no-positive-control limitation added; sign-flip softened to "illustration"
  (n=2, Chronicles translation layer noted); topology "0.22 singular readings" → token-non-alignment;
  falsified-detector code flagged as documented-not-released; MPH cite added (Huggins 1992); author
  contact line added. Open (need GPU/user): Mt-target `mai_cv_mt`, edition-swap, held-out Koine-T5
  test split, optional figure.
- 2026-07-11 — honest paper drafted, audited, refs web-verified (`paper_limit/`); Mt-target E2
  entrypoint `mai_cv_mt` added to `modal/app_fid.py`.
- `56432db` (2026-07-10) — Koine-T5 trained + published; training-bug fixes (LR-schedule units,
  T5 eval right-padding, case-fold POS eval — memory `greta-tokenizer-lowercases-pos-eval`).
- `8cbc8a1` — Koine-T5 standalone multitask trainer (PROIEL-fed).
- `f0115aa` — Phase-5 M0–M4 code + preregistration + DAPT decontamination.
- `a5b4d35` (2026-07-07) — **removed** Phase-3 direction scorer + Phase-6 Bayesian comparison
  (`docs/DIRECTION_NEGATIVE_RESULT.md`); `Corpus.direction_pairs` → `Corpus.parallel_pairs`.

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
│   │                       #   contamination audit, reconstruction F1, linear probes, verdict core
│   ├── interpretability/   # SHAP, Hawkins comparison, BERTViz
│   ├── reader/             # Koine Reader: TF reader, gold engine, morphology, synoptic parallels, index, neural
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
│   ├── pool_mai.py         # Phase 5 M4: pool per-fold rows → CV verdict
│   ├── build_reader_index.py # Koine Reader: serialize N1904 gold → gzipped Space artifact
│   └── _cli_utils.py       # Shared CLI helpers (canonical detect_device())
├── modal/                  # Modal GPU: app_dapt.py, app_fid.py (Phase 5), app_koine_t5.py, app_koine_hexapla.py
├── spaces/                 # HF Gradio Spaces: koine-t5-demo · koine-reader (interlinear GNT reader + parallels)
├── paper/                  # Paper A: KoineFormer (XeLaTeX) — LOCAL-ONLY (git-ignored, not on GitHub)
├── paper_limit/            # The honest paper (XeLaTeX) — LOCAL-ONLY (git-ignored, not on GitHub)
├── tests/                  # pytest suite (mirrors synoptiq/ structure) — 224 pass
├── data/ models/ outputs/  # All git-ignored (corpora, adapters, logs)
├── docs/DIRECTION_NEGATIVE_RESULT.md   # why copying-direction detection was closed
├── docs/SOURCE_CRITICISM_STUDY.md      # Phase 5 preregistration (Q reconstruction + source ID)
├── docs/PAPER_PLAN.md                  # the honest paper plan (negative result + corpus + Koine-T5 + E2 null)
└── {SYNOPTIQ_MASTER,IMPLEMENTATION}_PLAN.md · paper/project_overview.tex (local planning, git-ignored)
```

## Koine-T5 — standalone general-purpose model (`modal/app_koine_t5.py`)

A second, self-contained model line, **distinct from the KoineFormer encoder work above**.
`modal/app_koine_t5.py` trains **Koine-T5**: a general-purpose multitask Ancient Greek seq2seq
model (GreTa + LoRA r=64, α=128), sibling to `app_dapt.py` (which trained KoineFormer #1). One
model, **four balanced task pools** sampled every batch so none is starved:

1. `denoise` — online T5 span corruption on **raw** Greek prose
2. `pos` — POS tagging (MorphGNT tagset)
3. `lemma` — lemmatization
4. `synoptic` — Mark→Matthew / Mark→Luke style transfer (the tiny 155-pair pool, upsampled ~97×)

`pos`/`lemma`/`denoise` are fed by the Gospel corpus **plus the UD_Ancient_Greek-PROIEL treebank**
(NT Koine + Herodotus Classical, ~214K tokens). PROIEL's granular **XPOS** column is mapped to the
13-code MorphGNT tagset (`S-`→`RA` etc. — *not* UPOS/FEATS, whose `PronType=Dem` on the article is a
trap; mapping in `PROIEL_XPOS_TO_MORPHGNT`); this cures the POS "task collapse" that afflicted the
earlier tiny version. Validation is POS token-acc + Exact-Match on PROIEL dev, reported NT vs
Classical, with the best adapter saved to `best/`. Config-fingerprinted checkpoints prevent a stale
run from resuming into a changed config. Standalone (no `synoptiq` imports). Run:
`modal run modal/app_koine_t5.py::train` · `python modal/app_koine_t5.py demo` · outputs to the
`koine-t5-outputs` volume (`best/` + `final/`).

**Status (2026-07-10): TRAINED + PUBLISHED** at
[huggingface.co/ainouche-abderahmane/koine-t5](https://huggingface.co/ainouche-abderahmane/koine-t5)
(CC BY-NC-SA 4.0). Best checkpoint (step 28000) at `models/koine_t5/best/`: PROIEL-dev POS token-acc
**0.966 NT / 0.877 Classical / 0.917 pooled** (EM 0.852 / 0.520 / 0.686). NT 96.6% matches the
KoineFormer linear probe's 96.62% — but via free seq2seq generation on a different eval set (PROIEL
dev vs SynoptiQ test), so cite as comparable, not identical. Training-bug fixes (LR-schedule units,
T5 eval right-padding, case-fold POS eval) are committed in `56432db`. Koine-T5 is the honest paper's
R2 (`paper_limit/main.tex` §6). Uncommitted flag: the resume path restores only adapter weights, not
optimizer/scheduler state — fine for a fresh run, suboptimal on crash-resume.

## Koine-T5-Hexapla — the generation MAX edition (`modal/app_koine_hexapla.py`)

> **SHELVED (2026-07-12) — negative result.** Two GPU revs both failed to beat Koine-T5: rev 1
> (two-stage curriculum) collapsed POS to ~0.38; rev 2 (single-stage, uncommitted) fixed that but
> learned too slowly for the ~3–4× compute. The generation-MAX *strategy* does not lift a GreTa-220M
> backbone past Koine-T5 — the ceiling is the backbone, not the diet. **Koine-T5 remains the published
> generation model.** Code kept (not deleted); the LXX Text-Fabric reader in
> `synoptiq/data/koine_corpus.py` (623,693 words) is the reusable salvage. The section below documents
> the (shelved) design; any revival should change the backbone, not re-run this strategy.

A third model line, evolving Koine-T5 to fix the **discourse-level** failures in
`docs/gospel_of_the_savior.md` (speaker/pericope bleed, mode-collapse — all semantic, not
grammatical) **without sacrificing** POS/lemma/synoptic. Named after Origen's *Hexapla* (the
six-column parallel-scripture alignment — SynoptiQ's ancient precedent). Standalone, like
`app_koine_t5.py`. Full preregistration in `docs/GENERATION_PLAN.md`.

The pipeline: `scripts/prepare_koine_maxi_corpus.py` (built on `synoptiq/data/koine_corpus.py`)
ingests every raw source on disk — the Rahlfs LXX via a **self-contained Text-Fabric reader**
(623,693 words; the section features are dense per-word-slot so verse grouping is a `zip`), plus
first1k/apostolic/sblgnt via the proven `_extract_text_from_dir` — chunks into passage windows,
decontaminates against the Gospel test+val splits, and emits a **16.8M-word** artifact +
held-out eval splits to `data/processed/koine_maxi/` (git-ignored; upload with `--upload` →
`synoptiq-data:/koine_maxi`).

**Koine-T5 vs Koine-T5-Hexapla**

| | Koine-T5 (published) | Koine-T5-Hexapla (MAX) |
|---|---|---|
| Goal | multitask analysis + basic gen | powerful generation, **zero analysis regression** |
| Base | GreTa T5-base (220M) | GreTa T5-base (220M) |
| Adapter | LoRA r=64 α=128 (27.1M) | LoRA **r=128 α=256** (54.3M, 18%) |
| Context | 256 tok | **512 tok** |
| Tasks | denoise · pos · lemma · synoptic (4) | + **continuation/prefix-LM** (5) |
| Gen diet | Gospel + PROIEL (~263K tok) | **+16.8M words** (LXX 622K · first1k 16M · apostolic 335K · sblgnt-minus-synoptics 204K) |
| Gen objective | span-infill only | span-infill + **autoregressive continuation** |
| Curriculum | single-stage balanced | **two-stage** (rev 1 — collapsed POS to 0.38) → **single-stage** POS-favorable (rev 2) |
| Register control | none | **70/30 Koine/Classical** sampling (REGISTER_WEIGHTS) |
| Eval | POS tok-acc + EM | + lemma · perplexity · continuation token-F1 · **morphological self-consistency** |
| Selection | best POS tok-acc | **regression-gated**: gates (POS-NT≥0.966, POS-Cl≥0.877, lemma≥gate) → maximize gen score |
| Eff. batch / steps | 4×8=32 / 30K | 2×16=32 / 40K |
| Volume | koine-t5-outputs | koine-hexapla-outputs |
| Status | ✓ trained + published | ✗ **SHELVED** — 2 revs, both < Koine-T5 (see banner) |

The **no-sacrifice** guarantee is mechanical: `evaluate_all`'s best-selection key is `(1, f1+morph)`
once all gates pass, else `(0, pos_tok)` — any gate-passer outranks any non-passer, and among passers
the generation score decides. (`GATE_LEMMA` was corrected 0.80→**0.76** in rev 2 = Koine-T5's measured
lemma dev-acc under this harness; the 0.80 placeholder sat *above* Koine-T5 itself and was unmeetable.)
Run: `modal run modal/app_koine_hexapla.py::train` ·
`python modal/app_koine_hexapla.py demo models/koine_hexapla/best`.

## Current status

| Component | Status | Key result |
|-----------|--------|------------|
| Phase 0–1 | ✓ Foundation + data | SynoptiQ corpus: 49,061 tokens, 170 pericopes, 235 alignments |
| Phase 2A  | ✓ KoineFormer DAPT | 96.62% POS, 81.34% lemma, 14 MB adapter (Paper A) |
| Phase 2B  | ○ Multi-task encoder | Code ready, not yet trained |
| Phase 3/6 | ✗ Removed | Direction detection + Bayesian scoring — **closed negative** (`docs/DIRECTION_NEGATIVE_RESULT.md`) |
| Phase 5   | ◐ Code-complete | Operators strong PASS; **E2 = symmetric null-after-calibration** (Lk-target +0.169 < floor 0.194; Mt-target +0.135 < floor 0.173; both DiD span 0; T4 symmetry passes → robust-in-fact); Track A bounded negative. Remaining: edition-swap, gates (`docs/SOURCE_CRITICISM_STUDY.md`) |
| Phase 7   | ○ Interpretability | Not started |
| Koine-T5  | ✓ Published | 96.6 NT / 91.7 pooled POS; HF, CC BY-NC-SA 4.0 |
| Koine-T5-Hexapla | ✗ Shelved | Generation-MAX strategy = **negative result**: 2 GPU revs, both < Koine-T5 (rev 1 POS-collapse 0.38; rev 2 too-slow). Ceiling is the GreTa-220M backbone, not the diet. Code kept; LXX TF reader salvageable (`docs/GENERATION_PLAN.md`) |
| Paper A   | ✓ Draft | `paper/main.tex` — complete, verified numbers (local-only) |
| Honest paper | ◐ Drafted | `paper_limit/main.tex` — negative result + corpus + Koine-T5 + E2 null; Overleaf-only; refs web-verified |

## Key files to know

- `synoptiq/utils/types_.py` — shared TypedDicts (`TokenRecord`, `PericopeAlignment`, `MorphRecord`,
  `SplitResult`), `Book`/`Tradition`/`Genre` literals, Protocols
- `synoptiq/utils/constants.py` — **Aland pericope table** (bedrock), MorphGNT tagset maps,
  `MARK_Q_OVERLAP_{CORE,EXTENDED}`
- `synoptiq/utils/greek.py` — Greek text normalization (NFD accent stripping, sigma normalization)
- `synoptiq/data/corpus.py` — central `Corpus` class. `parallel_pairs(tradition=…, split=…)` yields
  `(book_a, tokens_a, book_b, tokens_b, alignment)`; `iter_parallel_pairs` also yields the pericope_id
- `synoptiq/data/alignment.py` — Needleman-Wunsch token alignment via Bio.Align.PairwiseAligner.
  Scoring is **binary on the (normalized lemma, POS) key** (identical = match +2.5, else −100 → gap).
- `synoptiq/data/{pericope,splits}.py` — tradition classification + pericope-atomic stratified splits
- `synoptiq/data/study_design.py` — Phase-5 folds/census/overlap partition/prereg hashing
- `synoptiq/data/redaction.py` — Phase-5 training-data builder (redaction pairs, fusion examples, dropout)
- `synoptiq/models/koineformer.py` — **KoineFormer**: GreTa + LoRA wrapper, factory, save/load, generate
- `synoptiq/models/encoder.py` — MultiTaskEncoder (POS / biaffine parser / pericope heads) for Phase 2B
- `synoptiq/models/redactor.py` — **Redactor** (Phase 5): source→target seq2seq LoRA operator; the four
  instances R_Lk/R_Mt/G_Mt/G_Lk differ only in training data. `score()` = teacher-forced conditional NLL
- `synoptiq/models/fid.py` — **FusionInDecoder** (Phase 5, Track A): encodes witnesses separately, `torch.cat`
  their encoder states, decoder cross-attends over the fusion → reconstruct Mark from Mt+Lk (then proto-Q)
- `synoptiq/training/_config.py` — frozen dataclasses (`DataConfig`, `ModelConfig` [incl. `fid_*`],
  `TrainingConfig`, `DAPTConfig`, **`StudyConfig`** [the hashed Phase-5 prereg artifact])
- `synoptiq/training/dapt.py` — **DAPT**: data loader (+ `exclude_books` decontamination) + AMP loop
- `synoptiq/evaluation/scoring.py` — teacher-forced per-token NLL + log-mean-exp bottleneck estimator
- `synoptiq/evaluation/verdict.py` — **M3/M4 decision core**: `minor_agreement_test` (E2 excess-lift),
  `did_contrast`, `null_threshold` (G3 floor), `channel_recovery_gate` (G1/G2) — pure, unit-tested
- `synoptiq/evaluation/model_comparison.py` — lift/DiD statistics + MDE/power simulation (Track B)
- `synoptiq/evaluation/contamination.py` — memorization audit (perplexity gap, exact-match)
- `synoptiq/evaluation/reconstruction.py` — bag-of-tokens F1 vs nearest-witness (Track A grading)
- `synoptiq/evaluation/bootstrap.py` — pericope-grouped + paired cluster bootstrap CIs; `statistic_ci`,
  `difference_in_differences` (real-valued, for NLL-lift verdicts)
- `scripts/_cli_utils.py` — canonical `detect_device()`; all training/eval scripts import it
- `scripts/{prepare_study,audit_contamination,train_redactors,train_fid,run_mai_test,pool_mai}.py` — Phase-5 CLIs
- `modal/app_dapt.py` — `start_training`, `start_training_ns`, `run_ablation`
- `modal/app_fid.py` — `train_redactors`, `train_fid`, `train_fid_mai`, `run_mai`, `mai_cv` (Lk-target E2 CV),
  **`mai_cv_mt`** (symmetric Mt-target E2 CV — Farrer vs MPH + T4 symmetry check)
- `modal/app_koine_t5.py` — `train`, `generate`, `build_tokenizer` (+ `demo` local)
- `paper_limit/main.tex` — **the honest paper** (Theorem 1 + sign-flip + corpus + Koine-T5 + E2 null)
- `docs/DIRECTION_NEGATIVE_RESULT.md` · `docs/SOURCE_CRITICISM_STUDY.md` · `docs/PAPER_PLAN.md`

## Phase 2A results (Paper A)

### POS + Lemma tagging (linear probe, SynoptiQ test set)

| Model | POS Acc. | Lemma Acc. | Params | Checkpoint |
|-------|----------|------------|--------|------------|
| GreTa zero-shot | 95.32% | 82.37% | 0 | 880 MB |
| Full fine-tune (220M) | 96.11% | — | 220M | 880 MB |
| **KoineFormer LoRA** | **96.62%** | 81.34% | **3.7M** | **14 MB** |

Headline: LoRA DAPT eliminates 28% of POS errors vs zero-shot, beats full FT on accuracy, and
produces a 14 MB checkpoint. Lemma is flat — DAPT improves syntax but not vocabulary.

### DAPT corpus + config
- Koine (70%): SBLGNT full NT (~773K tokens) + Apostolic Fathers (~732K tokens) ≈ 1.5M tokens
- Classical replay (30%): First1KGreek (Homer, Plato, Xenophon). LXX yields 0 chunks (TextFabric
  `.tf` files are metadata, not Greek text).
- GreTa (T5-base, 220M) frozen, LoRA r=16 α=32 targeting `["q","v","o","wi","wo"]`. Note: `wi` does
  NOT apply (PEFT uses `endswith`; `wi` doesn't match `wi_0`/`wi_1`) → actual targets W_q/W_v/W_o +
  W_o(FFN) = 3.7M trainable. 20,000 steps, batch 8, seq_len 512, AMP, AdamW lr=1e-4 cosine→0. A10G,
  58 min, crash-safe (SIGTERM handler).

## Modal commands

```bash
# KoineFormer DAPT (app_dapt.py)
modal run modal/app_dapt.py::upload_data                    # upload data (once)
modal run modal/app_dapt.py::start_training                # train KoineFormer (auto-resumes)
modal run modal/app_dapt.py::start_training_ns             # KoineFormer-NS (gospels excluded)

# Phase-5 study (app_fid.py)
modal run modal/app_fid.py::mai_cv                         # Lk-target E2 5-fold CV verdict (DONE)
modal run modal/app_fid.py::mai_cv_mt                      # Mt-target E2 CV (symmetric null — DONE)

# Koine-T5 (app_koine_t5.py)
modal run modal/app_koine_t5.py::train                     # train Koine-T5 (auto-resumes)

# Download outputs (dir quirk: `modal volume get` mangles directory downloads →
#   fetch files individually; audit/loader scripts auto-resolve nested adapter paths)
modal volume get synoptiq-outputs dapt/ models/koineformer/dapt/
```

Modal volumes: `synoptiq-data` (`/data/`), `synoptiq-outputs` (`/outputs/{dapt,dapt_ns,study,…}`),
`koine-t5-outputs` (Koine-T5 `best/` + `final/`).

## HuggingFace

### Model: ainouche-abderahmane/koineformer (CC-BY-SA 4.0)
```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM
base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koineformer").merge_and_unload()
```
96.62% POS, 81.34% lemma, 14 MB adapter.

### Model: ainouche-abderahmane/koine-t5 (CC BY-NC-SA 4.0; NC mirrors PROIEL — deliberate)
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
tokenizer.pad_token = "<pad>"; tokenizer.eos_token = "</s>"   # do NOT add [PAD]/resize
tokenizer.add_special_tokens({"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]})
base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koine-t5")
```
Best checkpoint (step 28000), LoRA r=64 α=128, 27.1M params, 104 MB. Task prefixes: `pos: ` /
`lemma: ` / `synoptic mark_to_matt: ` / `synoptic mark_to_luke: ` / **denoise = NO prefix**. POS
preds must be `.upper()`'d (GreTa lowercases). Card uses `pipeline_tag: text-generation`.

### Dataset: ainouche-abderahmane/synoptiq-corpus (CC-BY-SA 4.0)
```python
from datasets import load_dataset
ds = load_dataset("ainouche-abderahmane/synoptiq-corpus")
# ds["train"] → 27,289 tokens, ds["validation"] → 9,170, ds["test"] → 10,618
```
49,061 tokens, 170 pericopes, 235 alignments.

### GitHub: github.com/abderahmane-ai/SynoptiQ

## Tokenizer notes

- GreTa SentencePiece ships `<pad>`=0 and `</s>`=1. For the **encoder** work (KoineFormer probes) a
  `[PAD]` token is added + embeddings resized; for **seq2seq scoring/generation** (redactor/FiD,
  Koine-T5) do NOT add `[PAD]`/resize — it desyncs the pad id from the decoder-start id and collapses
  generation to empty (and inflates NLL ~4×). Use the existing `<pad>`/`</s>`.
- Koine text tokenizes at 1.38 subwords/word (Classical: 1.95) — simpler morphology.
- Nomina sacra (Ἰησοῦς, Χριστός, κύριος, θεός) are single tokens.
- Subword-to-word alignment uses `▁` prefix (U+2581) for SentencePiece word boundaries.

## Paper compilation

```bash
cd paper && xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex   # Paper A
cd paper_limit && xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex  # honest paper
```
Custom fonts (Poppins, TeX Gyre Pagella, FreeSerif for Greek), XeLaTeX. Palette: marine (#123C48),
terracotta (#B75B36), papyrus (#F7F2E7). **Compile on Overleaf** — local TeX lacks `pdfcol.sty` and
the fonts (memory `paper-compile-on-overleaf`; do not retry local `xelatex`).

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
Ruff TC, UP, RUF, ANN, B90 warnings are cosmetic — only fix F and E. (`modal/` is not in the lint
gate; its `_build_image() -> Any` ANN401 is expected.)

## Architecture summary

**KoineFormer** = GreTa (T5 encoder-decoder, 220M, Classical Greek) after PEFT-DAPT on a Koine corpus
(SBLGNT + Apostolic Fathers ~1.5M tokens). LoRA only (~3.7M trainable, r=16 α=32). 96.62% POS.

**Koine-T5** = GreTa + LoRA (r=64 α=128, 27.1M) multitask (denoise/POS/lemma/synoptic) on the Gospel
corpus + PROIEL. 96.6 NT / 91.7 pooled POS.

**Source-criticism study** (Phase 5 — `docs/SOURCE_CRITICISM_STUDY.md`). One primitive: the
**teacher-forced conditional NLL** `−log p(target | context)` (`evaluation/scoring.py`); every verdict
is a paired difference on the *same* target, so length/style cancel (threat T1). **Redaction operators**
(`models/redactor.py`): GreTa+NS+LoRA `p(target|source)` — R_Lk/R_Mt (forward), G_Mt/G_Lk (inverse) —
each a learned approximation of one evangelist's editing, fit on hypothesis-neutral triple-tradition
pairs. **Track A = Fusion-in-Decoder** (`models/fid.py`): witnesses encoded independently → encoder
states `torch.cat` → one decoder cross-attends over the fusion → reconstruct Mark from Mt+Lk.
**Source-dropout** keeps one- and two-witness conditionals in the *same* weights (capacity-fairness).
**Track B** compares those conditionals (E2 minor-agreement lift; E1 direct-vs-bottleneck channel) to
weigh 2SH vs Farrer — never per-pair direction (`docs/DIRECTION_NEGATIVE_RESULT.md` stands). The E2
verdict (`evaluation/verdict.py` + `scripts/run_mai_test.py`) = paired excess lift
`NLL(Lk|Mk,control) − NLL(Lk|Mk,Mt)`, clustered over pericopes, + overlap-vs-rest DiD + G3 floor.
**Result: operators strong PASS (~1-nat gap); E2 null-after-calibration; Track A bounded negative.**

**Interpretability** (Phase 7, later) = SHAP feature importance vs Hawkins (1899); BERTViz attention;
multi-edition sensitivity (NA28/TR/Majority/WH).

## Tech stack

- Python 3.12+, PyTorch 2.6+, HuggingFace transformers, PEFT
- Modal (GPU cloud: A10G)
- BioPython (token alignment), SHAP + BERTViz (interpretability)
- ruff (linting), pytest, XeLaTeX (papers)
- Data: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), Apostolic Fathers, First1KGreek (CC-BY-SA),
  UD_Ancient_Greek-PROIEL (CC BY-NC-SA)
- GitHub: [abderahmane-ai/SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ)
- HF: [koineformer](https://huggingface.co/ainouche-abderahmane/koineformer) ·
  [koine-t5](https://huggingface.co/ainouche-abderahmane/koine-t5) ·
  [synoptiq-corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus)
```
