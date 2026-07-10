# Source-criticism study — Q reconstruction (Track A) + source identification (Track B)

*(Project roadmap: this is Phase 5.)*

**Status: DESIGN / PREREGISTRATION DRAFT — not yet frozen.**
This document is written to double as a preregistration: §6 (decision rules) and §7 (kill
criteria) must be frozen — committed, hash recorded in §10 — *before* any double-tradition
result is computed.

---

## 0. Relation to the closed negative result

`docs/DIRECTION_NEGATIVE_RESULT.md` stands unchanged. This phase does **not**:

- score per-pair copying direction from the texts alone;
- use RPM-style textual-criticism canons, MDL/codelength asymmetry, or any corpus-free
  direction feature;
- score the four classical source hypotheses (Griesbach and Augustinian remain excluded by
  the surviving topology result and are not revisited);
- train on synthetic redaction corpora (documented generator-artifact trap).

What changed is the *question*. The impossibility result concerns inferring direction from a
pair of texts with no prior over the redactor. This phase instead compares **generative models
of the double tradition** whose redactor-specific operators are learned from supervision that
**every live hypothesis accepts as correctly labeled** — so nothing here assumes what it sets
out to test. Independent support for the closure, found in the 2026-07-07 literature review:
supervised direction detection in the translationese literature reaches 83–100% in-domain and
collapses to ~60% cross-domain (Sominsky & Wintner 2019; arXiv:1609.03205), exactly the
genre-limit pattern the Phase-3 RootModel exhibited. The design below never leaves the corpus
domain for its verdicts.

## 1. Question and hypothesis space

The topology result (Mark is the agreement-structure hub; Mark's ~0.22 singular-reading rate
excludes conflation hypotheses) plus scholarly near-consensus leaves a live space in which
**all members agree on Mark's priority in the triple tradition**:

| Hypothesis | Triple tradition (TT) | Double tradition (DT) | Mt↔Lk contact in TT? |
|---|---|---|---|
| **2SH** (Two-Source) | Mk→Mt, Mk→Lk | Mt←Q→Lk (latent common source) | No (minor agreements = noise/text-critical) |
| **Farrer** | Mk→Mt, Mk→Lk | Mt→Lk (Luke used Matthew) | Yes (Luke consulted Mt throughout) |
| **MPH** (Matthean Posteriority, Garrow) | Mk→Mt, Mk→Lk | Lk→Mt (Matthew used Luke) | Yes (symmetric: Matthew consulted Lk) |
| **3SH** (Three-Source) | Mk→Mt, Mk→Lk | Q + Luke also used Matthew | Yes (weaker) |

Two consequences we exploit:

1. **The TT is hypothesis-neutral, direction-labeled training data.** 65 *full triples*
   (of 88 triple-tradition pericopes; 23 lack one book's tokens after alignment), with
   ~8.5k Mark tokens aligned to ~9.0k Matthew and ~7.9k Luke tokens, all with agreed direction
   (Mk→Mt, Mk→Lk), same authors, same genre-family, same period. This is precisely the
   "prior over each author's injection operator" that the negative-result doc says direction
   inference requires — obtained without assuming anything the live hypotheses dispute.
2. **The hypotheses make *opposite* predictions about measurable dependence structure**, not
   about an unmeasurable direction bit:
   - Under 2SH: in the TT, Lk ⊥ Mt | Mk (minor agreements aside), *except* in Mark-Q overlap
     pericopes where genuine Mt↔Lk information beyond Mark is expected.
   - Under Farrer: Mt carries information about Lk beyond Mk *uniformly* across the TT.
   - Under MPH: the mirror image.

## 2. Corpus census (computed from `data/processed/` — reproduce with `scripts/prepare_study.py freeze`)

The naive count and the *usable* count differ, and the difference is load-bearing:

| Tradition | Pericopes | usable units | Matthew tok | Mark tok | Luke tok |
|---|---|---|---|---|---|
| triple | 88 | **65 full triples** (Mt & Mk & Lk all present) | 8,980 | 8,484 | 7,870 |
| double | 17 | **17** (all have Mt & Lk) | 4,041 | — | 2,913 |
| matthean_unique (M) | 18 | — | 2,377 | — | — |
| lukan_unique (L) | 45 | — | — | — | 7,155 |
| mark_unique | 2 | — | — | 140 | — |

23 nominal triples lack one book's tokens after alignment and cannot train a three-way
operator — the effective TT sample is **65, not 88** (token totals above are over the 65).

Genre strata over the **usable** units (full triples / double tradition):
TT = 37 other / 13 passion / 11 narrative / 4 discourse / **0 wisdom**;
DT = 9 other / 4 **wisdom** / 3 discourse / 1 narrative. The `wisdom` stratum (4 of 17 DT
pericopes) has **zero triple-tradition analog** — the operators cannot be supervised on it, so
it is flagged *unidentifiable a priori* and excluded from any per-genre claim (§5-T3). Mark-Q
overlaps present as full triples: **5 core, 9 core+extended.**

Design consequence (this reverses the naive attack order): **N=17 independent DT units is
the power bottleneck; N=65 TT units is where discrimination is feasible.** Therefore the
primary experiment is the TT minor-agreement information test (E2), and the DT channel test
(E1) is confirmatory.

## 3. Formal setup

Notation, per aligned pericope *p*: token sequences `Mt_p`, `Mk_p`, `Lk_p`. All models are
KoineFormer-NS (see §5-T7) seq2seq scorers/generators; all likelihoods are teacher-forced
per-token NLLs aggregated per pericope.

Learned components (all trained **only on TT pericopes**, cross-validated, DT never touched):

- `R_Lk` — Luke's redaction operator: seq2seq `source → Lk`, trained on (Mk_p → Lk_p).
- `R_Mt` — Matthew's redaction operator: seq2seq `source → Mt`, trained on (Mk_p → Mt_p).
- `G_Mt` — single-witness source reconstructor: `Mt → source`, trained on (Mt_p → Mk_p).
- `G_Lk` — single-witness source reconstructor: `Lk → source`, trained on (Lk_p → Mk_p).
- `F` — Fusion-in-Decoder two-witness reconstructor: `(Mt, Lk) → source`, trained on
  ((Mt_p, Lk_p) → Mk_p). `F` is Track A's deliverable and E2's scoring backbone.

**Capacity fairness rule (binding):** every comparison uses *one* shared scoring network per
target author; competing hypotheses differ **only in what is placed in the encoder context**,
never in architecture, parameters, or training data. Trained with source-dropout (random
subsets of available witnesses in the encoder) so that one-source and two-source conditionals
come from the same weights.

### E2 (PRIMARY) — Minor-Agreement Information test, on the TT (N = 65 full triples)

Does Matthew carry predictive information about Luke *beyond Mark*, where both hypotheses
agree Mark is the source?

Per held-out pericope (k-fold CV over TT pericopes, folds frozen in `folds.json`):

```
Δ_p(Lk)  =  NLL(Lk_p | Mk_p)  −  NLL(Lk_p | Mk_p, Mt_p)         # lift from adding Matthew
Δ̃_p(Lk) =  NLL(Lk_p | Mk_p)  −  NLL(Lk_p | Mk_p, C_p)          # lift from adding a control
```

with negative-control contexts `C_p`: (a) Matthew text from a *different, genre-matched*
pericope; (b) a length-matched LXX narrative chunk. The verdict statistic is the paired
excess lift `Δ_p − Δ̃_p`, cluster-bootstrapped over pericopes. The symmetric test (targets
Matthew, adds Luke) runs **always and unconditionally** (§5-T4).

**Difference-in-differences sharpening.** Partition TT into Mark-Q overlap pericopes
(standard list — Baptist's preaching, temptation, Beelzebul, mustard seed, mission charge,
sign demand, …— mapped to Aland IDs and frozen in `constants.py` before unblinding) versus
the rest. The hypotheses now predict an *internal contrast*:

- 2SH: excess lift **only** in overlap pericopes (Matthew used Q there; elsewhere Lk ⊥ Mt | Mk).
- Farrer: excess lift **uniform** (Luke consulted Matthew everywhere), asymmetric (Lk-target only).
- MPH: uniform excess lift, Mt-target only.
- 3SH: uniform lift with overlap excess on top.

This contrast is internally controlled — genre, style, contamination, and model pathologies
hit both partitions equally. It is the single most bias-resistant statistic in the design.

Localization check: alignments (`alignments.json`) let us map where the lift concentrates.
If real, it should concentrate on catalogued minor agreements (Neirynck 1974). This is a
mechanism-transparency check, not a verdict input.

### E1 (CONFIRMATORY) — Channel test, on the DT (N = 17)

Both live single-direction hypotheses and 2SH disagree about the *channel* Mt→Lk:

- Farrer: `p(Lk_p | Mt_p) = R_Lk(Lk_p | Mt_p)` — direct redaction, same operator Luke
  applied to Mark.
- 2SH: `p(Lk_p | Mt_p) = E_{Q ~ p(Q|Mt_p)}[ R_Lk(Lk_p | Q) ]` — a bottleneck through the
  reconstructed common source, with `p(Q|Mt_p)` approximated by `G_Mt` samples
  (K ∈ {1, 5, 10} sensitivity; log-mean-exp aggregation).

Verdict statistic, per DT pericope:

```
Λ_p = log p̂_bottleneck(Lk_p | Mt_p) − log p̂_direct(Lk_p | Mt_p)
```

Same scorer `R_Lk` on both branches — only the encoder context differs (reconstructed-source
samples vs Matthew verbatim), so length, style, and memorization of the *target* cancel to
first order. Mirror statistic with `R_Mt`/`G_Lk` for MPH runs unconditionally. Note the
acknowledged plug-in bias: `G_Mt` reconstructs *Mark-flavored* sources; if the true Q was
stylistically closer to Matthew, the bottleneck branch is penalized. Gate G2 and the fidelity
sweep (§5-T6) bound this empirically; it is also stated as a limitation in any write-up.

### E3 — Calibration gates (run before any verdict; all on known-answer data)

- **G1 (direct-channel recovery):** on held-out (Mk_p, Lk_p) TT pairs, the machinery must
  prefer the direct channel Mk→Lk over a bottleneck through `G` reconstructions. Both
  hypotheses agree direct is true here.
- **G2 (latent-channel recovery):** on held-out (Mt_p, Lk_p) TT pairs *with Mark hidden*,
  the machinery must prefer the bottleneck (they do share a latent-ish source: Mark). Both
  hypotheses agree the dominant channel here is common-source.
- **G3 (null lift):** E2 negative controls must show excess lift statistically
  indistinguishable from zero — this defines the empirical noise floor.
- **G4 (external, secondary):** Josephus *Antiquities* 12–13 ← 1 Maccabees (direction
  certain, Greek→Greek narrative paraphrase, openly available) as an out-of-domain direct-
  dependence sanity case. Failure of G4 alone does not kill the phase (genre transfer is a
  *known* limit); it calibrates how much the external world can be leaned on (answer per the
  Phase-3 record: little).
- **Empirical power analysis (DELIVERED — `scripts/prepare_study.py power`, results in §3a):**
  a simulation over the corpus's real per-pericope target-token weights sweeps the per-token
  signal-to-noise ratio and reports each test's minimum detectable effect (MDE) at 80% power.
  This is kill criterion K2's input. Once the gates fix the empirical noise floor (G3), the
  gate-demonstrated effect is compared to these MDE curves; if it falls below E1's MDE, E1 is
  never run. **Computed before DT unblinding, and binding.**

### 3a. Power-analysis results (delivered)

Per-token snr = mean per-token likelihood lift ÷ its per-token SD (dimensionless; the absolute
nats scale is fixed later by G3). τ = between-pericope effect heterogeneity in the same units —
the quantity a *cluster* bootstrap exists to capture. MDE = smallest snr reaching 80% detection.

| τ (between-pericope) | E2 lift (N=65) | E2 DiD (5 vs 60) | E1 channel (N=17) |
|---|---|---|---|
| 0.00 | 0.05 | 0.15 | 0.10 |
| 0.10 | 0.10 | 0.20 | 0.10–0.15 |
| 0.20 | 0.10 | 0.30 | 0.20 |
| 0.30 | 0.15 | 0.40–0.45 | 0.25 |

Three findings that shape the phase:

1. **Power is set by pericope *count* and effect heterogeneity, not token counts.** Pericopes
   are large enough (median 96 Luke tokens in the TT, 165 in the DT) that within-pericope
   sampling noise is negligible; the MDE is driven almost entirely by N and τ. More Greek per
   pericope would not help — more *independent pericopes* would.
2. **The DiD contrast is the true bottleneck**: its 5-pericope overlap partition needs ~3× the
   per-token effect of the whole-TT lift test. The DiD is the most bias-resistant statistic but
   also the least powerful — report it with this MDE attached, never as a bare null.
3. **The percentile bootstrap is mildly anti-conservative at N=17** (false-positive rate ≈
   0.07–0.08 vs nominal 0.05). E1 claims therefore use a stricter threshold (BCa correction and
   the G3-derived floor), not the raw 95% interval.

### 3b. Results log (append-only; delivered runs)

**M1 contamination audit (2026-07-08).** KoineFormer-NS trained (`excluded_files=12`).
Memorization gap (log-ppl DiD, original vs NS) **0.016** (flag threshold 0.25); verse-completion
exact-match **0%**; gospel/control perplexity 2.08/1.66 (orig) ≈ 2.12/1.66 (NS). **Contamination
negligible** — LoRA's 1.5% capacity cannot memorize verbatim; the likelihood approach is safe.

**M2 operators + FiD, fold 0 (2026-07-08).** *A loader bug (a spurious `[PAD]` token + embedding
resize that desynced the tokenizer pad id from the model's decoder-start id) had degraded the
first run to ~8 nats and empty generation; fixed by wiring GreTa's existing `<pad>`=0/`</s>`=1.*
Corrected fold-0 results:

- **Redaction operators — strong PASS.** Held-out NLL (nats/token), all four beating both baselines:
  R_Lk 1.995 (mismatch 3.094, free 11.6); R_Mt 1.978 (2.818, 37.6); G_Mt 2.182 (3.055, 38.4);
  G_Lk 2.032 (2.996, 38.3). The ~1-nat real-vs-mismatched-source gap is the key validity signal —
  the operators use pericope-*specific* source content, not just target-language style.
- **Track A reconstruction — bounded negative.** FiD (Mt+Lk→Mark) held-out token-F1 **0.311**
  [CI 0.24, 0.38] at 15 epochs, vs nearest-witness baseline **0.562**. Train loss kept falling
  (2.17→1.28) while held-out F1 barely moved (0.279→0.311): overfitting, not underfitting. **The
  witnesses are already ~56% overlapping with Mark, so abstractive fusion cannot beat extractive
  copying at this data scale.** Consequence (below): the reconstruction is demoted from headline
  to a reported limitation; the study leans on the operators + the scoring-based verdicts. The
  scoring path is unaffected (generation ≠ scoring), so E2/E1 remain viable.

**M4 E2 minor-agreement test, fold 0 (2026-07-09).** Source-dropout FiD on (Mk,Mt)→Lk; per-pericope
excess lift `NLL(Lk|Mk,control) − NLL(Lk|Mk,Mt)`. Raw excess **+0.182** [CI 0.108, 0.258], P(>0)=1.00
— *would read as strongly significant on a naive test*. **But it does NOT clear the G3 null floor
0.242** (95th-pct excess between two mismatched Matthews): the real Matthew's advantage is smaller
than the noise two wrong Matthews generate. **So after calibration, no minor-agreement signal at
fold 0** — the pre-registration's floor caught what would have been a published false positive.
Caveats: single fold, only **1 overlap pericope** in the held-out set (DiD uninterpretable), and the
small-N floor is inflated. The verdict is the **5-fold pooled** run below.

**M4 E2 pooled CV verdict (2026-07-09) — the preregistered E2 result.** `modal ::mai_cv` trained
and scored all 5 folds (`scripts/pool_mai.py` → `outputs/study/mai/mai_pooled.json`); 65 held-out
pericopes, all 5 overlap pericopes in one DiD. **Pooled excess lift +0.169** [CI 0.127, 0.213],
P(>0)=1.00 — again *naively significant*. **But it does NOT clear the G3 null floor 0.194**
(0.169 < 0.194): Matthew's real advantage over Mark stays inside the noise band two *mismatched*
Matthews generate. **DiD overlap(5) vs rest(60) +0.096** [CI −0.049, +0.296] — spans zero, **not
overlap-concentrated.** Both independent axes are therefore null-after-calibration: no calibrated
minor-agreement signal, and no overlap concentration. This confirms and generalizes fold 0 across
the full CV — the pre-registration's floor and DiD together kill what a naive test would have
published as a Farrer-supporting minor-agreement effect. Per the matrix (§4), this reads as the
**strict-independence-beyond-Mark** row: minor agreements behave as text-critical/stylistic noise,
a publishable null. (Single caveat retained: per-fold G3 floors are small-N-inflated; the pooled
floor 0.194 is the operative one. The symmetric Mt-target run and edition-swap ablation remain to
convert "null" into "null and robust.")

## 4. Outcome interpretation matrix (fixed in advance — every cell publishable)

E2 core = excess lift on TT-minus-overlap; E2 DiD = overlap-vs-rest contrast.

| E2 core (Lk-target) | E2 DiD | E1 (N=17) | Reading |
|---|---|---|---|
| lift ≈ 0 | overlap-only excess | bottleneck favored | **2SH** supported on both independent axes |
| lift > 0, Lk-target only | uniform | direct favored | **Farrer** supported |
| lift > 0, Mt-target only | uniform | direct favored (mirror) | **MPH** supported |
| lift > 0, both targets | mixed | any | Contact or shared non-Mark tradition; discuss 3SH / oral tradition / assimilation — no strong claim |
| lift ≈ 0 | flat | inconclusive | Strict independence beyond Mark; MAs are text-critical noise — a *result*, publishable |
| gates G1/G2 fail | — | — | Machinery cannot discriminate at this scale → negative-methods paper (K1) |

Mixed E1/E2 outcomes: E2 (N=65) dominates the reading; E1 is reported with its
pre-registered power and never overrides E2.

## 5. Threats to validity — each Phase-3 trap mapped to a design counter

| # | Threat (Phase-3 record / new) | Counter in this design |
|---|---|---|
| T1 | Length/style asymmetry masquerading as signal (the NLL/MDL trap) | All verdicts are **paired differences of NLL of the same target sequence** under different encoder contexts; length and target style are common to both branches |
| T2 | Generator artifacts from synthetic data | No synthetic training data anywhere in the phase |
| T3 | Genre-transfer failure (RootModel trap; translationese literature) | Verdicts trained and evaluated within-corpus; genre-stratified reporting (the `wisdom` stratum flagged as unidentifiable a priori); E2's DiD contrast is genre-controlled by construction |
| T4 | Author-style detector masquerading as direction (the Markan-style/RPM trap) | Symmetric execution: every test always runs in both target directions; an author-style artifact produces *symmetric* lift and is read as "no directional claim" (matrix row 4) |
| T5 | Latent-variable flexibility bias (a latent Q can always fit better) | Conditional formulation (no free latent optimization at test time — `G` is frozen, trained on TT); net pipeline bias *measured empirically* on known-answer gates G1/G2 rather than argued theoretically |
| T6 | Operator-transfer assumption — "Luke treats Matthew as he treats Mark" (Hägerland 2019's objection to fatigue arguments) | Treated as a *sensitivity parameter*, not an assumption: sweep a fidelity/temperature interpolation between `R_Lk` and a free-composition Lukan LM (trained on L material + Acts); report the verdict as a function of it; verdict claimed only if stable across the plausible range |
| T7 | **Contamination**: DAPT corpus included the full NT (`sblgnt` in `DAPTConfig.corpus_components`) — KoineFormer memorized the evaluation texts | **KoineFormer-NS**: rerun DAPT excluding Mt/Mk/Lk (and Josephus War/Ant parallels + 1 Macc if G4 is run) — ~1 hr A10G on the existing Modal pipeline. Audit GreTa's base pretraining for NT presence (n-gram probes; perplexity gap vs matched unseen Koine). Residual memorization is bounded by the paired-difference design (T1) and the edition-swap ablation (T8) |
| T8 | Textual assimilation: later scribes harmonized Luke toward Matthew, faking MAs | Edition-swap sensitivity: rerun E2 on Westcott–Hort and Robinson–Pierpont texts (public domain); MAs that vanish under edition swap are attributed to transmission, not composition |
| T9 | Sparsity/power (the "double tradition is too sparse" conclusion) | Pre-registered empirical power analysis (§3-E3); if MDE > gate-demonstrated effect size, E1 is **not run** and the sparsity conclusion is confirmed quantitatively (K2) |
| T10 | Post-hoc analysis drift (the Phase-3 meta-failure: no stopping rules) | Preregistration protocol (§6), single-shot DT unblinding, kill criteria (§7) |

## 6. Preregistration protocol

1. All modeling iteration (architecture, hyperparameters, prompts, K, fold structure) happens
   on TT training folds and gates only. **The DT is scored once**, under a config whose hash
   is committed in §10 beforehand.
2. `scripts/run_channel_test.py` (E1) hard-refuses to run unless: (a) a passing gates report
   exists in `outputs/study/gates/`; (b) the runtime config hash equals the frozen hash;
   (c) sentinel `outputs/study/DT_UNBLINDED` does not exist. On completion it writes the
   sentinel and the result artifact. Re-runs require deleting the sentinel by hand — which is
   visible in the shell history and must be disclosed in any write-up.
3. Decision thresholds are **derived, not chosen**: the E2/E1 claim threshold is the G3 noise
   floor's 95th percentile; the power claim is the §3-E3 detection-rate curve. Both are
   written into the frozen config before unblinding.
4. Statistics: paired cluster bootstrap over pericopes (extend `evaluation/bootstrap.py`
   with a real-valued `paired_statistic_delta` — current API is accuracy-only), BCa
   intervals, 10k resamples; per-token NLLs never pooled across pericopes without clustering.

## 7. Kill criteria (encoded lessons of Phase 3)

- **K1:** G1 or G2 fails after at most **two** pre-declared architecture iterations →
  stop; publish the negative-methods result ("neural model comparison cannot discriminate
  channel structure at synoptic scale") with the gates as evidence.
- **K2:** MDE at N=17 exceeds the effect size the gates demonstrate on known cases → E1 is
  declared underpowered and **never run**; E2 alone proceeds (it has its own gate-based
  power estimate at N=65; if that also fails, K1 logic applies).
- **K3:** contamination audit finds edition-specific memorization that the edition-swap
  ablation cannot bound → likelihood-based claims are withdrawn; only the DiD contrast
  (internally controlled, T7-resistant) may be reported, with the audit attached.
- **K4 (scope):** no experiment beyond E1/E2/E3 and Track A is added to this phase after
  freeze. New ideas go to a Phase-5-follow-up doc, not into this analysis.

## 8. Track A — Q reconstruction (unchanged in intent, now precisely scoped)

Deliverable independent of Track B's verdict: FiD `F` trained on (Mt_p, Lk_p) → Mk_p with
ground-truth evaluation on held-out TT folds (token-level F1 against aligned Mark; lemma-level
and surface-level; pericope-grouped CIs), then applied to the 17 DT pericopes to emit a
**candidate proto-Q text with per-token confidence** (agreement across beam/nucleus samples +
fold ensembles). Framing rule for the write-up: the DT output is "what a Mark-shaped common
source would look like *if* one existed" — Track A makes **no existence claim**; existence is
Track B's question. The TT-validated reconstruction quality number is the headline; the
proto-Q text is the scholarly artifact.

## 9. Implementation plan

### New/changed modules

Legend: ✅ delivered (M0) · ⏳ pending (needs models/GPU).

```
synoptiq/evaluation/
├── scoring.py          ✅          # per-token NLL, per-pericope aggregation, log-mean-exp IS
│                                   #   estimator for the bottleneck branch (unit-tested, no model)
├── bootstrap.py        ✅          # statistic_ci + difference_in_differences (clustered)
├── model_comparison.py ✅          # lift / DiD / MDE / power simulation, GateReport scaffold
├── reconstruction.py   ✅          # Track-A bag-of-tokens F1 vs nearest-witness
├── contamination.py    ✅          # M1 memorization audit (perplexity gap, exact-match)
└── verdict.py          ✅          # M3/M4 decision core: excess-lift, DiD, gate thresholds

synoptiq/models/                    # redactor + FiD live with koineformer.py / encoder.py
├── redactor.py         ✅          # R_Mt / R_Lk / G_Mt / G_Lk: seq2seq LoRA over KoineFormer-NS
├── fid.py              ✅          # Fusion-in-Decoder: encode witnesses separately, fuse in decoder
└── _seq2seq_base.py    ✅          # shared GreTa(+NS)+LoRA loader for both

synoptiq/data/
├── study_design.py     ✅          # full-triple membership, deterministic seeded k-folds,
│                                   #   DT freeze list, overlap partition, census, hashing
├── redaction.py        ✅          # source→target pairs, fusion examples, source-dropout
└── (prepare_data.py)   ⏳ M1(opt)  # ingest Acts (auxiliary, for the Lukan free-composition LM)

synoptiq/utils/constants.py  ✅     # ADDED: MARK_Q_OVERLAP_{CORE,EXTENDED} (frozen, cited)
synoptiq/training/_config.py ✅     # ADDED: frozen StudyConfig (the hashed prereg artifact)

scripts/
├── prepare_study.py    ✅          # M0: `freeze` (census/folds/hashes) + `power` subcommands
├── train_redactors.py  ✅          # train R_*/G_* with TT CV (+ modal/app_fid.py::train_redactors)
├── train_fid.py        ✅          # train FiD, any --witnesses/--target (Track A or E2 model)
├── run_mai_test.py     ✅          # E2 excess-lift + DiD + G3 floor, one fold (writes per-pericope rows)
├── pool_mai.py         ✅          # pool folds 0–4 → CV verdict (+ modal ::mai_cv runs the whole thing)
├── run_gates.py        ⏳ M3       # G1/G2 channel recovery + G4 external → outputs/study/gates/
├── run_channel_test.py ⏳ M5       # E1, single-shot guard (§6.2)
└── reconstruct_q.py    ⏳ M6       # Track A inference (demoted) + confidence annotation

modal/app_fid.py                    # GPU training/eval entry points (mirrors app_dapt.py)
tests/data/test_study_design.py, tests/evaluation/test_{scoring,model_comparison}.py
```

### Milestones (dependency order; est. effort; all GPU on Modal A10G, total < $25)

- **M0 — Census, folds, freeze scaffolding** ✅ **DONE** (no GPU). Delivered:
  `synoptiq/data/study_design.py`, `StudyConfig`, `MARK_Q_OVERLAP_*` in `constants.py`,
  `scripts/prepare_study.py freeze`, `outputs/study/{census,folds,freeze}.json`. Folds
  deterministic under seed 20260707 (5-fold, sizes 12–14); overlap list frozen & hashed;
  33 new tests green. Census corrected the design (N=65 full triples, not 88; wisdom
  unidentifiable). Hashes in §10.
- **M3-power — Power analysis** ✅ **DONE** (partial M3; no GPU).
  `synoptiq/evaluation/model_comparison.py` + `scripts/prepare_study.py power`; MDE curves in §3a.
  Remaining M3 (gates G1–G4, threshold derivation) needs M1/M2 models.
- **M1 — KoineFormer-NS + contamination audit** ◐ **code delivered; GPU run pending.**
  Delivered (no GPU): DAPT decontamination (`exclude_books` on `DAPTConfig`; SBLGNT stem
  filter in the loader — verified on real data, drops the 12 gospel files / ~37% of chunks),
  `--no-synoptics` flag on `train_dapt.py`, Modal entrypoint `start_training_ns`, the audit
  statistics (`synoptiq/evaluation/contamination.py`: perplexity, DiD memorization gap,
  exact-match) + runner `scripts/audit_contamination.py`, 16 new tests. *Remaining (needs GPU,
  ~1 hr A10G):* `modal run modal/app_dapt.py::start_training_ns`, download, then
  `audit_contamination.py --compare-adapters …` → `outputs/study/audit.{md,json}`. *Accept:*
  NS adapters on the volume; memorization gap reported; NS POS probe within 1pt of KoineFormer.
- **M2 — Operators + FiD** ✅ **fold-0 run DONE** (results in §3b; full CV pending).
  Delivered + GPU-run: `synoptiq/models/redactor.py`, `synoptiq/models/fid.py`,
  `_seq2seq_base.py`, `synoptiq/data/redaction.py`, `synoptiq/evaluation/reconstruction.py`,
  `scripts/train_{redactors,fid}.py` (fid parameterised by `--witnesses/--target`),
  `modal/app_fid.py`. **Operators strong PASS** (~1-nat real-vs-mismatch gap). **Track A
  reconstruction: bounded negative** (F1 0.31 vs 0.56 baseline; overfits) → demoted to a
  reported limitation. *Remaining:* pool folds 0–4 for CV error bars (`train_*` per fold).
- **M3 — Gates (GO/NO-GO)** ◐ **decision core + G3 delivered; G1/G2/G4 pending.**
  `synoptiq/evaluation/verdict.py` (`channel_recovery_gate` for G1/G2, `null_threshold` for
  G3, all unit-tested); G3 null floor computed inside `run_mai_test.py`. *Remaining:* G1/G2
  channel-recovery orchestration (needs the `G_*` reconstruction scorers wired), G4 external
  (Josephus←1 Macc), and folding the M0 power result in. Threshold derivation is in place.
- **M4 — E2 MAI test (Lk-target)** ✅ **pooled CV verdict DONE** (§3b, 2026-07-09).
  `synoptiq/evaluation/verdict.py` (`minor_agreement_test`, `did_contrast`, excess-lift) +
  `scripts/run_mai_test.py` (loads a source-dropout FiD on (Mk,Mt)→Lk, computes per-pericope
  excess lift vs a mismatched-Matthew control, the overlap-vs-rest DiD, and the G3 floor) +
  Modal `train_fid_mai`/`run_mai`/`mai_cv`; `scripts/pool_mai.py` pools per-fold rows into the CV
  verdict. **5-fold pooled (N=65): excess lift +0.169 does NOT clear G3 floor 0.194; DiD +0.096
  spans zero → null-after-calibration on both axes** (§4 strict-independence row). *Remaining
  (GPU):* the symmetric **Mt-target** model (needed to distinguish Farrer from MPH and to run the
  T4 symmetry check); edition-swap ablation (T8). *Accept:* all pre-registered statistics reported
  with CIs, incl. nulls — met for the Lk-target axis.
- **M5 — E1 channel test** ⏳ Only if K2 passed. Single shot (`run_channel_test.py`, guard §6.2).
- **M6 — Write-up + (demoted) reconstruction artifact** ⏳ Paper: operators + whichever verdict
  matrix row obtained + Track-A reconstruction reported as a bounded limitation. Compile on Overleaf.

### Baselines required in every table

Copy-source (edit-distance/identity), source-free KoineFormer-NS LM, n-gram channel model
(Lee 2007-style), and — for Track A — nearest-witness (emit Matthew, resp. Luke, verbatim).
Neural numbers reported only as deltas over these.

## 10. Freeze block

Structural hashes below are **frozen now** (M0). The two data-dependent thresholds are filled
at M3 once the gates run — they cannot be computed before the models exist, by design.

```
config_hash (StudyConfig, 5-fold seed 20260707):
    9a9b0d345dc45d129a1c950e7869fde962152c383421b012e78a3878d0ad1227
folds_hash (folds.json):
    c6133c2f4e8820989a64c24f9fb39f2901d608ab921e1f7183470acf3c49747b
overlap_core_hash (MARK_Q_OVERLAP_CORE = 009,012,037,045,057):
    b380601ef5f9cd9044e1bbe15930d976fe2c6de229adc96342deb2947c78c1ae
overlap_extended_hash (core + 038,075,077,145):
    d801af0782576af2829eaa3d7f646c75d6f723ffc900dbab45ae70a807edab66
gates report commit:       <M3: git sha>
E2 claim threshold:        <M3: derived from G3 noise floor>
E1 go/no-go (K2):          <M3: detection rate @ N=17, threshold 0.80>
```

Regenerate and verify with `python scripts/prepare_study.py freeze` (idempotent; the hashes above
must reproduce exactly, or the folds/config/overlap list have drifted from the preregistration).

## 11. Related work anchors

- Lee 2007, *A Computational Model of Text Reuse in Ancient Literary Texts* (ACL) — noisy-
  channel model of Luke's reuse of Mark; our E1 is its neural, hypothesis-comparative successor.
- Abakuks 2006/2012 (JRSS A), *The Synoptic Problem and Statistics* (2015) — HMMs over
  verbal agreements; best-fitting orders place Luke last; nearest peer-reviewed neighbor.
- Goodacre 1998 (fatigue); Hägerland 2019 (NTS 65) — the operator-transfer objection, here a
  sensitivity axis (T6); Garrow (MPH), included symmetrically.
- Neirynck 1974 — minor-agreement catalog (E2 localization).
- Izacard & Grave 2021 — Fusion-in-Decoder.
- Sominsky & Wintner 2019; Rabinovich & Wintner — translationese direction: in-domain
  supervised success, cross-domain collapse; the external confirmation of Phase 3's closure.

## 12. What this phase will never claim

Per-pericope direction verdicts; anything about hypotheses outside §1's live space; Q's
existence from Track A's outputs alone; any conclusion whose gate failed or whose kill
criterion fired. A null on every axis is a publishable, pre-interpreted outcome — that is
the point of the design.
