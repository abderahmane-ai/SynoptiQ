# Paper plan — "Who Copied Whom? Why the Texts Can't Tell Us — and What They Can"

**Status: PLAN (not drafted). This is the *honest* paper. It supersedes an earlier outline
(from a non-specialist collaborator) that proposed conditional-NLL / perplexity-ratio
*direction detection* as a novel framework — that method is this project's CLOSED NEGATIVE
RESULT (`docs/DIRECTION_NEGATIVE_RESULT.md`): invalidated, not "unvalidated." Do not resurrect
it. See §0.**

- **Working title:** *Who Copied Whom? Why the Texts Can't Tell Us — and What They Can*
- **Subtitle:** *An information-theoretic limit, and a hypothesis-neutral framework, for
  computational Synoptic source criticism*
- **Venue (primary):** Computational Humanities Research (CHR) — negative-result + resources +
  a preregistered null is squarely in scope. **Alternatives:** LREC (resource framing), an
  ACL/EMNLP Findings or DH-track submission.
- **Author:** Abderahmane Ainouche. (A cousin offered to hand-submit; that raises the bar — a
  falsified method delivered in person is a credibility risk. This plan removes that risk.)

---

## 0. Why this paper, not the other one (read first)

The rejected outline's centerpiece was: *"if NLL(B|A) < NLL(A|B) then A is the source."* Three
fatal problems, all already established by this project:

1. **It's the closed negative result.** Conditional-NLL / MDL asymmetry measures the
   **information gradient** (length + dominant/Markan style), which is *statistically
   independent of historical direction* (copying both compresses and expands). Verified:
   every length/NLL feature **flips sign** across known-direction pairs (Jude→2 Peter copies
   *longer*; LXX Chronicles copies *shorter*).
2. **It contradicts its own impossibility section.** Proving symmetric metrics are
   direction-blind and then proposing an asymmetric metric that is blind *for the same reason*
   (isomorphic to distinguishing a lossy projection from its inverse without a prior over the
   author's injection operator).
3. **"Novel/unvalidated" is inaccurate — it is *invalidated*.** We ran it; it failed. Omitting
   that is a material omission.

**The honest paper keeps the good bones (impossibility framing, corpus, model) and flips the
"direction" section from *proposed solution* → *definitive negative result*, then adds the
reframe that actually sidesteps direction.** A clean negative result + released resources +
a preregistered null is a *stronger*, more defensible paper than an overclaim that any
reviewer who knows the translationese-direction literature will desk-reject.

---

## 1. Thesis and contributions

**Thesis:** The field has spent decades trying to read the *direction* of copying off the
texts. That is not achievable from parallel text alone. The productive move is to stop asking
"which text is the source?" and instead **compare generative hypotheses using supervision every
hypothesis already accepts.**

- **C1 (negative, theoretical + empirical):** Direction of literary copying is not recoverable
  from parallel texts alone. Symmetric metrics trivially; the conditional-NLL "escape"
  *because* it tracks the direction-independent information gradient. Framed as a rigorous
  **argument + empirical demonstration**, NOT a formal theorem (do not overclaim — see §5).
- **C2 (positive, reframe):** Source-hypothesis comparison (2SH vs Farrer/MPH) *without* solving
  direction — compare generative models of the double tradition trained only on
  hypothesis-neutral triple-tradition supervision (Mk→Mt, Mk→Lk). Report the **preregistered,
  calibrated null** honestly.
- **R1 (resource):** the **SynoptiQ corpus** — token-aligned, morphologically annotated
  Synoptic Gospels (49,061 tokens · 170 pericopes · 235 alignments).
- **R2 (resource):** **Koine-T5** — a general-purpose multitask Koine/Ancient-Greek seq2seq
  model (POS / lemma / denoise / synoptic style transfer). Trained and **published**
  (`huggingface.co/ainouche-abderahmane/koine-t5`, CC BY-NC-SA 4.0): **96.6% NT / 91.7% pooled
  POS token accuracy** on PROIEL dev.

---

## 2. Section-by-section outline

**Abstract (~250 w).** Synoptic Problem → prior computational work establishes *relatedness*,
not *direction* → we show (argument + sign-flip experiments) that direction is unrecoverable
from text, including the conditional-NLL approach the field keeps reaching for → we release
SynoptiQ + Koine-T5 → and demonstrate the hypothesis-neutral reframe, whose preregistered test
returns a calibrated null (minor agreements = text-critical noise). Contribution: a limit
result, two resources, and a methodology that sidesteps the limit.

**§1 Introduction.** The Synoptic Problem; near-consensus Markan priority; the live hypothesis
space (2SH / Farrer / MPH; Griesbach & Augustinian excluded by agreement-topology, see §3);
why word-overlap/stylometry only prove relatedness. Thesis + contribution list.

**§2 The limit: direction is not recoverable from parallel text.** (Theory core.)
- Symmetric metrics (cosine, Jaccard, Needleman–Wunsch) are trivially blind: `Sim(A,B)=Sim(B,A)`.
- The asymmetric escape fails: the only computable asymmetry is the information gradient
  (complex⇄simple), which is independent of historical direction because scribes both simplify
  and expand. Recovering direction ≈ distinguishing a lossy projection from its inverse without
  a prior over each author's injection operator — a prior that needs a large *independent*
  corpus of that author's own prose, which does not exist (Mark ≈ 11k words; special material
  tiny). **Frame as argument, not theorem.**

**§3 Empirical confirmation (what makes it a paper).** We built the machinery and it fails
every way:
- Table of 6 attempts (from `docs/DIRECTION_NEGATIVE_RESULT.md`; detail recoverable from git):
  similarity-geometry (chance), conditional-NLL/MDL (measures Markan style+length),
  learned-MDL-on-synthetic (generator artifact), RPM connective canon (a Markan-style
  detector; *backwards* on external pairs), editorial-fatigue "RootModel" (real on narrative,
  genre-limited, fails on gospels), agreement-topology (recovers *topology*, not direction).
- **The killer demo:** sign-flip on known-direction pairs (Jude→2 Peter copy-*longer*; LXX
  Chronicles copy-*shorter*) — every feature reverses, proving it tracks the gradient not
  direction.
- Cross-domain collapse (83–100% in-domain → ~60% cross-domain), matching the translationese
  direction literature (Sominsky & Wintner 2019; arXiv:1609.03205) — external corroboration.

**§4 The SynoptiQ corpus (R1).** Construction (SBLGNT + MorphGNT; Needleman–Wunsch token
alignment on the (lemma, POS) key; Aland pericope table); layers (surface / MorphGNT tag /
lemma); census (65 *full* triples of 88; DT N=17; `wisdom` stratum has no triple analog →
unidentifiable a priori). Released CC-BY-SA.

**§5 Koine-T5 (R2).** GreTa (T5-base) + LoRA (r=64, 27.1M params, 104 MB) multitask: denoise
(online T5 span corruption) / POS / lemma / synoptic style transfer, on the Gospel corpus **+
UD_Ancient_Greek-PROIEL** (~214K tokens, ~52–59% NT Koine + Herodotus). Contributions worth
naming: the **XPOS→MorphGNT** cross-scheme mapping (article `S-`→`RA`, *not* UPOS/FEATS whose
`PronType=Dem` is a trap); **balanced multitask sampling** (protects the 155-example synoptic
pool from catastrophic forgetting); the tokenizer **sentinel-slot** trick (no embedding
resize). Eval: POS accuracy + Exact-Match on PROIEL dev, **reported NT vs Classical**. Headline
(best checkpoint, step 28000): **POS token acc 96.6% NT / 87.7% Classical / 91.7% pooled**
(EM 85.2 / 52.0 / 68.6). Published: `huggingface.co/ainouche-abderahmane/koine-t5`.

**§6 What you CAN do: hypothesis comparison without direction (C2).**
- Insight: the triple tradition is **hypothesis-neutral, direction-labeled** supervision
  (Mk→Mt, Mk→Lk — every live hypothesis accepts it). Train redaction operators on it; then
  compare *generative models* of the double tradition (2SH latent-source bottleneck vs Farrer
  direct) — never per-pair direction.
- The preregistered **E2 minor-agreement test**: paired excess-lift `NLL(Lk|Mk,control) −
  NLL(Lk|Mk,Mt)`, overlap-vs-rest difference-in-differences, and a G3 null floor.
- **Result (honest, headline of C2):** null-after-calibration. Pooled 5-fold CV (N=65): excess
  lift **+0.169** [CI 0.127, 0.213], P(>0)=1.00 — *but does NOT clear the G3 floor 0.194*; DiD
  overlap(5) vs rest(60) **+0.096** [CI −0.049, +0.296] *spans zero*. The prereg floor + DiD
  killed what a naive CI-excludes-0 test would have published as Farrer support. Reading:
  strict independence beyond Mark; minor agreements behave as text-critical noise.
- Supporting: operators are validated (**strong PASS**, ~1-nat real-vs-mismatched-source gap);
  contamination is negligible (memorization gap 0.016 ≪ 0.25 threshold; 0% verse-completion
  exact match — LoRA can't memorize verbatim).

**§7 Discussion.** For the field: computational source criticism must abandon
direction-from-text; the productive path is neutral-supervision generative comparison, reported
with preregistered floors. Limitations: Track A Q-reconstruction is a *bounded negative* (FiD
Mt+Lk→Mark F1 0.31 < nearest-witness 0.56 — witnesses too close to Mark for abstractive fusion
at this scale); DT is power-bottlenecked (N=17).

**§8 Conclusion.** A limit result + two resources + a methodology that respects the limit.

**Future work.** Symmetric Mt-target E2 (Farrer vs MPH + T4 symmetry check) and edition-swap
ablation (T8) to turn the null "robust"; scaling Koine-T5 + the framework to LXX / wider
Classical copyist traditions.

---

## 3. Evidence inventory (what backs each claim, and where it lives)

| Claim | Evidence | Location |
|---|---|---|
| Symmetric metrics blind | trivial identity `Sim(A,B)=Sim(B,A)` | §2, standard |
| NLL escape fails | info-gradient ⟂ direction; sign-flip on known pairs | `docs/DIRECTION_NEGATIVE_RESULT.md`; git history (removed `synoptiq/direction/`) |
| 6-approach failure table | the closed investigation | `docs/DIRECTION_NEGATIVE_RESULT.md` |
| Markan priority (topology) | hub structure; ~0.22 singular-reading rate excludes conflation | same doc |
| SynoptiQ corpus stats | 49,061 tok / 170 peric / 235 align | corpus + Paper A `paper/main.tex` |
| Koine-T5 design | code + this session | `modal/app_koine_t5.py`, `CLAUDE.md` |
| E2 null (pooled CV) | lift +0.169 < floor 0.194; DiD +0.096 spans 0 | `docs/SOURCE_CRITICISM_STUDY.md` §3b; `outputs/study/mai/mai_pooled.json` |
| Operators strong PASS | ~1-nat real-vs-mismatch gap | `docs/SOURCE_CRITICISM_STUDY.md` §3b |
| Contamination negligible | gap 0.016; 0% exact-match | `docs/SOURCE_CRITICISM_STUDY.md` §3b; `outputs/study/audit.md` |

---

## 4. Status — DONE vs NEEDED

- **§2 argument + §3 empirical:** evidence DONE (investigation closed). NEEDS: write-up;
  recover experimental detail from git (`git log`, the removed `synoptiq/direction/` tree).
- **§4 SynoptiQ:** DONE (already documented in Paper A — reuse/cite).
- **§5 Koine-T5:** **DONE** — 30k-step run finished (96.6% NT / 91.7% pooled POS token acc),
  best checkpoint published to HuggingFace (CC BY-NC-SA 4.0). Nothing pending.
- **§6 E2 null:** the Lk-target pooled CV is DONE. NEEDS (to report as "null *and robust*"):
  the symmetric **Mt-target** E2 model + the **edition-swap** ablation (both are small GPU
  runs, <$25 total per `SOURCE_CRITICISM_STUDY.md` M4).
- **Overall:** the paper is writable *now* as (C1 negative + R1 corpus + C2 null); Koine-T5
  (R2) numbers and the robustness runs strengthen it but are not blockers for a first draft.

---

## 5. Honest framing rules (what we will NOT claim)

- **No direction verdicts.** We never claim to determine who copied whom from the texts.
- **"Argument + demonstration," not "theorem."** §2 is an information-theoretic argument
  backed by experiments, not a formal proof. Do not write "we mathematically prove."
- **NLL ratios are presented as a *failed* method**, with the sign-flip evidence — never as a
  working or "promising" framework.
- **The E2 null is reported as a null** (lift < floor; DiD spans 0). No spin toward Farrer.
- **Title signals the limit** ("Why the Texts Can't Tell Us").
- **No claims about Q's existence** from Track A outputs; Track A reconstruction is a reported
  limitation, not a headline.

---

## 6. Decisions (resolved 2026-07-10)

1. **Scope:** ✓ DECIDED — draft the full paper now (C1 negative + R1 corpus + R2 Koine-T5 +
   C2 null), and launch the Mt-target/edition-swap robustness runs in parallel; slot the
   "null *and* robust" numbers in as they land.
2. **Venue:** deferred — not a blocker; drafting to the KoineFormer manuscript's length/format.
3. **Relationship to Paper A (KoineFormer):** cite it as prior work (they're separable).
4. **Visual style / compile:** reuse the `paper/main.tex` XeLaTeX styling verbatim
   (marine/terracotta/papyrus palette, Poppins/Pagella/FreeSerif fonts). Compile on Overleaf
   (local TeX lacks `pdfcol.sty` — see memory `paper-compile-on-overleaf`).
