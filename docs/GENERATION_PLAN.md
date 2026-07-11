# Generation Plan — Koine-T5-Hexapla (the MAX edition)

**Preregistration.** Goal: make the model markedly more coherent, faithful, and expressive in
free generation **while holding POS / lemma / synoptic competence at or above the published
Koine-T5 levels** — a no-regression constraint enforced by a gate, not hoped for. Named after
Origen's *Hexapla*, the six-column parallel-scripture alignment: the ancient precedent for
SynoptiQ's parallel-text work.

Artifacts: `modal/app_koine_hexapla.py` (trainer), `scripts/prepare_koine_maxi_corpus.py` +
`synoptiq/data/koine_corpus.py` (corpus), this file (prereg). Sibling model line to Koine-T5
(`modal/app_koine_t5.py`), which stays as-is.

## Why (the diagnosis)

`docs/gospel_of_the_savior.md` — a long free-generation sample from the published Koine-T5 — is
the evidence. Its **surface form is excellent** (morphology, case agreement, Lukan register) but
its **discourse control is absent**: speakers swap (the Magnificat given to "he"; the thief's
line to Peter), pericopes bleed ("Mary your wife" fuses Annunciation-to-Mary with
angel-to-Joseph), theology garbles, and it mode-collapses onto high-frequency Lukan openers.
Every failure is semantic/referential, not grammatical.

Root cause (in `app_koine_t5.py`): the generative signal is starved and teaches the wrong thing.
`TASK_WEIGHTS = {pos:3, lemma:1, synoptic:1, denoise:3}` → half of every batch is tagging; the
only free-generation signals are span-infill **denoise** (mean span 3 — teaches infilling, not
fluency) and a **155-pair** synoptic pool. The denoise corpus is only Gospel (~49K tok) + PROIEL
(~214K tok); `MAX_SEQ_LEN=256` caps passage-level coherence.

**Orthogonality to the closed negative result.** This is NOT about copying direction. Direction
detection is unidentifiable because it recovers a *latent* (`docs/DIRECTION_NEGATIVE_RESULT.md`,
which stands). Generation quality is bounded only by capacity + data + objective — all
improvable. Improving generation neither reopens nor depends on the direction question.

## The five levers

1. **Diet** — `scripts/prepare_koine_maxi_corpus.py` builds ~16.8M words of coherent Koine/
   Classical prose (LXX 622K + First1KGreek 16M + Apostolic 335K + SBLGNT-minus-synoptics 204K)
   → 94,914 passage windows. The LXX (623,693 words) was previously "0 chunks": the on-disk copy
   is the *eliranwong Text-Fabric* edition, not the *biblicalhumanities* plaintext one `_parse_lxx`
   expects. `synoptiq/data/koine_corpus.py` adds a dependency-free TF reader (validated: Gen 1:1 =
   "ἐν ἀρχῇ ἐποίησεν ὁ θεὸς …").
2. **Objective** — a new **continuation (prefix-LM)** task (`continue: <prefix>` → rest) teaches
   autoregressive fluency the denoise-only diet never provided.
3. **Context** — `MAX_SEQ_LEN` 256 → **512** so a whole pericope/window fits in one sequence
   (cross-sentence state tracking). Micro-batch 2 × grad-accum 16 keeps the effective batch at 32.
4. **Capacity** — LoRA **r=128 α=256** (was 64/128; 54.3M trainable, 18%) so the added tasks stop
   competing with POS for the same directions.
5. **Curriculum + gate** — two-stage schedule + regression-gated selection (below).

## Data → task pools

Pools (`build_task_pools`): `pos`, `lemma`, `synoptic` (Gospel + PROIEL, unchanged), plus the
generative `denoise` and `continuation`, each **register-split** `{koine, classical}`.

Decontamination: the synoptic gospels are held out of SBLGNT, and every window is screened by
8-gram shingle overlap against the held-out Gospel **test + val** splits (`prepare_*` +
`dedup_passages`). 59,397 dup/contaminated windows were dropped; 94,914 kept.

Register note: Koine raw prose (~1.16M words → ~7K windows) is far smaller than First1KGreek
Classical (16M → 86K). Sampling is therefore **register-first at 70/30 Koine/Classical**
(`REGISTER_WEIGHTS`, matching DAPT's replay ratio) so Koine style is never swamped — Classical is
fluency *replay*, not the target register. Koine generative signal is ~4× the old diet (1.16M vs
263K tok) and remains the bottleneck (see Limitations).

## Two-stage curriculum

One run, weights switch at `STAGE_A_FRAC = 0.4` of `MAX_STEPS`:

| Stage | steps | denoise | continuation | pos | lemma | synoptic | intent |
|-------|-------|---------|--------------|-----|-------|----------|--------|
| A | 0 – 40% | 4 | 4 | 1 | 0.5 | 0.5 | build generative backbone (analysis *rehearsed*, not dropped) |
| B | 40 – 100% | 2 | 2 | 3 | 1 | 1.5 | rebalance for all-task competence |

Even the least-weighted task appears several times per optimizer step (32 draws), well above the
frequency at which a task is catastrophically forgotten — rehearsal is the anti-forgetting
mechanism.

## Regression-gated evaluation (`evaluate_all`, every 1000 steps)

Metrics on held-out data:
- **POS** tok-acc + EM on PROIEL dev, NT / Classical (unchanged from Koine-T5).
- **Lemma** tok-acc on PROIEL dev (new; same batched decoder).
- **Perplexity** — teacher-forced NLL of the gold continuation (fluency).
- **Token-F1** — free-form continuation vs gold (right-direction lexical overlap).
- **Morphological self-consistency** — run the model's OWN `pos:` on its own generation; the
  fraction of generated words that receive a valid, length-aligned MorphGNT tag. Degenerate or
  garbled Greek fails to self-tag → low score. Cheap, self-contained, defensible.

**Gates (no-sacrifice):** `GATE_POS_NT = 0.966`, `GATE_POS_CL = 0.877`, `GATE_LEMMA = 0.80`
(published Koine-T5 dev numbers; lemma conservative, raise to the measured baseline once known).

**Best-checkpoint selection key** = `(1, f1 + morph)` once all gates pass, else `(0, pos_tok)`.
Any gate-passing checkpoint outranks any non-passing one; among passers the **generation score is
maximized** — i.e. *maximize generation subject to no analysis regression*. Before the gates are
met, POS token-acc still tracks early progress so a best/ always exists.

## Success / stop criteria

- **Primary:** at some gate-passing checkpoint, continuation token-F1 and morphological
  self-consistency both exceed the Koine-T5 baseline (measured under identical eval), with POS-NT
  ≥ 0.966 and lemma ≥ baseline held. Qualitatively: fewer speaker/pericope errors on the
  `gospel_of_the_savior.md` prompts (A/B via `demo`).
- **Stop:** if no checkpoint passes the gates by end of training, the capacity/diet is
  insufficient at 220M — report as a finding and revisit the larger-base track (out of scope).
- **Null-but-honest:** gains in fluency that cannot clear the gates are *not* claimed as success;
  the gate is the definition of "no sacrifice".

## Reproduction

```bash
# 1. Build the corpus artifact (local; data/raw already on disk)
python scripts/prepare_koine_maxi_corpus.py            # → data/processed/koine_maxi/
python scripts/prepare_koine_maxi_corpus.py --validate # summarize
python scripts/prepare_koine_maxi_corpus.py --upload   # → synoptiq-data:/koine_maxi

# 2. Train (Modal; auto-resumes, falls back to Gospel+PROIEL if the artifact is absent)
modal run modal/app_koine_hexapla.py::train
modal app logs koine-t5-hexapla

# 3. A/B demo vs GreTa base on the failure-mode prompts
modal volume get koine-hexapla-outputs koine_hexapla/best models/koine_hexapla/best
python modal/app_koine_hexapla.py demo models/koine_hexapla/best
```

## Limitations / honest bounds

- **220M base.** GreTa is small; very-long-range coherence and world-knowledge have a real ceiling
  and there is no larger Ancient-Greek-native generative base. The diagnosis says the current
  failures are diet/objective/context-length, not parameter count — so gains should come before the
  ceiling binds; a larger base is the explicitly-deferred next track.
- **Koine generative data is the bottleneck** (~1.16M words). First1KGreek adds Classical fluency
  breadth but not Koine style; the 70/30 weighting mitigates but cannot manufacture Koine data.
  Diorisis (10M) + documentary papyri are the deferred next ingestion.
- **`GATE_LEMMA` is a placeholder** (0.80) until the exact Koine-T5 lemma dev-acc is measured; set
  it to that number before trusting the no-regression claim on lemma.
- **Contrastive-search generation** fetches a `custom_generate` repo on transformers ≥4.62
  (`trust_remote_code=True`); pin/mirror it for fully offline reproduction.
