# Direction Scorer (Phase 3): Findings

A rigorous, validation-gated investigation of whether copying **direction** between
parallel Koine passages can be detected.

**Three-line summary.** *Global* passage scores (similarity, NLL/compression, a
synthetic-trained head) are all confounded with length or Markan style — that boundary
result stands (below). Reframing direction as the stemmatology **rooting problem** and scoring
at the **variant** level with the textual-criticism canons finally works: the **Redactional
Polarization Model (RPM)** reaches **0.78 directed accuracy on the synoptics (0.876 on the
pericopes where it has evidence; 0.97 on its confident quartile via abstention)** using a
single length-free canon (connective smoothing), and correctly *rejects* the length confound.
Pooled over the corpus it **recovers Markan priority with zero synoptic supervision** (posterior
2SH 0.64 / Farrer 0.36 / Griesbach ≈ 0 / Augustinian ≈ 0); the Farrer-vs-Q question is left
open (the Q material carries too few connective edits to settle it). Adding editorial fatigue
as a second canon does **not** help on the synoptics (H4, genre-limited). See "Breakthrough"
below; the negative results that forced this design follow it.

## Breakthrough: the Redactional Polarization Model (RPM)

Online research reframed the task precisely: this is the stemmatology **rooting problem**
(assign direction to an otherwise-clear tree), solved classically by **variant
polarization** — deciding which of two readings is primitive via the 250-year-old canons
(*lectio difficilior*, *lectio brevior*, anti-harmonization). Global scores failed because
they never polarized individual variants. RPM aligns a pair, extracts typed **variants**,
scores each with signed, *local*, confound-controlled canon features (positive => X is the
source), and **aggregates** them into a per-pair score with **abstention**.

Gated hypothesis tests (all with block-grouped bootstrap CIs; `scripts/analyze_polarization.py`,
`scripts/train_polarization.py`):

- **H1 — which canons polarize?** *lectio brevior* is a **pure length proxy** (0.00 on
  copy-shorter, 1.00 on copy-longer — caught red-handed) and is dropped. *lectio difficilior*
  (harder-reading, frequency markedness) is at **chance**. **Connective smoothing** (καί→δέ
  is a directional edit) is the survivor: consistent on both external polarities and
  **0.80 on the synoptics**.
- **H2/H3 — aggregation + abstention.** Using only the length-free connective canon,
  scaled so magnitude can't sneak length back in:

  | eval set | directed acc | @50% cov | @25% cov |
  |---|---|---|---|
  | **synoptic** (Mark→Matt/Luke) | **0.78 [0.71, 0.85]** | 0.93 | **0.97** |
  | Jude→2 Peter (n=6) | 1.00 | 1.00 | 1.00 |
  | LXX Chronicles (in-domain) | 0.57 | — | 0.64 |

  Dropping the length feature *kept* the synoptic result (0.76→0.78), proving it is
  connective smoothing, not length. Abstention works: the confident 25% of pericopes are
  97% correctly directed.

**What it is and isn't.** RPM is the first component to beat chance on **real** synoptic
direction with an **interpretable, length-controlled, abstention-calibrated** signal, and
it directly supports Markan priority on its confident pericopes. Honest limits: connective
smoothing is a **gospel-genre** phenomenon (weak in LXX historical narrative), so
cross-corpus transfer is partial; and while the signal is directional *per edit* (not a
global καί count) and shows up on non-synoptic Jude→2 Peter, its synoptic strength partly
coincides with Mark's καί-heavy style.

### H4 (R4): does editorial fatigue add a second canon? — No, on the synoptics.

Folding the one length-robust fatigue feature (`intro_lateness`, shared entities introduced
later in the copy) into the RPM as a second feature was tested three ways
(`scripts/train_polarization.py`, models `canon` / `canon+fatigue` / `fatigue_only`):

| model | synoptic dir. acc | @25% cov | LXX in-sample | learned weights |
|---|---|---|---|---|
| canon (connective) | **0.78 [0.71,0.85]** | **0.97** | 0.57 | connective=0.078 |
| canon+fatigue | 0.41 [0.33,0.49] | 0.37 | 0.63 | fatigue=0.45, connective=−0.007 |
| fatigue_only | 0.44 [0.36,0.53] | 0.40 | 0.63 | fatigue=0.45 |

The naive blend *collapses* the synoptic result. The mechanism is diagnostic, not noise:
`intro_lateness` is the **stronger in-sample canon on LXX narrative** (0.57→0.63) but is at
**chance on the synoptics** (`fatigue_only` 0.44, CI includes 0.5). A linear blend fit on the
LXX training corpus therefore over-weights fatigue and suppresses the connective weight, then
carries that LXX weighting to a genre where fatigue is noise.

**The decisive fairness check (H4b).** A second canon can only help where the first is
*silent*. Partitioning the 153 synoptic pairs by whether any connective variant fires:

- connective **fires** (n=121): connective directed acc **0.876**; fatigue there 0.45 (chance).
- connective **silent** (n=32): fatigue 0.44 (chance) — **no complementary coverage**.

So fatigue adds nothing on the synoptics under *any* combination scheme, because it is at
chance both where the connective canon fires and where it is silent. This matches the earlier
prototype note: the crude aggregate proxy misses Goodacre's *sparse, specific* dangling
references. Fatigue is a real-but-**genre-limited** canon (it carries weak signal on LXX
abbreviation and on Jude), *complementary in genre* to connective smoothing rather than
additive on the gospels.

**Corollary (a real refinement).** RPM's headline 0.78 is dragged down by the 32 evidence-free
pericopes it is forced to guess on; on the **121 pericopes where it actually has a connective
variant, directed accuracy is 0.876**. The existing abstention curve already realizes this
(silent pericopes score 0 and are dropped first → 0.93@50%, 0.97@25%). The right RPM posture
is therefore connective-only + abstention on the synoptics; making fatigue useful *there*
would need a sharp per-pericope dangling-reference operator, not the aggregate.

### H5 (R5): rooting the tree — Markan priority recovered; Farrer vs Q left open.

The four hypotheses are four *rootings* of the one Matthew–Mark–Luke relationship. Pooling the
unsupervised connective-canon vote over every synoptic pericope into a Beta-Bernoulli marginal
likelihood per pairwise relationship (`synoptiq/bayesian/rooting.py`, `scripts/root_stemmata.py`)
gives a posterior over the four stemmata. The 2SH "independent" prediction on Matthew–Luke is
modelled explicitly as θ=0.5 (no consistent direction), so it competes on equal footing.

Per-relationship votes (k = pericopes voting the *first* book is the source, of n non-silent):

| relationship | k / n | source vote | reads as |
|---|---|---|---|
| Matthew–Mark (triple) | 15 / 59 | Mark 75% | **Markan priority** |
| Mark–Luke (triple) | 52 / 57 | Mark 91% | **Markan priority (decisive)** |
| Matthew–Luke (triple) | 36 / 48 | Matthew 75% | *confounded* (both used Mark here) |
| Matthew–Luke (double / Q) | 7 / 12 | Matthew 58% | **near chance — no consistent direction** |

Stemma posterior (Mt–Mk & Mk–Lk triple + Mt–Lk double, uniform prior):

| hypothesis | posterior |
|---|---|
| **2SH** | **0.64** |
| **Farrer** | **0.36** |
| Griesbach | ~0.00 |
| Augustinian | ~0.00 |

**Two clean conclusions.** (1) **Markan priority is recovered by an unsupervised canon.**
Griesbach and Augustinian are annihilated because both require Mark to be a *copy* (Mt→Mk),
yet the canon sees Mark as the *source* on both Markan relationships (75% and 91%). This is the
field-consensus result, reproduced with zero synoptic supervision. (2) **Farrer vs Q is left
open, leaning weakly to Q.** 2SH and Farrer agree on Markan priority and differ *only* on
Matthew–Luke; on the double tradition (the Q material) the canon finds **no consistent Mt→Lk
direction** (7/12 ≈ chance), which is what independence/Q predicts → **Bayes factor
Farrer:2SH = 0.56** (anecdotal, favouring Q). The striking triple-tradition Mt–Lk signal
(BF≈165) is **not** valid Farrer evidence: there both evangelists used Mark, so a Mt-source
connective polarity is expected under *every* hypothesis and says nothing about direct Mt→Lk
dependence. Using the double tradition is essential, and there the signal is simply too sparse
(only 12 pericopes carry a connective edit) to settle the question.

**Honest caveats.** The Markan-priority verdict partly coincides with Mark's καί-heavy style:
the canon judges the καί-richer text as primitive, and Mark is καί-richer. It is nonetheless
*per-edit* directional (καί→δέ substitutions, not a global count) and validated on non-synoptic
Jude→2 Peter; and crucially the *exclusion* of Griesbach/Augustinian is robust because those
require Mark to have smoothed away its own καί, an edit the canon does not see anywhere. The
Farrer/Q result is underpowered and should be read as "the connective canon alone cannot settle
it — more Q-material data or a sharper canon is needed," not as a refutation of Farrer.

---

## The negative results that forced this design

Short version: at the granularity of *whole-passage global scores*, there is no direction
signal that is simultaneously author-independent, length-independent, and transferable.
Every such signal is a confound (length or Markan style) or a synthetic-generator artifact.

## The question and the built-in trap

Under the Two-Source Hypothesis the labelled data is confounded by construction: in the
triple tradition **Mark is always the source**, and Matthew↔Luke are always labelled
*independent*. So "direction" is collinear with "is this Mark" — any probe can score by
detecting Markan style instead of direction. The goal was a component that detects
direction *per se*.

## Methods tried, and what each turned out to measure

| Probe / component | Where it looked | Result | What it actually measured |
|---|---|---|---|
| 10 similarity-geometry features (+ logistic reg.) | frozen-encoder cross-similarity | ~chance (0.33–0.36) | nothing usable |
| Conditional-NLL asymmetry `NLL(B\|A)−NLL(A\|B)` | T5 decoder codelengths | synoptic partial-r 0.41–0.53 | **Markan style + length** (see below) |
| Full MDL score (with marginals) | codelengths | synoptic strong, external ~0 | length/typicality |
| Learned MDL head on synthetic corpus | 11 codelength features | 99% synthetic, fails real | **generator artifact + length prior** |

## The three decisive controls

1. **Zero-shot vs DAPT ablation.** The synoptic conditional-NLL signal is just as strong
   with zero-shot GreTa as with DAPT — it is intrinsic to Mark's (simpler, more
   predictable) Greek, i.e. *style*, not something learned about copying.

2. **Length partial-correlation.** On the synoptics the NLL signal survives a length
   control (partial r ≈ 0.5); on Jude→2 Peter it collapses to ~0. Present only where a
   synoptic author is present → style, not direction.

3. **Both length polarities (the clincher).** Two real known-direction sets were built:
   - **Jude → 2 Peter** (copy is *longer*, expansion): deterministic NLL sign says
     source-in-A ⇒ **negative**; learned component 17% (backwards).
   - **LXX Samuel-Kings → Chronicles** (copy is *shorter*, compression): the NLL sign
     **flips** to positive; learned component 100%.
   The sign and the accuracy track *which text is longer*, not which is the source. A
   genuine detector would work on both; a length prior does the opposite on the two.

## The learned component's final evaluation ladder

Trained only on a synthetic same-author, length-decorrelated redaction corpus
(so it cannot use authorship or length), evaluated frozen:

| Eval set | Directed acc | 95% CI | Reading |
|---|---|---|---|
| synthetic val (held-out authors) | 0.98 | [0.94, 1.00] | learns the *generator's* edit |
| synthetic test (held-out authors) | 0.99 | [0.98, 1.00] | generalizes across authors… |
| external Jude→2 Peter (copy longer) | 0.17 | [0.00, 0.50] | …but backwards on real expansion |
| external LXX Chronicles (copy shorter) | 1.00 | [1.00, 1.00] | right only because copy compresses |
| synoptic test (confounded) | 0.36 | [0.18, 0.53] | chance |

The 99% synthetic is a real, generalizable, length-independent signal — but it is the
*generator's* signature (redactional smoothing toward frequent vocabulary), which real
copyists do not apply consistently. On real data the component collapses to a length
prior ("shorter = copy").

## Conclusion

- No tested method extracts transferable copying direction from short parallel passages.
- The apparent signals are confounds (length, Markan style) or generator artifacts,
  each exposed by a specific control.
- **The bottleneck is the data and the granularity, not the model.** Direction is
  collinear with author in the only labelled corpus (synoptics), and the compression/
  typicality cues that do carry weak directional information are not stable across real
  corpora.

## Implications for the project

- **Paper B** should report this as a principled negative/boundary result with the
  ladder above — it is more valuable (and more honest) than a "direction detector" that
  is really a Mark-or-length detector, and it directly engages the methodological
  standards the field (Goodacre et al.) would demand.
- **Phase 6 (Bayesian model comparison)** must NOT feed a direction classifier trained on
  2SH labels into the hypothesis test — that is circular. Any direction evidence used
  there has to be unsupervised on the synoptics and validated on external known-direction
  data.
- **What could still work (longer-range signal):** editorial fatigue (Goodacre 1998) is
  directional by nature and operates on *within-pericope inconsistency*, not global style
  or length — it is the one avenue not yet confounded here, but it needs full-pericope
  aligned data and careful semantic modelling, and can only be validated where direction
  is already known.

## Update: editorial-fatigue prototype (the first positive lead)

A follow-up prototype (`synoptiq/evaluation/fatigue.py`, `scripts/legacy/analyze_fatigue.py`)
tests *entity/reference-level* directional features instead of global scores — the key
being that they are **position-normalized within a pair**, so global length and style
are constants and cannot confound them. The features are signed and antisymmetric under
an A↔B swap; each is scored against the both-length-polarity criterion.

Result on real known-direction data (directed accuracy, block-grouped bootstrap CI):

| feature | Jude→2Pet (copy longer) | LXX Chronicles (mixed polarity, 30 pairs) | synoptic |
|---|---|---|---|
| `coverage_asym` | 0.50 | **0.50** (collapsed from 0.79 once length was decorrelated) | 0.57 |
| `intro_lateness_asym` | 0.67 [0.33,1.0] | **0.67 [0.50,0.82]** (did NOT collapse) | 0.45 |

Two things matter here. First, the harness *works*: `coverage_asym` was length in
disguise and died the moment the LXX set was made length-balanced — exactly the failure
mode that fooled every earlier global score. Second, `intro_lateness_asym` (shared
entities are introduced *later* in the copy — an abbreviator drops early introductions)
is the **first feature in the whole investigation that survives length-decorrelation and
points the same way on both external sets** (~0.67). It is real but weak and underpowered
(CIs touch chance), and it is at chance on the synoptics — where the crude aggregate proxy
misses Goodacre's *sparse, specific* inconsistencies.

**The lead, concretely:** direction detection needs (a) reference/edit-level features, not
global scores — confirmed, they don't length-collapse; (b) far more real known-direction
data with rich shared content (the LXX Chronicles synopsis is ~80 sections vs the 30 used;
Josephus paraphrases the LXX at length; patristic NT quotation; classical epitomes); and
(c) a sharper per-pericope fatigue operator that flags specific dangling-reference events
and *abstains* elsewhere, rather than averaging. Several weak-but-real entity-level signals,
combined and calibrated on enough real data, are the plausible path to a confident detector.

## Reproduce

```bash
python scripts/legacy/diagnose_direction.py                 # 10-feature scorer is chance
python scripts/legacy/analyze_direction_signal.py           # NLL variants + length control (add --no-dapt)
python scripts/legacy/build_redaction_corpus.py             # synthetic same-author corpus
python scripts/legacy/extract_direction_features.py         # cache NLL features (T5 passes)
python scripts/legacy/train_direction_component.py          # train + full ladder + verdict
python scripts/build_external_pairs.py               # Jude -> 2 Peter (copy longer)
python scripts/build_lxx_pairs.py --swete-dir <path> # LXX Chronicles (30 blocks, mixed polarity)
python scripts/legacy/eval_external_direction.py --pairs <known_direction.json>
python scripts/legacy/analyze_fatigue.py                    # entity-level fatigue both-polarity test
python scripts/analyze_polarization.py               # RPM H1: which canons polarize (both polarities)
python scripts/train_polarization.py                 # RPM H2/H3/H4: aggregation + abstention + fatigue
python scripts/root_stemmata.py                      # RPM H5: pooled rooting -> stemma posterior + Farrer/Q
```
Reports land in `outputs/direction/`. Bootstrap CIs are pericope/block-grouped
(`synoptiq/evaluation/bootstrap.py`).
