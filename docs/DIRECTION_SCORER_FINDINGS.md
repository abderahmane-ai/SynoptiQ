# Direction Scorer (Phase 3): Findings

A rigorous, validation-gated investigation of whether copying **direction** between
parallel Koine passages can be detected. Short version: **at the granularity of short
parallel passages, there is no direction signal that is simultaneously
author-independent, length-independent, and transferable across corpora.** Every signal
that looks strong turns out to be a confound (length or Markan style) or a synthetic-
generator artifact. This is a boundary result about the problem, established by
experiment rather than assumed.

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

A follow-up prototype (`synoptiq/evaluation/fatigue.py`, `scripts/analyze_fatigue.py`)
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
python scripts/diagnose_direction.py                 # 10-feature scorer is chance
python scripts/analyze_direction_signal.py           # NLL variants + length control (add --no-dapt)
python scripts/build_redaction_corpus.py             # synthetic same-author corpus
python scripts/extract_direction_features.py         # cache NLL features (T5 passes)
python scripts/train_direction_component.py          # train + full ladder + verdict
python scripts/build_external_pairs.py               # Jude -> 2 Peter (copy longer)
python scripts/build_lxx_pairs.py --swete-dir <path> # LXX Chronicles (30 blocks, mixed polarity)
python scripts/eval_external_direction.py --pairs <known_direction.json>
python scripts/analyze_fatigue.py                    # entity-level fatigue both-polarity test
```
Reports land in `outputs/direction/`. Bootstrap CIs are pericope/block-grouped
(`synoptiq/evaluation/bootstrap.py`).
