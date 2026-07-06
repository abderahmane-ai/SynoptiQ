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

## Reproduce

```bash
python scripts/diagnose_direction.py                 # 10-feature scorer is chance
python scripts/analyze_direction_signal.py           # NLL variants + length control (add --no-dapt)
python scripts/build_redaction_corpus.py             # synthetic same-author corpus
python scripts/extract_direction_features.py         # cache NLL features (T5 passes)
python scripts/train_direction_component.py          # train + full ladder + verdict
python scripts/build_external_pairs.py               # Jude -> 2 Peter (copy longer)
python scripts/build_lxx_pairs.py --swete-dir <path> # LXX Chronicles (copy shorter)
python scripts/eval_external_direction.py --pairs <known_direction.json>
```
Reports land in `outputs/direction/`. Bootstrap CIs are pericope/block-grouped
(`synoptiq/evaluation/bootstrap.py`).
