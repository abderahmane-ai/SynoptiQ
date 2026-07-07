# Copying direction is not recoverable from the texts alone (closed negative result)

**Status: closed. Do not re-attempt corpus-free direction detection or hypothesis "scoring."**
Phase 3 (direction scorer) and Phase 6 (Bayesian hypothesis comparison) were removed on 2026-07-07
after an extended investigation established, three independent ways, that inferring the *direction*
of literary copying between the Synoptic Gospels from the texts alone is not achievable. This note
is the one-page record so the dead ends are not repeated.

## What was tried, and what each turned out to measure

| approach | result | what it really measured |
|---|---|---|
| 10 similarity-geometry features + logistic reg. | chance (0.33–0.36) | nothing usable |
| conditional-NLL / MDL codelength asymmetry | synoptic partial-r ~0.5, external ~0 | **Markan style + length** (flips sign with length polarity) |
| learned MDL head on a synthetic redaction corpus | 99% synthetic, fails real | generator artifact + length prior |
| RPM: connective-smoothing canon (καί→δέ) | 0.78 synoptic | a **Markan-style** detector; *backwards* on external known-direction pairs |
| non-reversible "RootModel" (editorial-fatigue drift, pooled) | external 0.645 CI[0.53,0.76]; gospels ≈ chance | real on narrative, **genre-limited** — does not transfer to the gospels |
| agreement-structure "TopologyModel" (centrality/hub) | excludes Griesbach/Augustinian | *topology*, not direction ("hub = source" is a fork assumption) |

## Why it is not solvable from the texts alone

The only computable asymmetry between two parallel texts is the **information gradient** (complex ⇄
simple). But historical copying runs *both* ways — scribes both simplify (abbreviate, smooth) and
expand (harmonize, add detail) — so the information gradient is **statistically independent of
historical direction**. Any corpus-free method (MDL, graph-rewrite complexity, causal inference,
textual-criticism canons) therefore defaults to "complex → simple" and scores at chance on a balanced
set of real transformations. We confirmed this empirically (every length/lexical/NLL feature flips
sign across the two known-direction length polarities: Jude→2 Peter copy-longer, LXX Chronicles
copy-shorter). It is also the conclusion of an algorithmic-information-theory analysis: to break the
symmetry one needs a *prior over each author's injection operator*, learnable only from a **large
independent corpus of that author's own writing** — which does not exist for the evangelists (Mark is
~11k words; the special material is tiny). The binding constraint is data, not method.

## What survived (and is still true)

- **Markan priority** is supported by the **agreement structure** (Mark is the hub; Mark's ~0.22
  singular-reading rate excludes the conflation hypotheses Griesbach/Augustinian). This is a
  *topology* result, not a per-pair direction signal, and it needs no direction scorer.
- **2SH vs Farrer is undetermined** by anything measurable in the texts (the double tradition is too
  sparse; the connective "signal" was Markan-style false precision).

## Consequence for the project

The transformer (KoineFormer) and the SynoptiQ corpus (Paper A) stand on their own. The next
transformer effort is **Phase 5 — Q reconstruction** (Fusion-in-Decoder: reconstruct Mark from
Matthew+Luke where ground truth exists, then transfer to the double tradition), a genuine generative
task that does not depend on solving direction.

*(Full experimental detail lived in the former `DIRECTION_SCORER_FINDINGS.md` / `DIRECTION_REDESIGN.md`
and the `synoptiq/direction`, `synoptiq/bayesian`, `synoptiq/legacy` trees, all removed in this
cleanup. Recover from git history if ever needed for a write-up.)*
