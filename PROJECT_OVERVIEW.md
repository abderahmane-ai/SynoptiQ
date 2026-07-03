# SynoptiQ: A Computational Framework for the Synoptic Problem

**Author:** Abderahmane Ainouche
**Status:** Phase 1–2 complete (Paper A). Phases 3–7 in development.

---

## The Problem

For over two centuries, scholars have debated how the Gospels of Matthew,
Mark, and Luke are literarily related. The dominant hypothesis — Markan
priority with a hypothetical sayings source Q — explains many patterns in
the text, but alternative hypotheses (Farrer–Goulder, Augustinian,
Griesbach) each have their defenders. The debate persists because the
evidence is textual, nuanced, and resists simple statistical treatment.

Computational methods — transformer models, causal direction detection,
Bayesian inference — offer a new way forward. But applying them to the
Synoptic Problem requires building the infrastructure first: a machine-readable
corpus, a language model that understands Koine Greek, and a suite of neural
tools designed specifically for textual criticism.

**SynoptiQ** is that infrastructure.

---

## The Project

SynoptiQ is a multi-phase computational framework that applies modern NLP
to the literary relationships among Matthew, Mark, and Luke. The project
proceeds in three stages, each yielding a standalone paper.

### Paper A: Foundation — KoineFormer + Synoptiq Corpus ✓

**Status: Complete. Manuscript available.**

The Synoptiq Corpus is a token-aligned, morphologically annotated dataset
of the Synoptic Gospels: 49,061 tokens across 170 Aland pericopes, with
235 pairwise Needleman-Wunsch alignments (8,855 aligned token pairs),
13-class CCAT part-of-speech tagging, and stratified train/val/test splits.

KoineFormer is a domain-adapted T5 encoder-decoder for Koine Greek,
produced by training LoRA adapters (3.7M parameters, 14 MB) on GreTa,
a Classical Greek T5. After domain-adaptive pre-training on 1.5M Koine
tokens (SBLGNT + Apostolic Fathers) with a Classical Greek replay buffer,
KoineFormer achieves 96.62% POS accuracy — a 28% error reduction over
the zero-shot baseline — while training 60× fewer parameters than full
fine-tuning.

**Key results:**
- 96.62% POS accuracy (linear probe on SynoptiQ test set)
- 81.34% lemmatisation accuracy
- 14 MB adapter checkpoint (runs on a laptop)
- Open release: model, dataset, and code under CC-BY-SA 4.0

**Artifacts:**
- Paper: `paper/main.tex`
- Model: [hf.co/ainouche-abderahmane/koineformer](https://huggingface.co/ainouche-abderahmane/koineformer)
- Dataset: [hf.co/datasets/ainouche-abderahmane/synoptiq-corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus)
- Code: [github.com/ainouche-abderahmane/SynoptiQ](https://github.com/ainouche-abderahmane/SynoptiQ)

---

### Paper B: Direction Detection + Editorial Fatigue

**Status: Design complete. Implementation in progress.**

The core question — "who copied from whom?" — is addressed with two
complementary neural methods:

**Direction Scorer.** Cross-attention asymmetry between parallel
passages produces 8 directional features that feed a 3-way classifier
(A→B, B→A, independent). An adversarial gradient reversal layer strips
authorship style to isolate the copying signal. Trained on triple-tradition
pericopes where the Mark→Matthew and Mark→Luke direction is known.

**Editorial Fatigue Model.** Position-weighted consistency loss
quantifies the well-documented phenomenon that a copying author tends to
revert to their own stylistic norms over the course of a pericope.
`L_fatigue = Σ w(i) · D_KL(edit_dist_i || source_dist_i)`, where the
weight decays exponentially with position.

**Expected contribution:** The first neural model to simultaneously
detect copying direction AND quantify editorial fatigue on the same
passages, providing converging evidence from two independent signals.

---

### Paper C: Q Reconstruction + Bayesian Model Comparison

**Status: Architecture designed. Awaiting Papers A+B.**

**Q Reconstruction.** Fusion-in-Decoder (FiD): Matthew and Luke are
encoded independently, their hidden states concatenated, and a decoder
with cross-attention reconstructs the shared source. Trained first on
triple tradition (Matthew+Luke → reconstruct Mark, where ground truth
exists), then transferred to double tradition for Q reconstruction.

**Bayesian Model Comparison.** Direction scorer outputs → MC Dropout
(T=20) → per-pericope uncertainty estimates (μ_i, σ²_i) → PyMC Beta
hierarchical models. Four hypotheses compared via bridge sampling:
Two-Source (2SH), Farrer–Goulder (FGH), Augustinian, Griesbach.

**Expected contribution:** The first probabilistic comparison of all
four major Synoptic hypotheses using neural evidence, with full
uncertainty quantification — a methodologically novel approach that
treats the Synoptic Problem as a formal Bayesian model selection task.

---

## Technical Architecture

```
                   ┌──────────────────────────┐
                   │    Synoptiq Corpus        │
                   │  49K tokens, 170 pericopes │
                   │  235 alignments, CCAT POS  │
                   └─────────────┬────────────┘
                                 │
                   ┌─────────────▼────────────┐
                   │      KoineFormer          │
                   │  GreTa + LoRA (3.7M)      │
                   │  Encoder → hidden states  │
                   └─────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
  ┌───────▼───────┐    ┌────────▼────────┐    ┌────────▼────────┐
  │  Direction     │    │  Editorial      │    │  Q              │
  │  Scorer        │    │  Fatigue        │    │  Reconstruction │
  │  Cross-attn    │    │  Position-      │    │  FiD: Mt+Lk→Q   │
  │  asymmetry     │    │  weighted KL    │    │  Transfer learn  │
  └───────┬───────┘    └────────┬────────┘    └────────┬────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                   ┌─────────────▼────────────┐
                   │   Bayesian Comparison     │
                   │   PyMC hierarchical       │
                   │   2SH vs FGH vs Aug vs Gr │
                   │   Bridge sampling + BF    │
                   └──────────────────────────┘
```

---

## What Makes This Different

Most computational work on the Synoptic Problem has used surface-level
features: word frequencies, n-gram overlap, stylometric distances.
SynoptiQ operates at a deeper level:

- **Contextual representations** (transformer hidden states) rather than
  surface forms
- **Causal direction detection** rather than correlation
- **Full uncertainty quantification** via Bayesian inference, rather than
  point estimates
- **Adversarial debiasing** to separate copying signal from authorial style
- **Everything open:** model, dataset, code, training logs, evaluation scripts

---

## Current Releases

| Artifact | Location | License |
|----------|----------|---------|
| KoineFormer model | [hf.co/ainouche-abderahmane/koineformer](https://huggingface.co/ainouche-abderahmane/koineformer) | CC-BY-SA 4.0 |
| Synoptiq Corpus | [hf.co/datasets/ainouche-abderahmane/synoptiq-corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus) | CC-BY-SA 4.0 |
| Paper A manuscript | `paper/main.tex` (this repo) | CC-BY-SA 4.0 |
| Full codebase | [github.com/ainouche-abderahmane/SynoptiQ](https://github.com/ainouche-abderahmane/SynoptiQ) | CC-BY-SA 4.0 |

---

## Contact

Abderahmane Ainouche
GitHub: [@ainouche-abderahmane](https://github.com/ainouche-abderahmane)
