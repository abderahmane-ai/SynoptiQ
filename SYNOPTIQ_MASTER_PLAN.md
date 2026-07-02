# SynoptiQ — Master Plan

## A Multi-Task Neural Source Criticism Framework for the Synoptic Problem

**Version:** 1.0  
**Date:** 2026-06-24  
**Status:** Research Design Phase

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [The Problem & Why It Matters](#2-the-problem--why-it-matters)
3. [State of the Field](#3-state-of-the-field)
4. [Core Innovations](#4-core-innovations)
5. [Architecture](#5-architecture)
6. [Phase 1: Data Foundation](#6-phase-1-data-foundation)
7. [Phase 2: KoineFormer — Domain-Adaptive Language Model](#7-phase-2-koineformer--domain-adaptive-language-model)
8. [Phase 3: Source Detection & Causal Direction](#8-phase-3-source-detection--causal-direction)
9. [Phase 4: Editorial Tendency Modeling](#9-phase-4-editorial-tendency-modeling)
10. [Phase 5: Proto-Q Reconstruction](#10-phase-5-proto-q-reconstruction)
11. [Phase 6: Bayesian Model Comparison](#11-phase-6-bayesian-model-comparison)
12. [Phase 7: Interpretability & Robustness](#12-phase-7-interpretability--robustness)
13. [Technical Stack](#13-technical-stack)
14. [Risk Analysis & Mitigation](#14-risk-analysis--mitigation)
15. [Publication Strategy](#15-publication-strategy)
16. [Timeline & Milestones](#16-timeline--milestones)
17. [References](#17-references)

---

## 1. Executive Summary

The Synoptic Problem — determining the literary relationships among the Gospels of Matthew, Mark, and Luke — has divided biblical scholars since 1863. At stake is whether a hypothetical lost source ("Q") existed, or whether simpler dependency chains explain the data. Despite 160 years of debate, no consensus has emerged.

**SynoptiQ** applies modern transformers, causal direction modeling, and Bayesian inference to this problem. It is designed to be the definitive computational treatment — the paper that forces the field to engage with ML evidence at the level of the best philological scholarship.

### What we do that no one has done:

1. **Direction scorer** — cross-attention transformer that learns asymmetric traces of copying direction between parallel passages. The first transformer-based detector of "who copied whom" in ancient text. No prior neural model exists for this task (Grozea & Popescu 2010's Encoplot uses n-gram dot-plots; no deep learning approaches exist).

2. **Proto-Q reconstruction** — Fusion-in-Decoder (FiD) architecture that takes Matthew + Luke's versions of a saying and reconstructs the inferred common ancestor. Validated on a known-ground-truth proxy (reconstructing Mark from Matthew + Luke in triple tradition) and against the IQP Critical Edition of Q.

3. **Full Bayesian model comparison** — the first Bayesian treatment of the Synoptic Problem. Computes Bayes factors for all four major hypotheses (2SH, FGH, Augustinian, Griesbach) with prior sensitivity analysis. No Bayesian model comparison of Synoptic hypotheses exists in the literature; all prior statistical work is frequentist (Honoré 1968, Abakuks 2006/2007, Robert-Hayek et al. 2023).

4. **Editorial tendency learning** — formalizes Goodacre's "editorial fatigue" as a learnable loss function, trained on Chronicles → Samuel/Kings (OT ground truth for scribal copying behavior).

5. **Interpretability-to-scholarship bridge** — SHAP feature importance compared to Hawkins' *Horae Synopticae* (1899) as a sanity check; BERTViz attention visualization; multi-edition sensitivity analysis (NA28 vs. Textus Receptus vs. Majority Text).

### The organizing principle:

The project's core bet is that **editorial direction leaves asymmetric, detectable traces** in parallel text — traces that transformers can learn but that hand-crafted features miss. Everything else (KoineFormer, Q reconstruction, Bayesian comparison) serves or validates this central claim.

---

## 2. The Problem & Why It Matters

### 2.1 The Data

Three Greek texts with extraordinarily high verbatim agreement:

| Tradition | Content | ~Verse Count |
|-----------|---------|-------------|
| Triple Tradition | Shared by Matthew, Mark, Luke | ~330 verses |
| Double Tradition | Shared by Matthew + Luke, absent from Mark | ~235 verses |
| Markan Sondergut | Unique to Mark | ~40 verses |
| Matthean Sondergut | Unique to Matthew | ~310 verses |
| Lukan Sondergut | Unique to Luke | ~520 verses |

### 2.2 The Four Major Hypotheses

| Hypothesis | Composition Order | Q? | Key Proponents |
|------------|-------------------|-----|----------------|
| **Two-Source (2SH)** | Mk → Mt, Lk (independent) | Yes | Weisse (1838), Streeter (1924), Kloppenborg, Tuckett |
| **Farrer-Goulder (FGH)** | Mk → Mt → Lk | No | Farrer (1955), Goulder (1974/89), Goodacre (2002) |
| **Augustinian** | Mt → Mk → Lk | No | Augustine (c.400), Wenham (1992) |
| **Griesbach (Two-Gospel)** | Mt → Lk → Mk | No | Griesbach (1783), Farmer (1964) |

**Current scholarly distribution (rough):** 2SH dominates globally (~65-75%), FGH is strongest in British scholarship (~15-20%), Griesbach and Augustinian are minority positions (~5-10% each). Markan priority is near-consensus; the debate is now about the Matthew-Luke relationship.

### 2.3 Why ML Can Move the Needle

The debate has been stuck because:

1. **The evidence is pattern-based, not smoking-gun.** No external document settles the question. The arguments are about aggregate patterns: word agreement percentages, order statistics, redactional tendencies.

2. **Hand-crafted features miss non-linear interactions.** Robert-Hayek et al. (2023) used 103 stylometric features in a Random Forest. This is good but limited — a transformer can learn compositional features (e.g., "Matthean syntax operating on Markan vocabulary") that manual feature engineering cannot.

3. **Direction detection has never been modeled computationally.** The core question — "given two parallel texts, which direction is the copying?" — has one prior computational method (Encoplot, 2010, n-gram asymmetry, ~75% accuracy) and zero transformer-based approaches.

4. **Uncertainty has never been quantified.** All prior work reports point estimates. No Bayesian treatment exists. Bayes factors with prior sensitivity give scholars something they can actually reason with: "Under these assumptions, the data favor this hypothesis by this much."

---

## 3. State of the Field

### 3.1 Prior Statistical Work

| Author | Year | Method | Finding | Limitations |
|--------|------|--------|---------|-------------|
| Honoré | 1968 | Word-count statistics, triple-link probability | Mark as middle term | Frequentist, no direction model |
| Abakuks | 2006/2007 | Hidden Markov Models over word agreement | *Matthean priority* gave best fit under modified model | Frequentist, no Bayesian treatment |
| Mattila | 2004 | Sayings-only word counting | DT-TT gap nearly disappears | Methodology-dependent |
| McLoughlin | 2014 | Comprehensive recount | ~1.8% gap persists | Counting only; no direction |
| **Robert-Hayek et al.** | **2023** | **Random Forest, 103 features, pericope-level** | **Supports 2SH over FGH** | **No transformer, no direction, no Q reconstruction** |
| **SynoptiQ (proposed)** | **2026** | **Multi-task transformer + causal direction + Bayesian comparison** | **TBD** | **—** |

### 3.2 Language Models for Ancient Greek

| Model | Architecture | Performance | Limitations |
|-------|-------------|-------------|-------------|
| Ancient-Greek-BERT | BERT-base, warm-start from Modern Greek | SOTA on POS, >90% accuracy | Mixed corpus (Attic + Koine + Byzantine) dilutes Koine signal |
| **GreBERTa** (ACL 2023) | RoBERTa, monolingual AG | UAS 88.20 on UD Perseus | Collapses on AGDT (UAS 58.85) — parsing head not robust |
| **GreTa** (ACL 2023) | T5, monolingual AG | 91.17 F1 lemmatization on AGDT | Encoder-decoder, good for generation |
| **Trankit** (XLM-RoBERTa) | Multilingual (100 langs) | **Best overall on AGDT** (UPOS 96.18, LAS 76.67) | Not AG-specific; no generative capability |
| **SPhilBERTa** (2025) | Trilingual Sentence-RoBERTa (AG + Latin + English) | Cross-lingual semantic similarity | Sentence-level only; no token-level |

**Key insight:** No model exists that is both (a) specifically pre-trained on Koine Greek (NT + LXX register) AND (b) capable of sequence-to-sequence generation. The closest candidate is **GreTa** (T5-based, monolingual AG), but its training data is classical/literary, not Koine. We need **KoineFormer** — a domain-adapted encoder-decoder for the Hellenistic Jewish/Christian Greek register.

### 3.3 Text Reconstruction

The most directly applicable architecture is **Fusion-in-Decoder (FiD)** (Izacard & Grave, 2021):

- Encode each source independently (Matthew encoder, Luke encoder)
- Concatenate hidden states
- Single decoder with cross-attention over the concatenated representation
- Decoder can selectively draw from either source at each token position

This maps directly onto the IQP's manual methodology: scholars adjudicate between Matthew and Luke at each point of disagreement to recover Q. FiD learns to do this automatically.

**Critical validation strategy:** Train on triple tradition with Mark as target (Matthew + Luke → reconstruct Mark), evaluate against known Mark text. Then transfer to double tradition (Matthew + Luke → reconstruct Q). This provides supervised training signal where none otherwise exists.

### 3.4 Causal Direction Detection

**Encoplot** (Grozea & Popescu, 2010) is the ONLY prior method for "who copied whom" from text alone:
- N-gram dot-plot asymmetry: copied n-grams produce a "parasitic cloud" elongated parallel to the source axis
- ~75.4% accuracy overall; 77.9% for unobfuscated long passages; drops to 68.1% for short passages
- No transformer-based direction detector exists — this is a genuine research gap

**Our approach:** Cross-attention between parallel passage encodings is inherently asymmetric — the attention pattern from text A attending to text B differs from B attending to A when one is the source and the other is derived. We train a classifier on these **cross-attention asymmetry features**.

---

## 4. Core Innovations

### 4.1 The Direction Scorer (PRIMARY CONTRIBUTION)

**Problem:** Given two parallel Greek texts (e.g., Matthew's and Luke's versions of a pericope), determine the direction of literary dependence.

**Method:**
1. Encode each text independently through KoineFormer
2. Compute bidirectional cross-attention: Text A attending to Text B, and Text B attending to Text A
3. Extract asymmetry features from the cross-attention matrices
4. Classify direction: A→B, B→A, or independent

**Why this works:** When an author edits a source, they produce a text that is *psycholinguistically derivative* — certain source-phrase structures persist in the derived text, but not vice versa. Cross-attention patterns capture these asymmetries because the derived text's tokens attend more consistently to specific source-token clusters than the reverse.

**Training signal:** Triple tradition gives us known direction (Mark → Matthew, Mark → Luke). The model learns what "source→derived" cross-attention patterns look like. Trained discriminatively on pericope-level direction labels.

**Validation:** Leave-one-out on triple tradition; Chronicles → Samuel/Kings as OT transfer test.

### 4.2 Editorial Fatigue as a Learnable Loss (SECONDARY CONTRIBUTION)

Goodacre's "editorial fatigue" — where a copyist changes the beginning of a pericope but lapses back to the source's wording — is formalized as a **positional consistency loss**:

L_fatigue = Σ_i w(i) · d(h_i, h_target)

where `i` is token position within the pericope, `w(i)` is a monotonically decreasing weight function (changes at the end matter less — the copyist has already "fatigued"), and `d(h_i, h_target)` measures how far token `i`'s representation diverges from the inferred source representation.

The model is penalized for *inconsistent* editing patterns — changes that are not maintained throughout the pericope. This directly operationalizes Goodacre's insight.

**Training:** Ground truth editorial fatigue examples exist in triple tradition (Mark → Matthew/Luke) and in Chronicles → Samuel/Kings (Hebrew Bible copyist behavior). These provide labeled examples of "inconsistent" vs. "consistent" editing.

### 4.3 Triple-Tradition-to-Double-Tradition Transfer (METHODOLOGICAL CONTRIBUTION)

The key methodological innovation: **train on known source relationships, test on unknown ones.**

- **Training regime A:** Given (Mark, Matthew, Luke) in triple tradition, learn editorial tendencies and direction features where Mark is the known source
- **Training regime B:** Given (Chronicles, Samuel, Kings) in OT, learn editorial tendencies where Chronicles's use of Samuel/Kings is the known ground truth
- **Inference:** Apply learned models to double tradition (Matthew, Luke) where the source is unknown

This is transfer learning applied to *textual criticism* — analogous to training on labeled phylogenetic data and testing on an unresolved phylogeny.

### 4.4 Bayesian Model Comparison of Synoptic Hypotheses (STATISTICAL CONTRIBUTION)

The first fully Bayesian treatment of the Synoptic Problem. Components:

1. **Hierarchical likelihood:** Words nested within verses within pericopes within gospels
2. **Direction scores as data:** The direction scorer outputs per-pericope direction probabilities; these become the observations in the Bayesian model
3. **Four competing models:** Each hypothesis specifies a different pattern of directional dependencies
4. **Bayes factors:** Computed via bridge sampling (R `bridgesampling`) or SMC (PyMC)
5. **Prior sensitivity:** Full grid of prior hyperparameters, reporting BF range, not point estimate

**Why this matters:** It gives scholars a quantitative, uncertainty-aware answer to "how much does the data favor one hypothesis over another?" — something the field has never had.

### 4.5 The Hawkins Sanity Check (INTERPRETABILITY CONTRIBUTION)

Sir John Hawkins' *Horae Synopticae* (1899, 2nd ed. 1909) identified 198 "characteristic" words and phrases of Matthew, 151 of Mark, and 413 of Luke — the classic manual feature set for Synoptic stylometry.

We compute SHAP feature importance from our direction scorer and compare against Hawkins' list:
- **Convergent features:** SHAP-important features that appear in Hawkins → validation of both methods
- **Divergent features:** SHAP-important features NOT in Hawkins → candidate discoveries (transformer-learned patterns that escaped manual analysis)
- **Hawkins-only features:** Features in Hawkins but with low SHAP importance → candidates for scholarly re-examination

This bridges 19th-century philology and 21st-century deep learning in a way that makes the paper legible to both communities.

---

## 5. Architecture

### 5.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SYNOPTIQ ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ SBLGNT +     │   │ LXX +        │   │ NA28 / ECM   │                │
│  │ MorphGNT     │   │ Josephus +   │   │ Apparatus    │                │
│  │ (CC-BY)      │   │ Apost Fathers│   │ (variants)   │                │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            │                                            │
│                    ┌───────▼────────┐                                   │
│                    │  Corpus Builder │                                  │
│                    │  + Aland Align  │                                  │
│                    └───────┬────────┘                                   │
│                            │                                            │
│         ┌──────────────────┼──────────────────┐                         │
│         │                  │                  │                         │
│  ┌──────▼──────┐   ┌───────▼───────┐  ┌───────▼───────┐                │
│  │Triple Trad  │   │Double Trad    │  │OT Transfer    │                │
│  │(Mk,Mt,Lk)   │   │(Mt,Lk only)   │  │(Chr,Sam,Kgs)  │                │
│  └──────┬──────┘   └───────┬───────┘  └───────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            │                                            │
│                    ┌───────▼────────┐                                   │
│                    │   KoineFormer  │                                   │
│                    │ (Enc-Dec, LoRA)│                                   │
│                    └───────┬────────┘                                   │
│                            │                                            │
│         ┌──────────────────┼──────────────────┬──────────────────┐      │
│         │                  │                  │                  │      │
│  ┌──────▼──────┐   ┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼────┐│
│  │Source       │   │Direction      │  │Editorial      │  │Q Recon-   ││
│  │Detector     │   │Scorer         │  │Drift Module   │  │structor   ││
│  │(2× vs 3×    │   │(Cross-Attn    │  │(Fatigue Loss) │  │(FiD Arch) ││
│  │ tradition)  │   │Asymmetry)     │  │               │  │           ││
│  └──────┬──────┘   └───────┬───────┘  └───────┬───────┘  └──────┬────┘│
│         │                  │                  │                  │      │
│         └──────────────────┼──────────────────┘                  │      │
│                            │                                     │      │
│                    ┌───────▼────────┐                            │      │
│                    │   Synthesis    │◄───────────────────────────┘      │
│                    │                │                                   │
│                    │ • Bayes Factor │                                   │
│                    │ • MCMC         │                                   │
│                    │ • 4 Hypotheses │                                   │
│                    └───────┬────────┘                                   │
│                            │                                            │
│                    ┌───────▼────────┐                                   │
│                    │Interpretability│                                   │
│                    │ • SHAP         │                                   │
│                    │ • BERTViz      │                                   │
│                    │ • Hawkins 1899 │                                   │
│                    │ • Sensitivity  │                                   │
│                    └────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 KoineFormer Design

**Base architecture:** T5-base (encoder-decoder, ~220M params) — we need generation capability for Q reconstruction. BERT-only would work for classification/direction but not for the reconstruction task.

**Why T5 not BERT:** T5's encoder-decoder structure supports both:
- Encoder-only for classification/direction tasks (via pooling + classification head)
- Full encoder-decoder for Q reconstruction (FiD architecture)
- Shared encoder provides unified Koine representations across all tasks

**Pre-training strategy (3-stage):**

1. **Stage 0: Base initialization.** Warm-start from `pranaydeeps/Ancient-Greek-BERT` encoder weights for the encoder. Decoder initialized from scratch or from GreTa decoder weights.

2. **Stage 1: Domain-adaptive pre-training (DAPT).** Further MLM pre-training on Koine corpus (SBLGNT + LXX + Apostolic Fathers + Josephus). This is FULL fine-tuning, not LoRA — LoRA is insufficient for domain adaptation when the domain gap is as large as classical Attic → Koine.
   - **Critical benchmark:** After DAPT, evaluate MLM perplexity on held-out Koine. Must be within 5% of a from-scratch RoBERTa trained on Koine only. If not, increase DAPT steps or consider from-scratch pre-training.

3. **Stage 2: Multi-task fine-tuning with LoRA.**
   - Task 1: Masked Language Modeling (Koine-specific)
   - Task 2: POS tagging (MorphGNT labels)
   - Task 3: Dependency parsing (PROIEL treebank, biaffine head)
   - Task 4: Lemmatization (MorphGNT lemmas)
   - Task 5: Pericope boundary detection (Aland alignment)

**Tokenization:**
- Keep Ancient-Greek-BERT's subword tokenizer (de-accentuation + lowercasing)
- Add domain-specific tokens: nomina sacra (ΘΣ, ΚΣ, ΙΣ, ΧΣ, etc. — common in NT manuscripts)
- Max sequence length: 512 tokens (covers ~98% of pericopes)

### 5.3 Direction Scorer Architecture

```
Matthew Text ──► Encoder ──► h_M ──┐
                                     ├──► Cross-Attention M→L ──► asymmetry features ──► MLP ──► Direction
Luke Text ────► Encoder ──► h_L ──┘       Cross-Attention L→M ──► asymmetry features ──┘
```

**Cross-attention asymmetry features:**

For each token pair (i in M, j in L), the cross-attention weight α_{i→j} represents how much token i in Matthew attends to token j in Luke.

Asymmetry features:
1. **Mean attention entropy:** Is Matthew's attention to Luke more focused (lower entropy) than Luke's attention to Matthew? A derived text's attention to its source should be more focused.
2. **Attention alignment with word order:** Does the attention diagonal (i ≈ j) dominate? Parallel texts with direct copying show strong diagonal attention.
3. **Attention dispersion:** Standard deviation of attention weights per query token. Source→derived attention shows higher dispersion (derived tokens attend broadly to source context).
4. **Mutual information between attention matrices:** I(α^{M→L}; α^{L→M}). Lower MI suggests asymmetric relationship.
5. **Learned asymmetry features:** A small transformer layer that takes the concatenated cross-attention matrices and learns to extract direction-discriminative features.

**Classification head:** 3-way softmax over {M→L, L→M, independent} or 2-way for specific hypothesis tests.

**Training:** Pericope-level supervision from triple tradition (known direction: Mark → Matthew, Mark → Luke). Also trained adversarially: can the model still detect direction when the pericopes are shuffled (forcing it to learn genuine directional features rather than content-specific patterns)?

### 5.4 Q Reconstruction (FiD) Architecture

```
Matthew Text ──► Encoder ──► h_M ──┐
                                     ├──► Concatenate ──► Decoder (cross-attention) ──► Q text
Luke Text ────► Encoder ──► h_L ──┘
```

**Training:**

*Stage 1: Mark reconstruction (triple tradition)*
- Input: (Matthew's Mark-based version, Luke's Mark-based version)
- Target: Mark (the known source)
- This teaches the decoder to factor out Matthean and Lukan edits and recover the common source

*Stage 2: Q reconstruction (double tradition)*
- Input: (Matthew's Q-based version, Luke's Q-based version)
- Target: IQP Critical Edition {A}-rated readings (~3,000-4,000 words at high confidence)
- The model transfers its learned "editorial factoring" ability from triple to double tradition

**Inter-source cross-attention (critical addition):**

Before concatenation, add cross-attention layers between h_M and h_L:
```
h_M' = h_M + CrossAttn(h_M, h_L)    # Matthew's encoding refined by seeing Luke
h_L' = h_L + CrossAttn(h_L, h_M)    # Luke's encoding refined by seeing Matthew
```

This lets the model explicitly model agreement/disagreement patterns before decoding. Where Matthew and Luke agree, both encodings converge; where they disagree, the model learns to identify which is more likely to preserve Q.

**Generation constraints:**
- Beam search with n-best hypotheses (n=5)
- Temperature sampling at inference for uncertainty estimation
- Constrained decoding: vocabulary restricted to SBLGNT lexicon (prevents hallucinated words)
- Minimum BLEU/ROUGE against both Matthew and Luke texts (prevents degenerate copies)

### 5.5 Bayesian Model Specification

**Data model (pericope level):**

For each pericope p in the double tradition:
- y_p = direction score from the direction scorer (probability that Matthew is the source for Luke in this pericope)
- Under 2SH: y_p ~ Beta(α_independent, β_independent) — neither is source, both independently use Q
- Under FGH: y_p ~ Beta(α_Mt→Lk, β_Mt→Lk) — Matthew is source for Luke

**Hierarchical structure:**
```
Level 1 (word): direction indicators at token alignments
Level 2 (verse): aggregated direction scores per verse
Level 3 (pericope): y_p ~ Beta(μ_h, κ) where h indexes the hypothesis
Level 4 (hypothesis): μ_h determined by the hypothesis structure
```

**Prior elicitation:**
- Domain-expert priors: survey NT scholars for plausible parameter ranges
- Uninformative baseline: Beta(1,1) = Uniform(0,1)
- Informed baseline: Beta(a, b) calibrated to Honoré/Abakuks word-agreement statistics

**Computation:**
- Type 1 (simple): Bridge sampling via R `bridgesampling` package — lowest computational cost
- Type 2 (medium): SMC via PyMC (`pm.sample_smc()`) — returns `log_marginal_likelihood`
- Type 3 (thorough): Thermodynamic integration — most robust but most expensive

**Prior sensitivity:**
- Full grid search over prior hyperparameters
- Report BF contour plots, not point estimates
- "Under prior A (scholar X's assessment), BF = Y. Under prior B (scholar Y's assessment), BF = Z."

### 5.6 Interpretability Pipeline

**SHAP analysis:**
- Compute SHAP values for the direction scorer's predictions
- Identify the 10-15 pericopes that most discriminate between 2SH and FGH
- Compare to Hawkins' *Horae Synopticae* (1899) — the classic manual list of characteristic features

**BERTViz:**
- Visualize KoineFormer attention patterns in key pericopes
- Identify which tokens/constructions the model attends to when making direction decisions
- Produce publication-quality figures for the paper

**Multi-edition sensitivity:**
- Run the full pipeline on NA28, Textus Receptus, and Majority Text editions
- Does the verdict change? If yes → text-critical sensitivity is high, report with appropriate caution
- If no → robustness demonstrated, strengthens conclusions

**Counterfactual analysis:**
- For each key pericope: systematically perturb words/phrases identified as "Matthean redactional"
- Does the direction scorer flip? If so → evidence that those features genuinely drive the direction signal

---

## 6. Phase 1: Data Foundation

### 6.1 Corpus Assembly

**Primary texts (CC-BY or public domain — legally safe for ML):**

| Resource | Content | License | Tokens (est.) | Format |
|----------|---------|---------|---------------|--------|
| SBLGNT | Greek NT | CC-BY 4.0 | ~138,000 | XML, USFM |
| MorphGNT v6.12 | Morphological tags aligned to SBLGNT | CC-BY-SA 4.0 | ~138,000 | TSV |
| SBLGNT-lowfat | Syntax trees | CC-BY-SA 4.0 | ~138,000 | XML (XPath-compatible) |
| Nestle 1904 GNT | Alternative text base | Public Domain | ~138,000 | Text-Fabric |
| N1904-TF | Aland pericope numbers mapped to tokens | Open | ~64,000 (with pericope IDs) | Text-Fabric |
| Open Apostolic Fathers | Koine control corpus | CC-BY-SA 4.0 | ~35,000 | XML/TSV |
| First1KGreek | Broad Greek corpus (Homer → 300 CE) | CC-BY-SA 4.0 | ~23M | TEI XML |
| Perseus (Scaife) | Greek literary corpus | CC-BY-SA | ~10M | TEI XML |

**Restricted-access resources (use for validation only, not redistribution):**

| Resource | Content | Restriction |
|----------|---------|-------------|
| NA28 critical text | Modern critical edition | Copyright DBG — cannot include in public datasets |
| ECM Mark | Full digital apparatus | Freely viewable online; API for transcript retrieval |
| Rahlfs-Hanhart LXX | Standard Septuagint edition | Copyright DBG; SWORD module available |
| Gospel of Thomas (DBG ed.) | Coptic text + Greek retroversion | Copyright DBG; P.Oxy. fragments are public domain |

**Data pipeline:**
```
1. Download SBLGNT XML from GitHub (LogosBible/SBLGNT)
2. Load MorphGNT TSV (morphgnt/sblgnt)
3. Align token-by-token (SBLGNT words ↔ MorphGNT parses)
4. Load N1904-TF Aland pericope numbers
5. Merge: for each token, we have (text, lemma, POS, morphology, syntax, pericope_id)
6. Split into triple tradition, double tradition, single tradition using pericope presence/absence
```

### 6.2 Pericope Alignment

**Source:** Aland's *Synopsis Quattuor Evangeliorum* pericope numbering, mapped in N1904-TF.

**Alignment method:**
- **Coarse:** Pericope-level grouping by Aland number
- **Fine:** Verse-level alignment within each pericope
- **Finest:** Word-level alignment via Needleman-Wunsch global alignment on the Greek text, using lemma + POS as the scoring matrix (not surface form — accounts for morphological variation)

**Output:** For each pericope, a matrix of shape (n_tokens_gospel_A, n_tokens_gospel_B) where entry (i, j) indicates whether token i in A aligns with token j in B. This alignment matrix is the input to the cross-attention-based direction scorer.

### 6.3 Augmentation

**Bootstrap resampling:**
- Sample pericopes with replacement, stratified by tradition type (triple/double) and gospel
- Generate B=100 bootstrap datasets for uncertainty estimation

**Moving windows:**
- Slide a window of 3-5 verses across each pericope
- Train on both full-pericope and window-level representations
- Increases effective sample size for short texts

**Synthetic editing pairs:**
- Train a seq2seq model on Mark → Matthew and Mark → Luke triple-tradition pairs
- Use the trained model to generate synthetic "Matthew-style" and "Luke-style" edits of known Koine texts
- Creates additional training data for the direction scorer

**Ancient compositional simulation:**
- Simulate ancient scribal behaviors (haplography, dittography, harmonization, transposition) on known texts
- Creates labeled data for specific error types
- Validates model robustness to scribal noise

### 6.4 Data Splits

**Critical principle:** Pericopes must NOT be split across train/test. A pericope is the atomic unit of analysis — splitting it would leak information about editorial patterns within the same composition unit.

| Split | Composition | Purpose |
|-------|-------------|---------|
| Train | 60% of pericopes, stratified by type | Model training |
| Validation | 20% of pericopes | Hyperparameter tuning, early stopping |
| Test | 20% of pericopes | Final evaluation (used ONCE) |
| OT Transfer | All Chronicles ↔ Samuel/Kings pericopes | Separate validation of direction scoring |

**Stratification:** By tradition type (triple, double), gospel presence, and narrative vs. discourse material. Ensure balanced representation of key pericope categories (parables, miracles, controversy stories, passion narrative).

---

## 7. Phase 2: KoineFormer — Domain-Adaptive Language Model

### 7.1 Pre-training Corpus

**Composition:**
- SBLGNT (full NT): ~138,000 tokens
- LXX (Septuagint): ~600,000 tokens (SWORD module extraction)
- Apostolic Fathers: ~35,000 tokens
- Josephus (PACE/Perseus): ~300,000 tokens
- Total: ~1,073,000 tokens

**Comparison to prior models:**
- Ancient-Greek-BERT: ~82M tokens (mixed Attic + Koine + Byzantine + Modern)
- Our Koine corpus: ~1M tokens (pure Koine, but much smaller)
- **This is a data-scarcity regime.** Domain-adaptive pre-training is essential but must be done carefully to avoid catastrophic forgetting.

### 7.2 Stage 1: Domain-Adaptive Pre-training (Full Fine-Tune)

**Rationale for full fine-tuning (not LoRA):**
- LoRA adapters have rank r << d_model. A low-rank update cannot shift the underlying distribution from classical Attic to Koine Greek — the domain gap is too large.
- Full fine-tuning on the encoder (while keeping the decoder warm-started from GreTa) is the right trade-off.
- **Go/No-go benchmark:** After DAPT, MLM perplexity on held-out Koine must be within 5% of a from-scratch RoBERTa trained on Koine only. If the gap is larger, we increase DAPT steps or consider from-scratch pre-training (expensive but possible with ~1M tokens and a small model).

**Configuration:**
- Batch size: 32
- Learning rate: 5e-5 with linear warmup (1000 steps) and cosine decay
- Training steps: 50,000-100,000 (monitor validation perplexity)
- Masking rate: 15% (standard BERT)
- Sequence length: 512

### 7.3 Stage 2: Multi-Task Fine-Tuning with LoRA

**Tasks:**
1. **MLM** (weight: 0.3) — maintains Koine language understanding
2. **POS tagging** (weight: 0.2) — MorphGNT labels; linear classifier on top of encoder
3. **Dependency parsing** (weight: 0.2) — PROIEL treebank; biaffine parser on top of encoder
4. **Lemmatization** (weight: 0.15) — MorphGNT lemmas; linear classifier
5. **Pericope boundary detection** (weight: 0.15) — binary classifier on [SEP] tokens

**LoRA configuration:**
- Rank r = 16 (task adaptation is lower-rank than domain adaptation)
- Alpha = 32
- Target modules: all attention weight matrices (q_proj, k_proj, v_proj, o_proj)
- One LoRA adapter per task; shared base model

**Evaluation:**
- POS accuracy on held-out MorphGNT verses
- UAS/LAS on held-out PROIEL sentences
- Lemma accuracy on held-out MorphGNT lemmas
- Compare to Trankit, GreBERTa, Ancient-Greek-BERT baselines

### 7.4 Stage 3: Task-Specific Heads

**Source detection head:**
- Pool encoder outputs (mean pooling over tokens)
- 2-layer MLP with ReLU → 2-way softmax (double tradition vs. triple tradition)
- Trained on Luke's pericopes (the only gospel where the distinction is debated)

**Direction scorer head:**
- Cross-attention computation (see Architecture, section 5.3)
- 3-layer MLP on asymmetry features → 3-way softmax
- Trained on triple tradition with known direction

**Editorial drift head:**
- Position-aware regression: predict, for each token in the derived text, how much it diverges from the source
- Training signal: token-level edit distance between source and derived text

---

## 8. Phase 3: Source Detection & Causal Direction

### 8.1 Source Detection Task

**Task:** Given Luke's version of a pericope, classify as double tradition (Q material) or triple tradition (Mark material).

**Why this matters:** If Luke's double and triple tradition material are stylistically distinguishable, it supports the 2SH (different sources, different styles). If they are NOT distinguishable, it weakens the argument for Q. Robert-Hayek et al. (2023) found they ARE distinguishable (p=0.03).

**Our approach:**
- Encode each pericope through KoineFormer
- Extract [CLS] token representation
- Classify via MLP head
- 10-fold cross-validation, stratified by pericope type

**Baselines:**
- Random Forest on 103 Robert-Hayek features (replication)
- Ancient-Greek-BERT + linear classifier (off-the-shelf baseline)
- Majority class (trivial baseline)

### 8.2 Direction Scoring Task

**Task:** Given two parallel texts (A, B), predict direction: A→B, B→A, or independent.

**Training data:**

| Source | Direction | N (pericopes) | Ground Truth Quality |
|--------|-----------|---------------|---------------------|
| Triple: Mark → Matthew | Mk→Mt | ~280 | HIGH (scholarly consensus) |
| Triple: Mark → Luke | Mk→Lk | ~200 | HIGH (scholarly consensus) |
| OT: Chronicles → Sam/Kings | Chr→Sam | ~150 | HIGH (scholarly consensus) |
| Synthetic: Mark-edited | varies | ~500 | MEDIUM (synthetic) |
| **Total training** | | **~1,130** | |

**Negative examples:** Shuffle parallel pairs (same pericope, different gospels → independent). Random pairs from unrelated texts → independent.

**Architecture (detail):**

```python
class DirectionScorer(nn.Module):
    def __init__(self, encoder, hidden_dim=768):
        self.encoder = encoder  # Shared KoineFormer encoder
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.asymmetry_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 8, 512),  # 8 asymmetry features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # M→L, L→M, independent
        )

    def forward(self, text_a, text_b):
        h_a = self.encoder(text_a)  # (seq_len_a, hidden_dim)
        h_b = self.encoder(text_b)  # (seq_len_b, hidden_dim)

        # Bidirectional cross-attention
        attn_a_to_b, _ = self.cross_attn(h_a, h_b, h_b)
        attn_b_to_a, _ = self.cross_attn(h_b, h_a, h_a)

        # Asymmetry features
        features = self.compute_asymmetry(attn_a_to_b, attn_b_to_a)
        return self.asymmetry_mlp(features)
```

**Evaluation metrics:**
- Accuracy (direction classification)
- F1 per class
- Calibration error (ECE)
- Confusion matrix: A→B vs. independent
- OT transfer: train on Synoptic, test on Chronicles→Samuel/Kings

### 8.3 Baselines to Beat

| Method | Expected Accuracy | Architecture |
|--------|------------------|-------------|
| Trivial: "Matthew is longer" | ~70% | Length heuristic |
| Encoplot (Grozea 2010) | ~75% | N-gram dot-plot asymmetry |
| Ancient-Greek-BERT + cosine sim | ~65-70% | Embedding similarity |
| KoineFormer + direction scorer | **TARGET: >85%** | Cross-attention asymmetry |

---

## 9. Phase 4: Editorial Tendency Modeling

### 9.1 Learning Editorial Tendencies from Triple Tradition

**Mark → Matthew tendencies (learned):**
- Input: Mark's text of a triple-tradition pericope
- Output: Matthew's version
- Train a seq2seq model (KoineFormer encoder-decoder) on (Mark, Matthew) pairs

**Mark → Luke tendencies (learned):**
- Same architecture, (Mark, Luke) pairs

**Quantitative characterization:**
For each learned tendency, compute:
- **Vocabulary shift:** Which Markan words does Matthew/Luke systematically replace?
- **Syntax transformation:** Which syntactic structures are consistently modified?
- **Addition probability:** Where does Matthew/Luke insert material?
- **Omission probability:** What does Matthew/Luke drop?

**Compare to scholarship:**
- Do learned tendencies match the descriptions in Davies & Allison (Matthew) and Fitzmyer (Luke)?
- Are there learned tendencies that scholars haven't documented? (Discovery potential)

### 9.2 Editorial Fatigue as a Learnable Loss

**Formalization:**

For a pericope with N tokens, the fatigue loss is:

```
L_fatigue = (1/N) Σ_{i=1}^{N} w(i) · D_KL(edit_distribution_i || source_distribution_i)
```

where:
- `w(i) = exp(-λ · i/N)` — exponential decay weight (later tokens weighted less)
- `λ` controls the fatigue rate (learnable or set to 1.0)
- `D_KL` measures how much the edit at position i diverges from the expected source

**Ground truth examples (Goodacre's list):**

| Pericope | Gospel | Fatigue Pattern |
|----------|--------|-----------------|
| Death of John the Baptist | Matthew 14:1-12 | "Tetrarch" → "King" reversion |
| Parable of the Sower | Luke 8:4-15 | Omitted details reappear in interpretation |
| Healing of the Paralytic | Luke 5:17-26 | House setting omitted, still lowered "through tiles" |
| Feeding of the 5,000 | Luke 9:10-17 | Bethsaida = city, yet "desert place" |
| Parable of Pounds/Talents | Luke 19:11-27 | 10 servants → 3 appear; distribution incoherent |

**Training:**
- Binary classification: does this pericope contain detectable fatigue?
- Regression: at which token position does fatigue begin?
- Multi-task: fatigue detection + direction scoring (reinforcing signals)

### 9.3 OT Transfer Learning

**Task:** Train editorial tendency models on Chronicles → Samuel/Kings, test transfer to Synoptic Gospels.

**Why this works:** The Chronicler's use of Samuel/Kings is the best-understood example of editorial behavior in the Hebrew Bible (Smiley 2025 established transformer-based parallel detection). The editorial behaviors — abbreviation, theological correction, harmonization — are analogous to Gospel editing.

**Training:**
- Align Chronicles ↔ Samuel/Kings parallel passages (Smiley 2025's E5-based alignment)
- Train direction scorer on known direction (Chronicles used Samuel/Kings)
- Train editorial tendency model on (Samuel/Kings → Chronicles) pairs
- Apply both models to Synoptic data as a transfer test

**Success metric:** If a model trained solely on OT data correctly identifies Mark → Matthew/Luke direction in triple tradition, it demonstrates that editorial direction features are *genre-independent* — strengthening the generalizability of the approach.

---

## 10. Phase 5: Proto-Q Reconstruction

### 10.1 Training Strategy

**Step 1: Mark reconstruction (warm-up, triple tradition)**

Train the FiD architecture on triple tradition:
- Input: (Matthew's version of Mark, Luke's version of Mark)
- Target: Mark (the known source)
- Loss: Cross-entropy on token generation

This is our supervised training signal. The model learns to "factor out" Matthean and Lukan editorial changes to recover the common source.

**Validation:** Held-out triple-tradition pericopes. How well does the model reconstruct Mark from Matthew+Luke alone? Metrics: BLEU, ROUGE-L, chrF++, BERTScore (using KoineFormer as the embedding model).

**Step 2: Q reconstruction (double tradition)**

Transfer the trained FiD model to double tradition:
- Input: (Matthew's version of Q, Luke's version of Q)
- Target: IQP Critical Edition {A}-rated readings
- The model has learned editorial factoring from triple tradition; now it applies this to Q

**Step 3: Fine-tuning on IQP high-confidence data**

Use the IQP Critical Edition's {A} and {B} rated readings (~3,000-4,000 words at high confidence) as fine-tuning targets. The IQP's {C} and {D} rated readings are held out for evaluation.

### 10.2 Validation Strategy

**Primary: Mark reconstruction hold-out**
- Train on 80% of triple-tradition pericopes
- Evaluate on 20% held-out triple tradition
- Known ground truth (actual Mark text)
- This is the cleanest validation we have

**Secondary: IQP agreement**
- Compare reconstructed Q against IQP {A}-rated readings
- Agreement rate: % of words where reconstruction matches IQP
- Kappa: chance-corrected agreement with IQP editors

**Tertiary: Thomas triangulation**
- ~40 sayings with parallels in Gospel of Thomas
- Where Thomas, Matthew, and Luke agree, and Thomas is independent → triangulated Q wording
- Small sample but independent witness

**Quaternary: IQP {C}/{D} divergence analysis**
- Where our model disagrees with low-confidence IQP readings
- Are the disagreements systematic? (e.g., model consistently prefers Matthew, IQP prefers Luke)
- If systematic → evidence of bias in either model or IQP

### 10.3 Generation Quality Safeguards

**Vocabulary constraint:** Restrict decoder vocabulary to SBLGNT lexicon. Prevents hallucinated words.

**Minimum overlap constraint:** Generated text must have BLEU ≥ 0.3 against BOTH Matthew and Luke. Prevents the model from generating unrelated text.

**Consistency check:** Reconstructed Q should exhibit consistent vocabulary and style across pericopes. If Q sounds Matthean in one pericope and Lukan in the next → suspicious; flag for review.

**Lexical stability:** For words where Matthew and Luke agree verbatim, Q should preserve the agreed wording with high probability. This is the IQP's own principle and serves as a sanity check.

### 10.4 Contamination Awareness

**The circularity risk:** If we train on IQP data and then evaluate against IQP data, we're measuring agreement with IQP, not actual Q reconstruction quality.

**Mitigations:**
- Hold out IQP {A} pericopes entirely during training; train only on triple tradition + IQP {B} data
- The Mark reconstruction task provides a completely independent evaluation (IQP never touched Mark → Matthew/Luke relationships)
- Report IQP agreement alongside Mark reconstruction performance; the latter is the more trustworthy metric

---

## 11. Phase 6: Bayesian Model Comparison

### 11.1 Model Definitions

**H1: Two-Source Hypothesis (2SH)**
- Mark → Matthew (independently)
- Mark → Luke (independently)
- Matthew and Luke have NO direct dependency
- Q exists as a latent common source for double tradition

**H2: Farrer-Goulder Hypothesis (FGH)**
- Mark → Matthew
- Mark + Matthew → Luke
- No Q; Luke used Matthew directly

**H3: Augustinian Hypothesis**
- Matthew → Mark
- Matthew + Mark → Luke
- No Q; traditional patristic order

**H4: Griesbach Hypothesis (Two-Gospel)**
- Matthew → Luke
- Matthew + Luke → Mark
- No Q; Mark as conflation

### 11.2 Likelihood Specification

For each hypothesis H, we define a hierarchical likelihood over the observed direction scores y:

**Level 1 (token alignment):**
```
a_{ij} ~ Bernoulli(θ_p)  # For each aligned token pair i,j in pericope p
```
where θ_p is the probability that token i in gospel A aligns with (copies from) token j in gospel B for pericope p.

**Level 2 (pericope):**
```
y_p = (1/n_p) Σ a_{ij}  # Observed direction score for pericope p
y_p ~ Beta(α_h, β_h)    # Distribution under hypothesis h
```

**Level 3 (hypothesis-specific constraints):**
- 2SH: α_independent, β_independent (direction scores should be near 0.5 — ambiguous, no direction)
- FGH: α_{Mt→Lk} > β_{Mt→Lk} (direction scores should favor Matthew → Luke)
- Augustinian: α_{Mt→Mk} > β_{Mt→Mk} (direction scores should favor Matthew → Mark)
- Griesbach: α_{Mt→Lk} > β_{Mt→Lk} AND α_{conflate} > β_{conflate} (Mark shows conflation pattern)

**Level 4 (hyperpriors):**
```
α_h ~ Gamma(2, 0.5)  # Weakly informative
β_h ~ Gamma(2, 0.5)
```

### 11.3 Computation

**Option A: Bridge Sampling (recommended first pass)**
- Fit each model via PyMC (NUTS)
- Pass posterior samples to R `bridgesampling` via `reticulate` or write to file
- Returns log marginal likelihood
- BF_12 = exp(log_ml_1 - log_ml_2)

**Option B: Sequential Monte Carlo (PyMC)**
- `pm.sample_smc()` returns `log_marginal_likelihood` in `sample_stats`
- More computationally intensive but more robust for multimodal posteriors
- Good for models with < 20 parameters (our models have ~4-6)

**Option C: Thermodynamic Integration (gold standard)**
- 20-35 temperature rungs
- Most robust but most expensive
- Use if Options A and B give conflicting results

### 11.4 Prior Sensitivity Analysis

**Grid:** Vary prior hyperparameters across plausible ranges:
- α prior: Gamma(1, 0.5), Gamma(2, 0.5), Gamma(5, 0.5), Gamma(2, 1)
- β prior: same ranges

**Output:**
- Contour plot: BF_12 as a function of (α_prior, β_prior)
- Table: BF_12 under "scholarly informative prior" (elicited from 3-5 NT scholars)
- Table: BF_12 under "conservative prior" (wide, weakly informative)
- Statement: "The BF favors H1 by a factor of X under scholarly priors, and by a factor of Y under conservative priors."

### 11.5 Model Checking

**Posterior predictive checks:**
- Simulate direction scores from each fitted model
- Compare simulated distribution to observed distribution
- Bayesian p-value: ~0.5 = good fit; <0.05 or >0.95 = misspecification

**LOO-CV (complementary to Bayes factors):**
- Leave-one-pericope-out cross-validation
- `arviz.compare()` with stacking weights
- Answers: which model predicts held-out pericopes better?

**Convergence diagnostics:**
- R-hat < 1.01 for all parameters
- ESS > 400 for all parameters
- No divergent transitions
- Nested R-hat for hierarchical parameters

---

## 12. Phase 7: Interpretability & Robustness

### 12.1 SHAP Analysis

**Pipeline:**
1. For each pericope, compute SHAP values for the direction scorer's prediction
2. Aggregate by feature type: word-level, POS-level, syntax-level, discourse-level
3. Rank features by mean |SHAP|
4. Identify the 10-15 pericopes with the highest direction-discriminative SHAP signals

**Hawkins comparison:**
- Load Hawkins' 198 Matthean, 151 Markan, 413 Lukan characteristic features
- For each feature, check: is it also SHAP-important?
- Compute Jaccard similarity between SHAP-important features and Hawkins features
- Report convergent features (validating both methods) and divergent features (candidate discoveries)

**Expected output:** A table of the top 20 SHAP-important features with their Hawkins status, Greek gloss, and an example from a key pericope. This is the kind of output that a *New Testament Studies* reviewer will actually read and engage with.

### 12.2 BERTViz Attention Visualization

**Focus pericopes:** The 10-15 most SHAP-discriminative pericopes.

**Visualizations:**
- **Attention-head view:** Which heads attend to which tokens?
- **Neuron view:** Which neurons activate for cross-gospel parallels?
- **Layer view:** At which layers do direction-discriminative patterns emerge?

**Output:** Publication-ready figures showing attention patterns in, e.g., the Parable of the Pounds (Goodacre's strongest fatigue example), with annotations explaining what the model is attending to.

### 12.3 Multi-Edition Sensitivity Analysis

**Editions to test:**
1. NA28 (standard critical text)
2. Textus Receptus (underlying KJV, Byzantine text-type)
3. Majority Text (Hodges-Farstad or Robinson-Pierpont)
4. Westcott & Hort (1881 — the classic critical text that established Markan priority)

**Pipeline:** Re-run the full pipeline (direction scorer → Bayesian comparison → SHAP) on each edition. Do conclusions hold?

**Expected outcome (hypothesis):** The direction scorer should be robust to text-critical variation BECAUSE it's learning editorial patterns at the syntactic/discourse level, not at the individual word level. If it's NOT robust, that's an important finding — it means Synoptic conclusions are text-critical edition-dependent, which would be a significant methodological caution.

### 12.4 Counterfactual Analysis

**Method:** For a key pericope, systematically modify words/phrases identified as "editorial" by the Hawkins/SHAP analysis. Replace Matthean-specific vocabulary with Lukan equivalents (or vice versa). Does the direction scorer flip?

**Example:** In the Parable of the Pounds (Luke 19:11-27), replace Luke's "ten servants" with Matthew's "three servants" and the treasurer's "Well done, good servant" with Matthew's "Well done, good and faithful servant." If the direction scorer now says "Matthew → Luke" instead of "independent" (or vice versa), we've identified the specific features driving the model's decision.

---

## 13. Technical Stack

### 13.1 Core Libraries

| Component | Library | Version (as of June 2026) | Rationale |
|-----------|---------|--------------------------|-----------|
| Deep learning | PyTorch 2.x | ≥2.4 | Ecosystem for custom architectures |
| Transformers | HuggingFace `transformers` | ≥4.45 | Model hub, tokenizers, training loop |
| PEFT | `peft` | ≥0.12 | LoRA adapters |
| Bayesian | PyMC | ≥5.28 | Hierarchical models, NUTS, SMC |
| Bayesian diag | ArviZ | ≥0.20 | Diagnostics, model comparison, plotting |
| MCMC accel | NumPyro (JAX) | latest | GPU-accelerated MCMC when needed |
| Bridge sampling | `bridgesampling` (R) | latest | Marginal likelihood estimation |
| Interpretability | SHAP | ≥0.44 | Feature importance |
| Attention viz | BERTViz | ≥1.1 | Attention visualization |
| Experiment tracking | Weights & Biases | latest | Experiment logging, hyperparameter sweeps |
| Data processing | Pandas, NumPy | latest | Standard data wrangling |
| Text alignment | `biopython` (Needleman-Wunsch) | latest | Word-level alignment |
| Greek text | Text-Fabric | latest | N1904-TF data access |
| OT transfer | E5 model (intfloat/e5-base) + custom | latest | Chronicles alignment per Smiley 2025 |
| Laplace approx | `laplace-torch` | latest | Neural network marginal likelihood |

### 13.2 Hardware Requirements

| Task | GPU | RAM | Disk | Approx. Time |
|------|-----|-----|------|-------------|
| KoineFormer DAPT | 1× A100 (40GB) or 2× RTX 3090 | 64GB | 100GB | 24-48 hours |
| Multi-task fine-tuning | 1× RTX 3090 (24GB) | 32GB | 50GB | 6-12 hours |
| Direction scorer training | 1× RTX 3090 | 32GB | 50GB | 4-8 hours |
| FiD Q reconstruction | 1× A100 or 2× RTX 3090 | 64GB | 100GB | 12-24 hours |
| Bayesian MCMC | CPU (16+ cores) | 64GB | 20GB | 2-6 hours |
| SHAP computation | 1× RTX 3090 | 32GB | 20GB | 2-4 hours |
| Full pipeline end-to-end | — | — | ~300GB | 3-7 days |

**Cloud options:** Lambda Labs (A100 @ $1.10/hr), Vast.ai (RTX 3090 @ $0.30/hr), or institutional cluster.

### 13.3 Reproducibility

- All random seeds fixed and logged
- Environment: `conda env export` or `pip freeze` + Docker image
- Data preprocessing: versioned scripts, not manual steps
- Training: Weights & Biases logging of all hyperparameters and metrics
- Evaluation: held-out test set used exactly once; all metrics pre-registered
- Code: public GitHub repository with MIT license
- Data: CC-BY-compliant dataset release (SBLGNT + MorphGNT + derived features); no copyrighted NA28 text in public release

---

## 14. Risk Analysis & Mitigation

### 14.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|------------|
| KoineFormer DAPT fails to close domain gap (MLM perplexity >5% from from-scratch baseline) | MEDIUM | HIGH | Fall back to from-scratch T5 training on Koine only (~1M tokens, smaller model). More expensive but eliminates domain gap. |
| Direction scorer accuracy <80% (insufficient for Bayesian comparison) | MEDIUM | HIGH | Augment with synthetic editing pairs. Increase cross-attention depth. Try multi-scale features (character, subword, word, phrase levels). If still <80%, reframe paper as "direction features exist but are weak" finding. |
| Q reconstruction produces near-copies of Matthew or Luke (degenerate solution) | MEDIUM | MEDIUM | Add minimum divergence constraint. Multi-teacher distillation: enforce that Q must be distinguishable from BOTH inputs. |
| Bayes factors are prior-driven (conclusions flip under plausible alternative priors) | HIGH | MEDIUM | This is EXPECTED with n=3 gospels. Report honestly with BF curves and prior sensitivity. The contribution is the METHOD, not the conclusion. "Under a wide range of plausible priors, the data favor X" is a valid finding even if not definitive. |
| Catastrophic forgetting during DAPT | LOW-MEDIUM | MEDIUM | Gradual unfreezing, replay buffer with classical Greek examples, EWC (Elastic Weight Consolidation) regularization. |
| FiD overfits on triple tradition, fails to transfer to double tradition | MEDIUM | HIGH | Reduce model capacity; add domain-adversarial training to enforce shared representation; early stopping based on held-out triple-tradition validation. |

### 14.2 Scholarly Risks

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|------------|
| NT scholars dismiss ML approach as "black box" and ignore results | MEDIUM | HIGH | SHAP + Hawkins comparison makes model interpretable. BERTViz visualizations make attention patterns legible. Write paper with substantial humanities framing (not just equations). |
| IQP scholars object to using their reconstruction as training target | HIGH | LOW-MEDIUM | Frame Q reconstruction as *exploratory*, not definitive. Emphasize that IQP is training signal, not ground truth. Mark reconstruction task provides independent validation. |
| One camp seizes on favorable results to claim "vindication" | HIGH | MEDIUM | Bayesian prior sensitivity framing prevents cherry-picking. Report full BF range. Conclusion should be methodological: "here's what ML can tell us" not "X is right, Y is wrong." |
| Paper rejected from NLP venues for "not enough ML novelty" AND from humanities venues for "too technical" | MEDIUM | HIGH | Target dual audience: CHR (Computational Humanities Research) primary, ACL secondary, NTS (New Testament Studies) as outreach. Write different versions for different audiences. |

### 14.3 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|------------|
| SBLGNT/MorphGNT aligned dataset has tokenization mismatches | MEDIUM | MEDIUM | Manual spot-check of 100 random tokens. Automated consistency checks (verse length, token count). Document all remaining issues. |
| Aland pericope boundaries are disputed for some passages | LOW | LOW | Document disputed pericopes separately. Sensitivity analysis: exclude disputed pericopes; do conclusions change? |
| NA28 copyrighted text leaks into public dataset | LOW | HIGH | Use SBLGNT as primary text. NA28 used only for variant comparison in private analysis. Public dataset contains no DBG-copyrighted material. |

---

## 15. Publication Strategy

### 15.1 Primary Venue

**CHR 2027 (Computational Humanities Research)** — top venue for computational approaches to humanities questions. Robert-Hayek et al. (2023) published here. Strong fit: humanities question, ML methodology, interpretability focus.

**Backup:** ACL 2027 (if ML contributions are strong enough), *Digital Scholarship in the Humanities* (journal, if CHR timing doesn't work).

### 15.2 Paper Structure

1. **Introduction:** The Synoptic Problem as a test case for ML in textual criticism
2. **Related Work:** Synoptic scholarship (Streeter, Goodacre, Kloppenborg) + ML for ancient text (Singh, Riemenschneider, Robert-Hayek)
3. **KoineFormer:** Domain-adaptive language model for Hellenistic Greek
4. **Direction Scorer:** Cross-attention asymmetry for causal direction detection
5. **Editorial Fatigue:** Formalizing Goodacre's argument as a learnable loss
6. **Proto-Q Reconstruction:** FiD architecture + triple-to-double transfer
7. **Bayesian Comparison:** Model evidence for four hypotheses with prior sensitivity
8. **Interpretability:** SHAP + Hawkins comparison + BERTViz
9. **Limitations:** Sample size, prior sensitivity, IQP circularity
10. **Conclusion:** What ML adds to the Synoptic debate

### 15.3 Supplementary Materials

- Full dataset (CC-BY compliant)
- Trained KoineFormer model (HuggingFace)
- All code (GitHub)
- Interactive BERTViz visualizations for key pericopes
- Bayesian prior sensitivity contour plots
- Reconstructed Q text (exploratory, with caveats)

### 15.4 Outreach to Biblical Studies

- Submit a shortened, less-technical version to *New Testament Studies* or *Journal of Biblical Literature*
- Present at SBL Annual Meeting (Computational Linguistics and Biblical Studies section)
- Blog post / Twitter thread translating technical findings for NT scholars
- Direct engagement with Goodacre (Duke), Kloppenborg (Toronto), Garrow — the key scholars will need to engage with the results

---

## 16. Timeline & Milestones

### Phase 1: Foundation (Months 1-2)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Corpus assembly complete | SBLGNT + MorphGNT + Aland alignment dataset |
| 3-4 | Pericope alignment complete | Needleman-Wunsch token alignments for all pericopes |
| 5-6 | OT corpus (Chronicles ↔ Sam/Kings) aligned | OT transfer dataset ready |
| 7-8 | Data splits, augmentation, preprocessing | Train/val/test sets; augmentation pipeline |

**Go/No-go:** Pericope alignment quality spot-check passes (95%+ correct at token level).

### Phase 2: KoineFormer (Months 3-4)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 9-10 | DAPT complete | Koine-adapted T5 encoder-decoder |
| 11-12 | Go/No-go benchmark: MLM perplexity | Within 5% of from-scratch Koine baseline |
| 13-14 | Multi-task LoRA fine-tuning complete | POS + dependency + lemma + boundary detection heads |
| 15-16 | KoineFormer evaluation vs. baselines | Benchmark vs. Ancient-Greek-BERT, GreBERTa, Trankit |

**Go/No-go:** KoineFormer outperforms Ancient-Greek-BERT on at least 3/4 downstream tasks.

### Phase 3: Source Detection & Direction Scoring (Months 4-5)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 15-16 | Source detector trained | Double vs. triple tradition classification results |
| 17-18 | Direction scorer v1 trained | Cross-attention asymmetry model |
| 19-20 | Direction scorer evaluation | Accuracy, F1, calibration; Encoplot baseline comparison |
| 21 | OT transfer test | Does OT-trained model detect Synoptic direction? |

**Go/No-go:** Direction scorer accuracy >80% on held-out triple tradition.

### Phase 4: Editorial Tendency Modeling (Months 5-6)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 22-24 | Mark→Matthew and Mark→Luke editors trained | Learned editorial tendencies |
| 25 | Editorial fatigue loss implemented | Fatigue detection on Goodacre's examples |
| 26 | OT transfer: Sam/Kings→Chronicles editor trained | Cross-corpus editorial tendency comparison |

**Go/No-go:** Fatigue detection correctly identifies ≥4/6 of Goodacre's classic examples.

### Phase 5: Q Reconstruction (Months 6-8)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 27-30 | FiD Mark reconstruction trained | BLEU/ROUGE on held-out triple tradition |
| 31-32 | FiD transferred to double tradition | Preliminary Q reconstruction |
| 33-34 | Fine-tuning on IQP {A}/{B} data | Improved Q reconstruction |
| 34-35 | Validation: Thomas triangulation | Agreement with Thomas independent parallels |
| 36 | Q reconstruction quality assessment | Human evaluation by 2-3 NT scholars |

**Go/No-go:** Mark reconstruction BLEU >0.40 (substantially better than random/trivial baseline).

### Phase 6: Bayesian Model Comparison (Months 7-8)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 31-32 | Hierarchical model specification | PyMC model code |
| 33-34 | Prior elicitation | 3-5 NT scholars provide plausible prior ranges |
| 35-36 | Bridge sampling / SMC computation | Log marginal likelihoods for all 4 hypotheses |
| 37-38 | Prior sensitivity analysis | BF contour plots across prior grid |
| 39 | Model checking | PPC, LOO-CV, diagnostics |

**Go/No-go:** All R-hat < 1.01, ESS > 400. BF stable across computational methods (bridge sampling vs. SMC).

### Phase 7: Interpretability & Write-up (Months 8-10)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 37-38 | SHAP analysis complete | Top 20 features, Hawkins comparison |
| 39-40 | BERTViz visualizations | Publication-quality figures for key pericopes |
| 41-42 | Multi-edition sensitivity analysis | Results across NA28, TR, Majority Text |
| 43-44 | Full paper draft | Complete manuscript |
| 45-46 | Internal review and revision | Feedback from collaborators |
| 47-48 | Supplementary materials | Dataset, models, code, visualizations |

### Phase 8: Submission & Dissemination (Months 10-12)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 49-50 | Paper submission | CHR 2027 or ACL 2027 |
| 51-52 | Outreach materials | Blog post, SBL abstract, simplified explainer |

---

## 17. References

### Synoptic Problem

- Abakuks, A. (2006). "A statistical study of the triple-link model in the Synoptic Problem." *JRSS Series A*, 169(1):49-60.
- Abakuks, A. (2007). "A modification of Honoré's triple-link model in the Synoptic Problem." *JRSS Series A*, 170(3):841-850.
- Farrer, A. (1955). "On Dispensing with Q." In D.E. Nineham (ed.), *Studies in the Gospels*, 55-88.
- Goodacre, M. (1998). "Fatigue in the Synoptics." *NTS* 44:45-58.
- Goodacre, M. (2002). *The Case Against Q*. Trinity Press.
- Goulder, M. (1974). *Midrash and Lection in Matthew*. SPCK.
- Goulder, M. (1989). *Luke: A New Paradigm*. JSOT Press.
- Haegerland, T. (2019). "Editorial Fatigue and the Existence of Q." *NTS* 65:190-206.
- Honoré, A.M. (1968). "A statistical study of the Synoptic Problem." *Novum Testamentum* 10(2/3):95-147.
- Kloppenborg, J.S. (1987). *The Formation of Q*. Fortress.
- Robinson, J.M., Hoffmann, P., & Kloppenborg, J.S. (eds.) (2000). *The Critical Edition of Q*. Peeters/Fortress.
- Robert-Hayek, S., Istas, J., & Rey, F. (2023). "Unraveling the Synoptic puzzle: stylometric insights into Luke's potential use of Matthew." *CHR 2023*, CEUR Vol. 3558, pp. 799-813.
- Streeter, B.H. (1924). *The Four Gospels*. Macmillan.

### Machine Learning

- Izacard, G. & Grave, E. (2021). "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." EACL 2021. [FiD architecture]
- Singh, P., Rutten, G., & Lefever, E. (2021). "Ancient-Greek-BERT." LaTeCH-CLfL 2021.
- Riemenschneider, F. & Frank, A. (2023). "GreBERTa, GreTa, PhilBERTa, PhilTa." ACL 2023.
- Smiley, D.M. (2025). "Computational Detection of Intertextual Parallels in Biblical Hebrew." arXiv:2506.24117.
- Grozea, C. & Popescu, M. (2010). "Who's the Thief? Automatic Detection of the Direction of Plagiarism." CICLing 2010.
- Huertas-Tato et al. (2024). "STAR: Style Transformer for Authorship Representations." *Knowledge-Based Systems*, July 2024.
- Immer, A. et al. (2021). "Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning." ICML 2021.

### Bayesian Methods

- Kass, R.E. & Raftery, A.E. (1995). "Bayes Factors." *JASA* 90(430):773-795.
- Meng, X.L. & Wong, W.H. (1996). "Simulating ratios of normalizing constants via a simple identity." *Statistica Sinica* 6:831-860.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing* 27:1413-1432.
- McCollum, J. & Turnbull, C. (2024). "Using Bayesian Phylogenetics to Infer Manuscript Transmission History." *DSH* 39(1):258-279.

### Data Sources

- Holmes, M.W. (2010). *The Greek New Testament: SBL Edition*. CC-BY 4.0.
- Tauber, J.K. (2017). *MorphGNT: SBLGNT Edition*, v6.12. DOI: 10.5281/zenodo.376200.
- Haug, D.T.T. & Jøhndal, M.L. (2008). "Creating a Parallel Treebank of the Old Indo-European Bible Translations." LaTeCH 2008. [PROIEL]
- N1904-TF. CenterBLC. DOI: 10.5281/zenodo.13117911.

---

## APPENDIX A: Data Sources Quick Reference

| Source | URL | License |
|--------|-----|---------|
| SBLGNT | https://github.com/LogosBible/SBLGNT | CC-BY 4.0 |
| MorphGNT | https://github.com/morphgnt/sblgnt | CC-BY-SA 4.0 |
| SBLGNT-lowfat syntax | https://github.com/biblicalhumanities/greek-new-testament | CC-BY-SA 4.0 |
| N1904-TF | https://github.com/CenterBLC/N1904 | Open |
| Open Apostolic Fathers | https://github.com/jtauber/apostolic-fathers | CC-BY-SA 4.0 |
| PROIEL Treebank | https://github.com/proiel/proiel-treebank | CC BY-NC-SA 4.0 |
| First1KGreek | https://github.com/OpenGreekAndLatin/First1KGreek | CC-BY-SA 4.0 |
| ECM Mark | https://ntvmr.uni-muenster.de/nt-transcripts | Free online |
| Codex Sinaiticus XML | https://github.com/itsee-birmingham/codex-sinaiticus | CC BY-NC-SA |
| PACE Josephus | https://pace-ancient.mcmaster.ca | Free |
| API.Bible | https://scripture.api.bible | Free API |

## APPENDIX B: Key Pericopes for Validation

Goodacre's fatigue examples (Triple Tradition, Mark → Matthew/Luke):
1. Death of John the Baptist — Matt 14:1-12 // Mark 6:14-29
2. Cleansing of the Leper — Matt 8:1-4 // Mark 1:40-45
3. Jesus' Mother and Brothers — Matt 12:46-50 // Mark 3:31-35
4. Parable of the Sower — Luke 8:4-15 // Mark 4:1-20
5. Healing of the Paralytic — Luke 5:17-26 // Mark 2:1-12
6. Feeding of the 5,000 — Luke 9:10-17 // Mark 6:30-44

Goodacre's fatigue examples (Double Tradition, Luke → Matthew):
7. The Centurion's Servant — Luke 7:1-10 // Matt 8:5-13
8. Parable of the Pounds/Talents — Luke 19:11-27 // Matt 25:14-30 (STRONGEST)
9. The Mission Charge — Luke 9:1-6 // Matt 10:5-15

Key Minor Agreements (anti-2SH):
10. Mocking of Jesus — Mark 14:65 // Matt 26:67-68 // Luke 22:64

Great Omission (Luke omits Mark 6:45-8:26):
11. Walking on Water, Syro-Phoenician woman, Feeding of 4,000, etc.

---

*SynoptiQ is designed to be the most thorough computational treatment of the Synoptic Problem ever attempted. The plan above is ambitious but achievable within 12 months of focused work by 2-3 researchers. The key risk mitigations — Mark reconstruction as validation, prior sensitivity as framing, and Hawkins comparison as interpretability — ensure that even partial success yields publishable results.*
