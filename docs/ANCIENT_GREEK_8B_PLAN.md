# Ancient-Greek 8B LLM — Research & Plan (codename *Krikri-Koine*)

**Status: proposal / draft preregistration (2026-07-12).** Not yet started. This is the feasibility
scan + plan requested after Koine-T5-Hexapla was shelved. Every external fact is sourced at the bottom.

> **Naming note.** "Krikri-8B" is already taken — *Llama-Krikri-8B* is ILSP's **Modern**-Greek model
> and is the proposed **base**, not the target. Pick a distinct name for the derivative (see Risks).
> This doc uses *Krikri-Koine* as a placeholder codename.

---

## 1. TL;DR / recommendation

- **Base model: `ilsp/Llama-Krikri-8B-Base`.** It is Llama-3.1-8B already continued-pretrained on
  **56.7B Greek tokens** with a **Greek-extended tokenizer**, and it **already reads polytonic +
  some Ancient Greek**. A 2026 paper QLoRA-fine-tuned it on Ancient→Modern translation and found its
  **native tokenizer handles polytonic diacritics with no vocabulary expansion** (NMT baselines
  collapsed to 0.62 BLEU on polytonic; Krikri **zero-shot = 8.29 BLEU**). This removes the single
  hardest problem a raw Llama/Qwen base would create.
- **Strategy: QLoRA continued-pretraining (CPT), Unsloth recipe** — LoRA on **all linear layers
  incl. `embed_tokens` + `lm_head`**, **rank 256 + rsLoRA**, **decoupled LR** (embeddings 2–10× lower),
  **10–20 % Modern-Greek replay** to prevent forgetting — then a **multitask task-SFT** stage (port
  Koine-T5's recipe). Escalate to full-FT (DeepSpeed ZeRO-3) only if a key metric demands it.
- **This is cheap.** A CPT run is **~$30–60 on Modal** (A100-80GB, per-second billing); the whole
  project fits in **~$150–300** including experimentation.
- **This is the direct experiment the Hexapla negative result ordered.** Hexapla proved the
  *GreTa-220M backbone* was the ceiling for generation. Krikri-Koine swaps in an 8B backbone that
  already knows Greek — the clean test of "does capacity fix the discourse-level failures 220M could not?"
- **Pair it with the benchmark.** No unified Ancient-Greek *generative* benchmark exists. Build the
  eval first (low compute, own "first", and it's what makes any "best Ancient-Greek LLM" claim credible).

---

## 2. Base-model selection

| Base | Size | Greek ability | Polytonic / Ancient | License | Verdict |
|---|---|---|---|---|---|
| **`ilsp/Llama-Krikri-8B-Base`** | 8B | 56.7B-token Greek CPT, Greek-extended tokenizer | **native polytonic + some Ancient Greek** (zero-shot 8.29 BLEU AG→MG) | Llama 3.1 Community | **Chosen base** |
| Gemma-2-9B | 9B | best open translation/multilingual, but no Greek CPT | polytonic unverified (likely weak) | Gemma | **Fallback / comparison base** |
| Qwen2.5-7B | 7B | strong reasoning, **weaker** low-resource multilingual/translation | polytonic likely poor | Apache-2.0 | Cleanest license, weaker Greek |
| Llama-3.1-8B (raw) | 8B | weak Greek | poor (this is what Krikri fixed) | Llama 3.1 | No reason over Krikri |
| Meltemi-7B | 7B | first Greek LLM, superseded by Krikri | Mistral tokenizer, weak polytonic | Apache-2.0 | Don't use |

**Decision:** start on **Krikri-8B-Base**; keep **Gemma-2-9B** as a fallback base *and* a comparison
point (does Greek-specific CPT beat a strong generic multilingual model?). Instruct-vs-Base: start
from **Base** for clean CPT, then build our own Instruct via task-SFT.

---

## 3. Data plan (~50–80M open Ancient-Greek tokens)

TLG is the largest Ancient-Greek corpus but its license **forbids redistribution/derivatives** → unusable
for an open model. Everything below is open:

| Source | ~Size | Register | Have it? |
|---|---|---|---|
| LXX (Rahlfs, via our TF reader) | 623k words | Koine (translation) | ✅ (Hexapla work) |
| SBLGNT / NT | ~140k words | Koine | ✅ |
| Apostolic Fathers | ~330k words | Koine | ✅ |
| PROIEL treebank | ~214k tok | Koine NT + Classical | ✅ |
| First1KGreek | ~16M words | Classical | ✅ |
| **Perseus (Open Greek & Latin)** | ~10–30M words | Classical/Koine | ⬜ open on GitHub |
| **Patrologia Graeca (out-of-copyright)** | ~28.5M tok | **patristic Koine/Byzantine** | ⬜ great register match |
| **Opera Graeca Adnotata (OGA)** | 34M tok, annotated | Classical→Koine (−250 CE) | ⬜ ready-made, overlaps Perseus |
| Papyri (papyri.info / IDP) | optional | documentary Koine | ⬜ optional |

After dedup/decontamination: **~50–80M tokens** is realistic. Small for from-scratch, **ample for
specializing a base that already knows Greek**. Add **10–20 % Modern-Greek replay** (from Krikri's
own distribution) into every batch so Modern-Greek competence is not eroded.

Reuse `synoptiq/data/koine_corpus.py` (the TF reader + chunking + 8-gram decontamination) and the
existing 16.8M-word decontaminated artifact as the seed — the corpus pipeline is already built.

---

## 4. Training strategy

**Stage A — Continued pretraining (CPT), QLoRA via Unsloth.**
- LoRA on **all linear layers including `embed_tokens` and `lm_head`** (Unsloth's fix for "LoRA Learns
  Less and Forgets Less"; plain-attention LoRA underperforms for CPT).
- **rank 256, rsLoRA** scaling (`alpha/√rank`), dropout 0.05.
- **Decoupled LR**: base ~1–2e-4 cosine; **`embed_tokens`/`lm_head` at 2–10× lower** (Unsloth: training
  them at the same LR *hurts*).
- **Data**: ~60M Ancient-Greek tokens + 15 % Modern-Greek replay, **sequence-packed**, 2–3 epochs.
- **Efficiency**: 4-bit QLoRA, flash-attention-2, gradient checkpointing, bf16, seq-packing. Unsloth is
  **~2× faster / ~52 % less VRAM** vs HF+FA2 QLoRA on Llama-3-8B.

**Stage B — Task-SFT (multitask instruction).** Port the Koine-T5 recipe: balanced pools of `pos`,
`lemma`, `synoptic`/translation, `continuation`, plus AG→MG translation instructions. Produces the
Instruct variant. Optional **Stage C**: DPO/preference on generation quality.

**Ceiling option:** the AG→MG paper's *best* BLEU came from **full-FT with DeepSpeed ZeRO-3 on 4×A100**;
QLoRA was "competitive." Start QLoRA; only escalate to full-FT if a headline metric plateaus below target.

---

## 5. Efficiency & compute budget

| GPU (Modal) | $/hr | Fit for 8B QLoRA (rank 256 + embeds) |
|---|---|---|
| A100-40GB | ~$2.10 | Workable (tight with long ctx) |
| **A100-80GB** | ~$2.50 | **Comfortable — default** |
| H100 | ~$3.95 | ~2× faster, ~1.6× price → similar total, less wall-clock |

Rough estimate: ~60M tokens × ~3 epochs ≈ 180M tokens seen; at ~3–5k tok/s (Unsloth, A100-80GB,
packed) ≈ **8–16 h/run** → **~$30–60 per CPT run**. Budget **~$150–300** total incl. baselines, SFT,
and false starts. (Throughput is an estimate — measure it in Phase 2 and re-cost.)

---

## 6. Evaluation (reuse the Koine-T5 harness; this is also the standalone benchmark)

1. **POS + lemma** on PROIEL + Perseus UD (existing harness, XPOS→MorphGNT map) — vs GreTa/Koine-T5.
2. **Dependency parsing** (PROIEL UAS/LAS) — new axis.
3. **Perplexity** on held-out Ancient-Greek windows (never trained on).
4. **AG→MG translation** BLEU on the AG-MG benchmark — directly comparable to the paper's Krikri numbers.
5. **Discourse-generation probe** (`docs/gospel_of_the_savior.md`) — the through-line: does 8B fix the
   speaker/pericope-bleed + mode-collapse Hexapla could not?
6. **Morphological self-consistency** (existing metric).
7. **Modern-Greek regression gate** — a Modern-Greek subset (e.g. a Krikri/DemosQA slice) that must
   **not** drop, mirroring the Hexapla no-regression discipline.

**Baseline-first rule:** measure Krikri-Base **zero-shot** on all of the above *before* any training.
That baseline is exactly what the specialized model must beat, and reporting the gap honestly is itself
a contribution (even if it turns out modest).

---

## 7. Tokenizer

Krikri's Greek-extended tokenizer already handles polytonic without UNK-collapse. **Action:** measure
**fertility** (subwords/word) on our Ancient-Greek corpus; only add tokens if fertility is high. The
AG→MG result suggests **no vocabulary expansion is needed** — a major simplification vs a Llama/Qwen base.

---

## 8. Phased plan

- **Phase 0 — Data + baseline.** Assemble/decontaminate the ~60M-token corpus (Perseus + Patrologia
  Graeca + OGA + what we have); measure tokenizer fertility; run **Krikri-Base zero-shot** on the full
  eval battery. *Deliverable: corpus artifact + baseline numbers.*
- **Phase 1 — Benchmark.** Freeze the unified Ancient-Greek eval (the standalone "first" artifact).
- **Phase 2 — CPT.** Unsloth QLoRA CPT (§4). Gate: no Modern-Greek regression; improved AG perplexity +
  POS/lemma over base.
- **Phase 3 — Task-SFT.** Multitask instruction tuning → Instruct variant.
- **Phase 4 — Eval + honest write-up + release.** Full battery vs Krikri-Base, Koine-T5, GreTa; release
  model + benchmark on HF (as Koine-T5 was). Report where gains are modest.
- **Phase 5 — (optional) generation deep-dive / DPO.** Close the Hexapla loop on discourse failures.

---

## 9. Risks & mitigations

- **Marginal gain over base Krikri** (it already knows some Ancient Greek). → Baseline-first; frame the
  contribution as *quantify + specialize + benchmark + open release*, not "bigger is better."
- **Small corpus (50–80M tok).** → This is specialization on a Greek-native base, not language-teaching;
  set expectations accordingly; LoRA-CPT is "CPT-flavored SFT," which is the right tool here.
- **QLoRA vs full-FT SOTA.** → Start QLoRA; escalate to DeepSpeed full-FT only for a plateaued metric.
- **License.** Llama-3.1 Community License: derivatives allowed, must ship "Built with Llama" and (per
  some readings) a `Llama-`-prefixed name; the 700M-MAU clause is irrelevant here. **Verify before release.**
- **Competition (GFOSS "three-millennia" vision, Nov 2025).** No released model yet → not preempted;
  differentiate on **Koine focus + rigor + the benchmark**, and move with intent.
- **Naming collision** with ILSP Krikri. → Pick a distinct name (e.g. tie to SynoptiQ/Koine).

---

## 10. What transfers directly from SynoptiQ (little is wasted)

`koine_corpus.py` (LXX TF reader + chunking + decontamination) · the eval harness (POS/lemma/perplexity/
morph self-consistency + regression gates) · PROIEL integration + XPOS→MorphGNT map · the Modal GPU
pipeline · the discourse probe · and the 16.8M-word decontaminated corpus as the CPT seed.

---

## 11. Open decisions (need the user)

1. Start from **Base** (recommended) or **Instruct**?
2. **QLoRA-first** (recommended) or go straight to full-FT for max BLEU?
3. Scope: model **+ benchmark** (recommended) or model only?
4. Name for the derivative.

---

## Sources

- ILSP Krikri — base model card: <https://huggingface.co/ilsp/Llama-Krikri-8B-Base> ·
  announcement: <https://www.ilsp.gr/en/news/krikri/> · paper: <https://arxiv.org/abs/2505.13772>
- Ancient→Modern Greek MT (QLoRA/full-FT on Krikri; polytonic tokenizer finding):
  <https://arxiv.org/abs/2605.18504>
- Unsloth continued-pretraining recipe (all-linears + embed/lm_head, rank 256, rsLoRA, decoupled LR):
  <https://unsloth.ai/blog/contpretraining>
- CPT best-practice / replay / forgetting: <https://arxiv.org/html/2407.07263v1> ·
  <https://futureagi.com/blog/continued-llm-pretraining/> ·
  <https://www.spheron.network/blog/continuous-pretraining-llm-gpu-cloud-domain-adaptation/>
- Corpora: Opera Graeca Adnotata <https://arxiv.org/html/2404.00739> · Diorisis
  <https://brill.com/view/journals/rdj/3/1/article-p55_55.xml> · Open Greek & Latin
  <https://sites.tufts.edu/oglworkshop/about/>
- Base-model landscape (Gemma-2 vs Qwen2.5 for low-resource MT): Tower+ <https://arxiv.org/pdf/2506.17080>
- Eval: LLMs for Classical Philology (GreTa) <https://arxiv.org/pdf/2305.13698> · ML for Ancient
  Languages survey <https://ora.ox.ac.uk/objects/uuid:7c017a1d-d859-4a6d-abb6-f9151abc9636> · GLEM lemmatizer
- Modal GPU pricing: <https://modal.com/pricing>
- Competition (GFOSS three-millennia vision): <https://gfoss.eu/building-a-fully-open-greek-llm-a-three-millennia-language-model-powered-by-open-data-infrastructure/>
