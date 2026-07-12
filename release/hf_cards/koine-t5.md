---
license: cc-by-nc-sa-4.0
language:
- grc
base_model: bowphs/GreTa
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- t5
- ancient-greek
- koine-greek
- pos-tagging
- lemmatization
- text-infilling
- new-testament
- classics
- digital-humanities
datasets:
- ainouche-abderahmane/synoptiq-corpus
- universal-dependencies/universal_dependencies
model-index:
- name: Koine-T5
  results:
  - task:
      type: text2text-generation
      name: POS tagging (seq2seq)
    dataset:
      type: universal_dependencies
      name: UD Ancient Greek PROIEL (dev)
      config: grc_proiel
      split: validation
    metrics:
    - type: accuracy
      name: POS token accuracy (pooled)
      value: 0.9169
    - type: accuracy
      name: POS token accuracy (NT Koine)
      value: 0.9655
    - type: accuracy
      name: POS token accuracy (Classical)
      value: 0.8768
    - type: exact_match
      name: POS sentence exact match (pooled)
      value: 0.686
---

<div align="center">

# Koine-T5

**One model, four tasks for Ancient Greek: POS tagging · lemmatization · text infilling · synoptic style transfer**

*A 104 MB LoRA adapter that turns [GreTa](https://huggingface.co/bowphs/GreTa) into a multitask Ancient Greek workhorse — 96.6% POS token accuracy on New Testament Koine.*

🎮 **[Try it live](https://huggingface.co/spaces/ainouche-abderahmane/koine-t5-demo)** · 📦 [SynoptiQ on GitHub](https://github.com/abderahmane-ai/SynoptiQ)

</div>

---

## What is this?

**Koine-T5** is a [LoRA](https://arxiv.org/abs/2106.09685) adapter (r=64, 27.1M trainable parameters — 12% of the base) for [bowphs/GreTa](https://huggingface.co/bowphs/GreTa), a T5-base model pre-trained on Ancient Greek. It was trained **jointly on four task pools sampled into every batch**, so a single checkpoint handles all four — no task-specific heads, no per-task fine-tunes:

| Task | Prefix | Input → Output |
|---|---|---|
| **POS tagging** | `pos: ` | Greek text → [MorphGNT](https://github.com/morphgnt) part-of-speech codes, one per word |
| **Lemmatization** | `lemma: ` | Greek text → dictionary form of every word |
| **Text infilling** | *(none)* | Greek text with `<extra_id_N>` masks → the masked spans (T5 span corruption) |
| **Synoptic transfer** | `synoptic mark_to_matt: ` / `synoptic mark_to_luke: ` | Markan text → Matthean / Lukan rendering *(experimental)* |

It was built as part of [SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ), a neural source-criticism framework for the Synoptic Gospels, but the POS / lemma / infilling tasks are general-purpose: they cover both **New Testament Koine** and **Classical Greek** (Herodotus), thanks to the [PROIEL treebank](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL).

## Results

Evaluated on the held-out **UD Ancient Greek PROIEL dev set** (greedy decoding, 250 NT + 250 Classical sentences; best checkpoint selected on pooled POS token accuracy):

| Eval subset | POS token accuracy | POS sentence exact match |
|---|---:|---:|
| **New Testament (Koine)** | **96.6%** | 85.2% |
| Classical (Herodotus) | 87.7% | 52.0% |
| Pooled | 91.7% | 68.6% |

For context: a dedicated linear probe on a DAPT'd GreTa encoder ([KoineFormer](https://huggingface.co/ainouche-abderahmane/koineformer)) reaches 96.62% POS accuracy on a comparable NT corpus — Koine-T5 matches that **via free seq2seq generation**, while also lemmatizing, infilling, and paraphrasing with the same weights.

## Quickstart

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
tokenizer.pad_token = "<pad>"   # GreTa ships <pad>=0 and </s>=1 but doesn't register them
tokenizer.eos_token = "</s>"
# Map the 100 T5 sentinels onto GreTa's pre-trained "ghost" slots (32003–32102).
# No embedding resize needed — the slots already exist in the checkpoint.
tokenizer.add_special_tokens(
    {"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]}
)

base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koine-t5")
model.eval()
```

> ⚠️ **Do not** call `tokenizer.add_special_tokens({"pad_token": "[PAD]"})` + `resize_token_embeddings` (a common GreTa recipe). It desyncs the pad id from T5's decoder-start id and collapses generation. Bind the existing `<pad>` / `</s>` as above.

### POS tagging

```python
inputs = tokenizer("pos: καὶ φωνὴ ἐγένετο ἐκ τῶν οὐρανῶν", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=256, num_beams=1, do_sample=False)
tags = tokenizer.decode(out[0], skip_special_tokens=True).upper()
print(tags)
# C- N- V- P- RA N-
# καί=conjunction, φωνή=noun, ἐγένετο=verb, ἐκ=preposition, τῶν=article, οὐρανῶν=noun ✓
```

The `.upper()` matters: GreTa's tokenizer lowercases all text, so the model emits lowercase tag codes. MorphGNT codes are case-unique, so upper-casing restores them losslessly.

<details>
<summary><b>MorphGNT tag codes</b> (click to expand)</summary>

| Code | Part of speech | Code | Part of speech |
|---|---|---|---|
| `A-` | adjective | `RA` | definite article |
| `C-` | conjunction | `RD` | demonstrative pronoun |
| `D-` | adverb | `RI` | interrogative/indefinite pronoun |
| `I-` | interjection | `RP` | personal pronoun |
| `N-` | noun | `RR` | relative pronoun |
| `P-` | preposition | `V-` | verb |
| `X-` | particle | | |

</details>

### Lemmatization

```python
inputs = tokenizer("lemma: καὶ φωνὴ ἐγένετο ἐκ τῶν οὐρανῶν", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=256, num_beams=1, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# καί φωνή γίγνομαι ἐκ ὁ οὐρανός
```

Lemma conventions follow the PROIEL treebank (the majority training source), e.g. Classical `γίγνομαι` rather than Koine `γίνομαι`. Output is lowercase (tokenizer property, see above).

### Text infilling (span corruption)

No prefix — pass the masked text directly, exactly as in T5 pre-training:

```python
text = "Ἀρχὴ τοῦ εὐαγγελίου Ἰησοῦ <extra_id_0> καθὼς γέγραπται ἐν τῷ <extra_id_1> τῷ προφήτῃ"
inputs = tokenizer(text, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=64, num_beams=4, early_stopping=True)
print(tokenizer.decode(out[0], skip_special_tokens=False))
# <extra_id_0> χριστοῦ <extra_id_1> νόμῳ
```

The model fills Mark 1:1–2's gaps with `χριστοῦ` (exact) and `νόμῳ` ("the law" — the gold is `Ἠσαΐᾳ`, but grammatically and idiomatically coherent). Another example: `καὶ <extra_id_0> ἐγένετο ἐκ τῶν <extra_id_1>` → `κραυγὴ` / `οὐρανῶν` ("a cry came from the heavens").

### Synoptic style transfer *(experimental)*

```python
mark = "καὶ πρωῒ ἔννυχα λίαν ἀναστὰς ἐξῆλθεν καὶ ἀπῆλθεν εἰς ἔρημον τόπον κἀκεῖ προσηύχετο."  # Mark 1:35
inputs = tokenizer("synoptic mark_to_luke: " + mark, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=256, num_beams=4,
                     repetition_penalty=1.25, no_repeat_ngram_size=3,
                     encoder_no_repeat_ngram_size=3, early_stopping=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# ἰδὼν δὲ ὁ ἰησοῦς τοὺς ὄχλους ἀνέβη εἰς τὸ ὄρος προσεύξασθαι καὶ ἐγένετο ἐν τῷ
# προσεύχεσθαι αὐτὸν καὶ αὐτὸς ἦν διανυκτερεύων ἐν τῷ ἱερῷ καὶ προσευχόμενος
```

This pool is tiny (155 curated Mark→Matthew / Mark→Luke parallel pericope pairs from the [SynoptiQ corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus)), so treat outputs as **stylistically evocative, not faithful translations**: the example above is fluent, recognizably Lukan-flavored Greek about withdrawing to pray (ἀνέβη εἰς τὸ ὄρος προσεύξασθαι, διανυκτερεύων — cf. Luke 6:12), but it is a free composition, not Luke's actual parallel.

## Training

**Data.** Three sources, four pools:

| Pool | Size | Sources |
|---|---|---|
| `pos` | ~15K sentences | UD Ancient Greek PROIEL (XPOS → MorphGNT mapping) + SynoptiQ corpus (MorphGNT) |
| `lemma` | ~15K sentences | same |
| `denoise` | raw Greek prose | same texts, corrupted online (fresh masks every batch, T5 §3.1: 15% noise density, mean span 3) |
| `synoptic` | 155 pairs | SynoptiQ corpus aligned pericopes (SBLGNT text) |

PROIEL contributes ~214K tokens spanning NT Koine (Gospels, Acts, Epistles, Revelation) **and** Classical Greek (Herodotus' *Histories*) — which is why the model tags Classical text at 87.7% despite the Koine focus. POS codes come from PROIEL's fine-grained **XPOS** column mapped to the 13-code MorphGNT tagset (not UPOS/FEATS, whose `PronType=Dem` on the article would mislabel it).

**Balanced sampling.** Every micro-batch draws each of its 4 slots from a task pool with weights `pos: 3/8, denoise: 3/8, lemma: 1/8, synoptic: 1/8` (with replacement per task). The 155-example synoptic pool is upsampled so it can never be starved out, and the massive PROIEL pools can never drown it — one adapter serves all four tasks without catastrophic forgetting.

**Procedure.** LoRA r=64, α=128, dropout 0.05 on all attention (`q,k,v,o`) and FFN (`wi,wo`) projections of encoder and decoder; base frozen in bfloat16. 30,000 micro-steps (batch 4 × grad-accum 8 = effective 32), AdamW lr 1e-4, weight decay 0.01, 5% warmup then cosine to zero, max sequence length 256, single A10G GPU on [Modal](https://modal.com). Checkpoint selection: pooled POS token accuracy on PROIEL dev every 1,000 steps; this repo ships the best checkpoint (step 28,000).

## Limitations

- **Lowercase output.** GreTa's tokenizer case-folds everything; the model cannot produce capital letters. Upper-case POS codes yourself (lossless); for lemmas/prose, expect lowercase.
- **256-token window.** Inputs beyond ~2–3 verses are truncated. Tag long passages sentence-by-sentence.
- **Classical Greek is the weaker domain** (87.7% vs 96.6% token accuracy) — the training mix, the tokenizer (1.38 subwords/word on Koine vs 1.95 on Classical), and the eval all favor Koine.
- **Lemma conventions are PROIEL's**, which can differ from NT dictionaries (γίγνομαι vs γίνομαι).
- **Synoptic transfer is exploratory.** 155 training pairs produce style pastiche, not reliable parallel reconstruction. Do not use it for text-critical claims.
- **No copying-direction inference.** This model does not (and, per the SynoptiQ project's negative result, cannot) determine the direction of literary dependence between gospel texts.

## License and attribution

Released under **CC BY-NC-SA 4.0** (NonCommercial, ShareAlike). The adapter was trained on the [UD Ancient Greek PROIEL treebank](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL) (CC BY-NC-SA 3.0), and this license mirrors that source's terms; the other sources are SBLGNT (CC BY) and MorphGNT (CC BY-SA), via the [SynoptiQ corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus) (CC BY-SA 4.0). The base model [bowphs/GreTa](https://huggingface.co/bowphs/GreTa) is Apache-2.0 and is **not** included in this repo — only the LoRA adapter is.

## Related work

- [**KoineFormer**](https://huggingface.co/ainouche-abderahmane/koineformer) — sibling adapter: GreTa DAPT'd on Koine prose (SBLGNT + Apostolic Fathers), for encoder/probing use (96.62% POS via linear probe, 14 MB).
- [**SynoptiQ corpus**](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus) — 49,061 morphologically annotated tokens, 170 pericopes, 235 alignments across Matthew, Mark, and Luke.
- [**SynoptiQ**](https://github.com/abderahmane-ai/SynoptiQ) — the parent project: neural source criticism of the Synoptic Gospels.

## Citation

```bibtex
@misc{ainouche2026koinet5,
  author = {Ainouche, Abderahmane},
  title  = {Koine-T5: a multitask LoRA adapter for Ancient Greek
            (POS tagging, lemmatization, infilling, synoptic style transfer)},
  year   = {2026},
  url    = {https://huggingface.co/ainouche-abderahmane/koine-t5},
  note   = {LoRA adapter for bowphs/GreTa, trained on UD Ancient Greek PROIEL
            and the SynoptiQ corpus}
}
```

Please also cite the base model ([Riemenschneider & Frank, 2023](https://aclanthology.org/2023.acl-long.846/)) and, if you use the POS/lemma tasks, the PROIEL treebank ([Haug & Jøhndal, 2008](https://www.hf.uio.no/ifikk/english/research/projects/proiel/)).
