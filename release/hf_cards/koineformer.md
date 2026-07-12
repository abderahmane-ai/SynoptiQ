---
license: cc-by-sa-4.0
language:
  - grc
library_name: peft
tags:
  - ancient-greek
  - koine-greek
  - new-testament
  - synoptic-problem
  - biblical-studies
  - digital-humanities
  - domain-adaptation
  - lora
  - peft
base_model: bowphs/GreTa
datasets:
  - ainouche-abderahmane/synoptiq-corpus
pipeline_tag: text-generation
model-index:
  - name: KoineFormer
    results:
      - task:
          type: token-classification
          name: POS tagging (linear probe)
        dataset:
          type: ainouche-abderahmane/synoptiq-corpus
          name: SynoptiQ Corpus (test)
          split: test
        metrics:
          - type: accuracy
            name: POS accuracy
            value: 0.9662
      - task:
          type: token-classification
          name: Lemmatization (linear probe)
        dataset:
          type: ainouche-abderahmane/synoptiq-corpus
          name: SynoptiQ Corpus (test)
          split: test
        metrics:
          - type: accuracy
            name: Lemma accuracy
            value: 0.8134
---

# KoineFormer

A domain-adapted T5 encoder-decoder for Koine Greek, produced by
training LoRA adapters on GreTa (a Classical Greek T5) with a
1.5M-token Koine corpus and a Classical Greek replay buffer.

## Overview

KoineFormer adapts [GreTa](https://huggingface.co/bowphs/GreTa)---a
T5-base model trained on Classical and Medieval Greek by Heidelberg
NLP---to **Koine Greek**, the Hellenistic dialect of the New
Testament, Septuagint, and Apostolic Fathers.

The adaptation uses **LoRA** (Low-Rank Adaptation) to train only
3.7M of the model's 220M parameters, producing a **14 MB adapter**
checkpoint. Training takes under one hour on a single GPU.

| Property | Value |
|----------|-------|
| Base model | [bowphs/GreTa](https://huggingface.co/bowphs/GreTa) (T5-base, 220M) |
| Adaptation | LoRA (r=16, α=32) |
| Trainable params | 3.7M (1.5%) |
| Training corpus | 1.5M Koine tokens + Classical replay |
| Training time | 58 minutes (NVIDIA A10G) |
| POS accuracy | 96.62% (linear probe) |
| Lemma accuracy | 81.34% (linear probe) |
| Adapter size | 14 MB |
| License | CC-BY-SA 4.0 |

## Performance

Linear probe evaluation on the SynoptiQ Corpus test set:

| Model | POS | Lemma | Params | Checkpoint |
|-------|-----|-------|--------|------------|
| GreTa (zero-shot) | 95.32% | 82.37% | 0 | 880 MB |
| Full fine-tune | 96.11% | — | 220M | 880 MB |
| **KoineFormer (LoRA)** | **96.62%** | 81.34% | **3.7M** | **14 MB** |

KoineFormer improves POS accuracy by 1.30 points (28% relative error
reduction) over zero-shot. Lemmatisation accuracy is comparable
(82.4% vs.\ 81.3%)---span-corruption DAPT improves syntactic
representations but does not expand vocabulary coverage. Full
fine-tune lemma results are pending.

## Intended Uses

- Part-of-speech tagging for Koine Greek texts (New Testament,
  Septuagint, Apostolic Fathers)
- Lemmatisation of Hellenistic Greek passages
- Feature extraction (encoder hidden states) for downstream tasks such
  as textual criticism, authorship attribution, and stylistic analysis
- Fine-tuning on task-specific Koine Greek datasets (e.g., dependency
  parsing, named entity recognition)
- Fill-in-the-blank text reconstruction for manuscript studies

## Limitations

- Trained on ~1.5M Koine tokens — small by modern LM standards; may
  not generalize to rare vocabulary or hapax legomena
- Span-corruption DAPT improves syntax (POS) but not lexical knowledge
  (lemmatisation is flat at 82.4%). Do not use as a lemmatiser without
  task-specific fine-tuning
- DAPT corpus covers New Testament and Apostolic Fathers only; the
  Septuagint (~500K additional Koine tokens) is not yet included
- Not evaluated on non-literary Koine (papyri, inscriptions, ostraca)
- Generative output is illustrative, not production-quality; the model
  was trained for representation learning, not text generation

## Usage

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(
    base, "ainouche-abderahmane/koineformer"
).merge_and_unload()  # bake LoRA into base weights
tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

# Fill-in-the-blank: comma marks the missing word
text = "Ἀρχὴ τοῦ, Ἰησοῦ Χριστοῦ υἱοῦ θεοῦ."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(
    **inputs, max_new_tokens=40, num_beams=5,
    no_repeat_ngram_size=3, repetition_penalty=2.0,
    early_stopping=True, pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# → πρώτην τοῦ εὐαγγελίου ἰησοῦ χριστοῦ υἱοῦ θεοῦ.
```

## Training Data

| Source | Tokens | Description |
|--------|--------|-------------|
| SBLGNT | ~773K | Full Greek New Testament (27 books) |
| Apostolic Fathers | ~732K | 1-2 Clement, Ignatius, Polycarp, Didache |

A Classical Greek replay buffer (First1KGreek: Homer, Plato, Xenophon)
was interleaved at 30% to prevent catastrophic forgetting.

## Training

- **Objective**: T5 span corruption (15% noise, 512-token packed sequences)
- **Optimizer**: AdamW (lr=1e-4, cosine to zero)
- **Precision**: FP16 (AMP)
- **GPU**: NVIDIA A10G (24 GB), 58 minutes
- **Reproducibility**: `python scripts/train_dapt.py --smoke-test`

Full pipeline at [SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ).

## Related

- [**Koine-T5**](https://huggingface.co/ainouche-abderahmane/koine-t5) — sibling model: a multitask
  seq2seq adapter (POS · lemma · infilling · synoptic transfer) on the same GreTa base. Try it in the
  [**live demo**](https://huggingface.co/spaces/ainouche-abderahmane/koine-t5-demo).
- [**SynoptiQ corpus**](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus) — the
  49,061-token aligned Mt/Mk/Lk dataset KoineFormer is evaluated on.
- [**SynoptiQ**](https://github.com/abderahmane-ai/SynoptiQ) — the parent project.

## Citation

```bibtex
@inproceedings{ainouche2026koineformer,
  title     = {KoineFormer: Domain-Adaptive Language Modeling for Koine Greek},
  author    = {Ainouche, Abderahmane},
  booktitle = {Proceedings of LaTeCH-CLfL},
  year      = {2026},
}
```

## Dataset

Evaluated on the [SynoptiQ Corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus).

## License

CC-BY-SA 4.0 (MorphGNT share-alike requirement).
