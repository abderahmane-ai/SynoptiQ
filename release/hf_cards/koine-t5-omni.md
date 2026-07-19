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
- morphological-analysis
- text-restoration
- text-infilling
- new-testament
- classics
- digital-humanities
datasets:
- ainouche-abderahmane/synoptiq-corpus
- universal-dependencies/universal_dependencies
model-index:
- name: Koine-T5-Omni
  results:
  - task:
      type: text2text-generation
      name: POS tagging (seq2seq)
    dataset:
      type: universal_dependencies
      name: UD Ancient Greek PROIEL (test)
      config: grc_proiel
      split: test
    metrics:
    - type: accuracy
      name: POS token accuracy (pooled, convention-neutral)
      value: 0.9419
    - type: accuracy
      name: POS token accuracy (NT Koine, convention-neutral)
      value: 0.9583
    - type: accuracy
      name: POS token accuracy (Classical, convention-neutral)
      value: 0.9176
    - type: exact_match
      name: POS sentence exact match (pooled, convention-neutral)
      value: 0.7450
---

# Koine-T5-Omni

A multitask Ancient Greek sequence-to-sequence model: a LoRA adapter over
[`bowphs/GreTa`](https://huggingface.co/bowphs/GreTa) trained jointly on part-of-speech tagging,
lemmatisation, full morphological parsing, diacritic restoration, Synoptic style transfer, and
span infilling.

It is the sibling of [`ainouche-abderahmane/koine-t5`](https://huggingface.co/ainouche-abderahmane/koine-t5),
which it improves on across every POS metric while carrying four additional tasks on the same
backbone and the same adapter rank. Koine-T5 remains a valid model and its published numbers
stand; this is not a deprecation.

Trained as part of [SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ).

## What it does

| Task | Prefix | Input → Output | Status |
|---|---|---|---|
| POS tagging | `pos: ` | Greek → MorphGNT 2-char codes | Works |
| Lemmatisation | `lemma: ` | Greek → lemma sequence | Works |
| Morphology | `morphology: ` | Greek → compact parse codes | Works |
| Diacritic restoration | `restore: ` | Unaccented/uncial → polytonic | Works |
| Synoptic transfer | `synoptic mark_to_matt: ` / `synoptic mark_to_luke: ` | Mark → Matthew/Luke register | Works |
| Span infilling | *(no prefix)* | Text with `<extra_id_N>` → fills | Works |
| Normalisation | `normalize: ` | — | **Does not work — see Known limitations** |
| Glossing | `gloss: ` | — | **Does not work — see Known limitations** |

## Results

Evaluated on the **UD Ancient Greek PROIEL test split**, held out from training and from
checkpoint selection (selection used the dev split).

### Against Koine-T5

Both models scored with identical code, split, decoding and tokenizer, restricted to tokens whose
gold label does not depend on tagset convention (94.4% of the test set — see
[Comparison methodology](#comparison-methodology)).

| Model | NT tok | NT EM | Classical tok | Classical EM | Pooled tok | Pooled EM |
|---|---|---|---|---|---|---|
| **Koine-T5-Omni** | **0.9583** | **0.8418** | **0.9176** | **0.5842** | **0.9419** | **0.7450** |
| Koine-T5 | 0.9351 | 0.8018 | 0.8718 | 0.5536 | 0.9097 | 0.7085 |
| Δ | +2.3 pp | +4.0 pp | +4.6 pp | +3.1 pp | +3.2 pp | +3.6 pp |

### Per-task (PROIEL dev / held-out slices)

| Task | Token accuracy | Exact match |
|---|---|---|
| POS (NT) | 0.9848 | — |
| POS (Classical) | 0.9189 | — |
| POS (pooled) | 0.9487 | 0.7000 |
| Morphology | 0.8850 | 0.2100 |
| Lemma | 0.8817 | 0.3750 |
| Restore | 0.8255 | 0.3000 |

## Usage

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"
tokenizer.add_special_tokens(
    {"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]}
)

base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
base.config.tie_word_embeddings = False
base.config.vocab_size = len(tokenizer)
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koine-t5-omni")

inputs = tokenizer("pos: μεγαλύνει ἡ ψυχή μου τὸν κύριον", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True).upper())
# V- RA N- RP RA N-
```

Three things matter for correct output:

- **Do not add a `[PAD]` token or resize embeddings.** GreTa ships `<pad>`=0 and `</s>`=1. Adding
  a pad token desynchronises the pad id from the decoder start id and collapses generation.
- **Upper-case POS and morphology predictions.** The GreTa tokenizer case-folds, so the model
  emits lowercase tags. MorphGNT codes are case-unique, making this lossless.
- **Use greedy decoding** (`num_beams=1, do_sample=False`) for the tagging and restoration tasks.
  Repetition penalties corrupt them: tags legitimately repeat, and restoration output largely
  repeats its input. Beam search suits the Synoptic task; contrastive search suits infilling only.

### Decoding morphology output

Morphology is emitted in a compact encoding that omits MorphGNT's positional padding
(`N-----NSF-` → `N-NSF`). This keeps targets inside the 256-token context. The mapping is
injective over all 602 tags attested in MorphGNT, so it is lossless; **`morph_tags.json`** in this
repository inverts it.

```python
import json
table = json.load(open("morph_tags.json"))
table["N-NSF"]    # "N-----NSF-"
```

## Training

| | |
|---|---|
| Base model | `bowphs/GreTa` (T5-base, 220M) |
| Adapter | LoRA r=64, α=128, dropout 0.05 |
| Target modules | `q`, `k`, `v`, `o`, `wi`, `wi_0`, `wi_1`, `wo` |
| Context | 256 tokens |
| Optimiser | AdamW, lr 1e-4, cosine decay, 3,750 warmup steps |
| Batch | 4 × 8 gradient accumulation = 32 |
| Steps | 75,000 (best checkpoint at 64,000) |
| Precision | bfloat16 |
| Hardware | 1× NVIDIA A10G |

Tasks are sampled per micro-batch slot in proportion to fixed weights, so every task appears in
most optimiser steps and none is starved. Checkpoint selection is gated: a checkpoint must hold
POS-NT and lemma above threshold before its secondary-task score can win, so selection optimises
the model as a whole rather than the tagger alone. The best checkpoint occurred at step 64,000 of
75,000 — the model plateaued rather than being cut off.

### Data

| Source | License | Used for |
|---|---|---|
| [UD Ancient Greek PROIEL](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL) | CC BY-NC-SA 3.0 | POS, lemma, infilling |
| [MorphGNT / SBLGNT](https://github.com/morphgnt/sblgnt) | CC BY-SA 3.0 | Morphology, POS, lemma, infilling |
| [MACULA Greek](https://github.com/Clear-Bible/macula-greek) | CC BY-SA 4.0 | Glossing (task did not succeed) |
| [SynoptiQ corpus](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus) | CC BY-SA 4.0 | Synoptic transfer, POS, lemma |

PROIEL's NonCommercial term propagates to this adapter, hence CC BY-NC-SA 4.0.

## Known limitations

Both prefixes below exist in the checkpoint and produce output. Neither should be relied on.

**`normalize` does not work.** It scores 0.886 token accuracy on its held-out slice, but emitting
the input unchanged scores approximately 0.90 on the same data: the model learned an identity
mapping rather than the task. Crasis expansion fails on every case tried.

```
normalize: κἀγὼ εἶπον αὐτῷ        →  κἀγὼ εἶπον αὐτῷ        (expected καὶ ἐγὼ …)
normalize: κἂν ἀποθάνῃ ζήσεται    →  κἂν ἀποθάνῃ ζήσεται    (expected καὶ ἐάν …)
```

The cause is training corruption too sparse to make copying suboptimal — most tokens in a
normalisation example are already correct, so the identity mapping is near-optimal under
token accuracy. Any evaluation of a task of this shape needs a copy baseline reported alongside it.

**`gloss` does not work.** 0.130 token accuracy. Output is English-shaped but not correct:

```
gloss: ἐν ἀρχῇ ἦν ὁ λόγος  →  in the_one having_been_to_the_one the should
```

GreTa has no English pretraining, and word-level glossing over a largely single-occurrence
vocabulary is not reachable from this backbone at this data scale. A Greek→English verse
translation task was attempted earlier and abandoned for the same reason.

**Classical Greek is weaker than Koine.** Trained predominantly on New Testament Greek; Herodotus
runs roughly 4 pp behind on token accuracy and substantially behind on exact match.

**Lemma follows PROIEL/MorphGNT conventions**, which differ from some lexica — for example first
person plural pronouns lemmatise to `ἐγώ` rather than `ἡμεῖς`.

**Morphology exact match is 0.21** at the sentence level while token accuracy is 0.885: long
sentences reliably contain at least one of ten joint grammatical axes that is wrong.

## Comparison methodology

The headline comparison masks tokens whose gold label depends on tagset convention. This is not
cosmetic, and reproducing it naively will give different numbers.

PROIEL and MorphGNT disagree on high-frequency postpositive particles: MorphGNT treats δέ, γάρ,
οὖν, μέν as conjunctions, PROIEL as adverbs. Koine-T5-Omni was trained on gold reconciled to the
MorphGNT convention; Koine-T5 predates that reconciliation. Scoring both against reconciled gold
therefore marks Koine-T5 wrong for answering as it was taught.

The affected tokens are only 5.6% of the test set, but they occur in **39.3% of NT and 85.5% of
Classical sentences**, so full-sentence exact match is affected far more than token accuracy.
Unmasked, the pooled gap reads +8.6 pp token / +41.4 pp exact match. Masked, it is +3.2 / +3.6.

The diagnostic is that masking barely moves this model (0.9442 → 0.9419 pooled token accuracy)
while it moves Koine-T5 substantially (0.8585 → 0.9097). The masked figures are reported here
because they measure tagging skill rather than answer-key alignment.

## Reproducing

The checkpoint was trained from tag
[`omni-v1-training`](https://github.com/abderahmane-ai/SynoptiQ/releases/tag/omni-v1-training)
(commit `45958b6`). Later commits on `main` remove the glossing task and rebuild normalisation, so
`main` does not reproduce this checkpoint.

```bash
modal run modal/app_koine_t5_omni.py::train
modal run modal/app_koine_t5_omni.py::run_test     # held-out test numbers
modal run modal/app_koine_t5_omni.py::compare      # convention-neutral comparison
```

## Citation

```bibtex
@software{ainouche_koine_t5_omni_2026,
  author = {Ainouche, Abderahmane},
  title  = {Koine-T5-Omni: A Multitask Ancient Greek Sequence-to-Sequence Model},
  year   = {2026},
  url    = {https://huggingface.co/ainouche-abderahmane/koine-t5-omni},
  note   = {LoRA adapter over bowphs/GreTa}
}
```

## Acknowledgements

Built on [GreTa](https://huggingface.co/bowphs/GreTa) (Riemenschneider & Frank). Training data from
the PROIEL treebank, MorphGNT/SBLGNT, MACULA Greek, and the SynoptiQ corpus.
