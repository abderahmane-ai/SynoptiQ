<div align="center">

# SynoptiQ
**Neural Source-Criticism for the Synoptic Problem**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Models_%26_Datasets-yellow.svg)](https://huggingface.co/ainouche-abderahmane)
[![Gradio Demo](https://img.shields.io/badge/🤗-Interactive_Demo-orange.svg)](https://huggingface.co/spaces/ainouche-abderahmane/koine-t5-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

SynoptiQ is a suite of transformer models, datasets, and analytical tools designed to study the Greek texts of Matthew, Mark, and Luke. It provides state-of-the-art NLP infrastructure for Koine Greek, while delivering a rigorous, computationally-backed negative result on the limits of inferring textual dependence.

</div>

---

## 🏛️ Releases & Artifacts

SynoptiQ provides foundational tools for computational classics and digital humanities, fully open-sourced on Hugging Face.

| Asset | Description | Links |
| :--- | :--- | :--- |
| **Koine-T5** | A multitask seq2seq LoRA adapter handling POS tagging (96.6% accuracy), lemmatization, text infilling, and synoptic style transfer. | [🤗 Model](https://huggingface.co/ainouche-abderahmane/koine-t5) · [🎮 Live Demo](https://huggingface.co/spaces/ainouche-abderahmane/koine-t5-demo) |
| **KoineFormer** | An encoder-only domain adaptation (DAPT) of GreTa, achieving 96.62% POS tagging via linear probing. Lightweight 14MB adapter. | [🤗 Model](https://huggingface.co/ainouche-abderahmane/koineformer) |
| **SynoptiQ Corpus** | A token-level dataset of the Synoptic Gospels with morphological annotations, Aland pericope boundaries, and Needleman-Wunsch alignments. | [🤗 Dataset](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus) |

---

## 🚀 Usage

You can load and run **Koine-T5** using standard Hugging Face `peft` and `transformers` libraries to perform advanced Ancient Greek NLP tasks out of the box.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# 1. Load the base Classical Greek model and the Koine-T5 adapter
tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"

base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koine-t5")
model.eval()

# 2. Perform POS Tagging on Koine Greek
text = "pos: καὶ φωνὴ ἐγένετο ἐκ τῶν οὐρανῶν"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, num_beams=1)

tags = tokenizer.decode(out[0], skip_special_tokens=True).upper()
print(tags) 
# Output: C- N- V- P- RA N-
```

---

## 🔬 The Science: A Closed Negative Result

The central question of the Synoptic Problem is identifying the *direction* of literary copying (e.g., did Luke copy Mark, or did Mark copy Luke?). 

SynoptiQ was built to test if modern neural models could detect this directionality from textual features alone. After extensive modeling, Phase 3 and Phase 6 of this project were concluded with a **firm negative result**: the direction of copying cannot be recovered purely from text. It is isomorphic to distinguishing a lossy projection from its inverse. 

We believe negative results are critical to scientific progress. The code attempting this has been removed to prevent misuse, and the methodology is documented transparently in [`DIRECTION_NEGATIVE_RESULT.md`](docs/DIRECTION_NEGATIVE_RESULT.md).

---

## 🔮 Roadmap: Krikri-Koine (8B)

Following the findings from our T5 models, we have established that 220M parameter encoder-decoders have a strict ceiling for fluent Koine Greek generative discourse.

Our next phase is **Krikri-Koine**, an 8B continued-pretraining (CPT) pipeline that will bootstrap off `Llama-Krikri-8B-Base`. By leveraging a model that already possesses deep knowledge of Modern Greek and polytonic character handling, we aim to build the world's first highly-capable generative LLM dedicated to Ancient Greek.

Read the full preregistered feasibility scan and implementation plan: [`ANCIENT_GREEK_8B_PLAN.md`](docs/ANCIENT_GREEK_8B_PLAN.md).

---

## 📚 Repository Architecture

- `synoptiq/` — Core Python package containing corpus parsing, dataset alignment algorithms, and evaluation metrics.
- `models/` — Training configurations for KoineFormer and Koine-T5.
- `spaces/` — Source code for the interactive Hugging Face Gradio applications.
- `docs/` — Scientific preregistrations, negative result documentations, and future architectural plans.

---

## 📄 License & Citation

The code is released under the **MIT License**. The trained models and datasets carry **CC BY-NC-SA 4.0** and **CC-BY-SA 4.0** licenses, inheriting from the open Ancient-Greek resources they were built upon (SBLGNT, MorphGNT, PROIEL).

If you use SynoptiQ in your research, please cite:

```bibtex
@software{synoptiq,
  author = {Ainouche, Abderahmane},
  title  = {SynoptiQ: Neural Source-Criticism for the Synoptic Problem},
  url    = {https://github.com/abderahmane-ai/SynoptiQ},
  year   = {2026}
}
```
