---
title: Koine-T5 Demo
emoji: 🏛️
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
models:
  - ainouche-abderahmane/koine-t5
  - bowphs/GreTa
---

# Koine-T5 Demo

Interactive demo of [**Koine-T5**](https://huggingface.co/ainouche-abderahmane/koine-t5) — a multitask
LoRA adapter for Ancient Greek (POS tagging, lemmatization, text infilling, synoptic style transfer).

Part of the [SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ) project. Runs on CPU; first load
downloads GreTa (~220M) + the adapter (~104 MB), so the initial request takes a moment.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
