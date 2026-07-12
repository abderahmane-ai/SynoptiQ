<div align="center">

# SynoptiQ

**Neural source-criticism for the Synoptic Problem — and an honest account of its limits.**

Transformer models of Koine Greek applied to the Gospels of Matthew, Mark, and Luke:
a curated parallel corpus, two published Ancient-Greek models, an interactive generative demo, a preregistered
source-criticism study, and a clear line between what the texts *can* and *cannot* tell us.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Models on HF](https://img.shields.io/badge/🤗-models%20%26%20dataset-yellow.svg)](https://huggingface.co/ainouche-abderahmane)
[![Demo](https://img.shields.io/badge/🤗-Gradio%20Demo-orange.svg)](https://huggingface.co/spaces/ainouche-abderahmane/koine-t5-demo)
[![License](https://img.shields.io/badge/code-MIT-green.svg)](#license)

</div>

---

## What this is

The **Synoptic Problem** asks how Matthew, Mark, and Luke — which share large amounts of nearly
identical wording — are literarily related. SynoptiQ approaches it with modern NLP, but with an
unusual discipline: **it reports negative results as loudly as positive ones.**

The headline finding is a limit. Inferring the *direction* of literary copying (who copied whom)
from the texts alone is **not achievable** — it is isomorphic to distinguishing a lossy projection
from its inverse, and no amount of modelling recovers it without an external prior. That result is
documented, and the code that once attempted it was removed rather than left to mislead
(see [`docs/DIRECTION_NEGATIVE_RESULT.md`](docs/DIRECTION_NEGATIVE_RESULT.md)).

What the texts *can* support — high-quality Koine representation learning, a hypothesis-neutral
parallel corpus, and a calibrated test of dependence *beyond* Mark — is what this repository
delivers.

## Deliverables

| Component | What it is | Result | Where |
|-----------|-----------|--------|-------|
| **SynoptiQ corpus** | Aligned Mt/Mk/Lk parallel corpus | 49,061 tokens · 170 pericopes · 235 alignments | [🤗 dataset](https://huggingface.co/datasets/ainouche-abderahmane/synoptiq-corpus) |
| **KoineFormer** | GreTa DAPT'd to Koine (LoRA) | **96.62%** POS · 81.34% lemma · 14 MB adapter | [🤗 model](https://huggingface.co/ainouche-abderahmane/koineformer) |
| **Koine-T5** | Multitask Ancient-Greek seq2seq | **96.6%** NT / 91.7% pooled POS | [🤗 model](https://huggingface.co/ainouche-abderahmane/koine-t5) |
| **Interactive Demo** | Live inference playground | Web UI for POS, lemma, infilling, translation | [🤗 space](https://huggingface.co/spaces/ainouche-abderahmane/koine-t5-demo) |
| **Source-criticism study** | Preregistered 2SH-vs-Farrer test | E2 = **symmetric null** · Track A bounded-negative | [`docs/`](docs/SOURCE_CRITICISM_STUDY.md) |
| **Copying direction** | Direction-of-copying detection | **closed negative result** | [`docs/`](docs/DIRECTION_NEGATIVE_RESULT.md) |
| **Future: Krikri-Koine** | 8B Generative LLM Proposal | Feasibility scan & implementation plan | [`docs/`](docs/ANCIENT_GREEK_8B_PLAN.md) |

## Quickstart

```bash
# Install (Python 3.12+)
pip install -e ".[dev]"

# Regenerate the corpus → data/processed/{tokens,pericopes}.parquet
python scripts/prepare_data.py --validate

# Fetch the KoineFormer DAPT adapter
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ainouche-abderahmane/koineformer', local_dir='models/koineformer/dapt/final')"

# Verify everything is wired up
python -m pytest tests/ -q
```

Use a published model directly:

```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

base = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa")
model = PeftModel.from_pretrained(base, "ainouche-abderahmane/koineformer").merge_and_unload()
# 96.62% POS · 81.34% lemma · 14 MB adapter
```

## Repository layout

```
synoptiq/          Python package (corpus, models, training, evaluation, interpretability)
  data/            corpus loading, alignment, splits, pericopes
  models/          KoineFormer, Koine-T5 multitask encoder, FiD
  training/        DAPT + multitask; frozen config dataclasses
  evaluation/      bootstrap CIs, NLL scoring, verdict core, reconstruction F1
scripts/           CLI entry points (prepare_data, study, etc.)
modal/             Modal GPU apps
tests/             pytest suite (mirrors synoptiq/) — 194 passing
spaces/            Hugging Face Spaces demo apps (koine-t5-demo)
release/           Model/Dataset cards and release metadata
docs/              negative-result note · source-criticism prereg · 8B future plan
```

## The closed negative result

Phases 3 (a per-pair direction scorer) and 6 (Bayesian hypothesis comparison) were investigated at
length and **removed** as a closed negative result. Copying-direction detection is not recoverable
from the texts alone. Please read
[`docs/DIRECTION_NEGATIVE_RESULT.md`](docs/DIRECTION_NEGATIVE_RESULT.md) before proposing anything
about copying direction, editorial fatigue as a direction signal, or scoring the source hypotheses —
this is a deliberate scientific stance, not an unfinished feature.

## Future Roadmaps

We have shelved `Koine-T5-Hexapla` as an experiment that hit the generative ceiling of the 220M-parameter GreTa backbone. Instead, we have drafted a detailed implementation plan for **Krikri-Koine**, an 8B continued-pretraining (CPT) pipeline that will bootstrap off the robust modern-Greek knowledge of `Llama-Krikri-8B-Base`. You can read the proposal in [`docs/ANCIENT_GREEK_8B_PLAN.md`](docs/ANCIENT_GREEK_8B_PLAN.md).

## Data & licenses

Built from open Ancient-Greek resources: SBLGNT (CC-BY), MorphGNT (CC-BY-SA), Apostolic Fathers,
First1KGreek / Open Greek & Latin (CC-BY-SA), UD_Ancient_Greek-PROIEL (CC BY-NC-SA), and the
Rahlfs-1935 Septuagint. Model licenses follow their training data — Koine-T5 is CC BY-NC-SA 4.0 (PROIEL NonCommercial); KoineFormer and the corpus are
CC-BY-SA 4.0.

## License

Code is released under the MIT License. Trained models and the dataset carry the Creative Commons
licenses noted above.

## Citation

```bibtex
@software{synoptiq,
  author = {Ainouche, Abderahmane},
  title  = {SynoptiQ: Neural Source-Criticism for the Synoptic Problem},
  url    = {https://github.com/abderahmane-ai/SynoptiQ},
  year   = {2026}
}
```
