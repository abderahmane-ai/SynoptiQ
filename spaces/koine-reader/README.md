---
title: Koine Reader
emoji: 🏛️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: cc-by-sa-4.0
models:
  - ainouche-abderahmane/koine-t5
  - bowphs/GreTa
---

# 🏛️ Koine Reader

An interlinear **reading assistant for Ancient Greek**, part of the
[SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ) project. Two engines:

### Read (New Testament) — gold
Look up any GNT reference (`John 1:1`, `Mark 1:9-11`, `Matthew 5`) and get, for every
word: **lemma · full morphology · English gloss · Strong's number** — human-curated, so
there are no model errors and no wait. For each Gospel passage it also shows the
**synoptic parallels** (the parallel passages in the other Gospels), resolved through
Aland's *Synopsis Quattuor Evangeliorum* pericope table — a feature no general Greek
reader offers.

### Analyse any Koine — neural
Paste Koine Greek that *isn't* in the GNT — a papyrus, an apocryphal gospel, the
Apostolic Fathers, a sentence of your own — and
[**Koine-T5**](https://huggingface.co/ainouche-abderahmane/koine-t5) predicts part of
speech and lemma, glossed against the GNT lexicon. This is the wedge: existing tools
only work on pre-annotated canonical text; this analyses *un-annotated* Koine.

## How it works

The gold engine ships a compact ~2 MB index (`data/n1904_index.json.gz`) built from the
Text-Fabric data by `scripts/build_reader_index.py`; the reader logic lives in
[`synoptiq.reader`](https://github.com/abderahmane-ai/SynoptiQ/tree/main/synoptiq/reader).
The neural model loads lazily on first use, so the Space starts instantly and the Read
tab works even where torch is unavailable.

## Data, sources & licensing

- **Greek text & morphology:** Nestle 1904 GNT via the
  [MACULA Greek](https://github.com/Clear-Bible/macula-greek) / Nestle1904 Text-Fabric
  edition (Cantanhêde, Jurg, Roorda). The Nestle 1904 text is public domain.
- **English glosses (BGVB):** Bible OnLine Learner (Oliver Glanz et al.), redistributed
  via the MACULA / `biblicalhumanities/Nestle1904` datasets — attributed here.
- **Strong's numbers:** public domain.
- **Louw–Nida semantic domains are deliberately *not* surfaced** (UBS-copyrighted),
  even though they exist in the source data.
- **Pericope parallels:** Aland, *Synopsis Quattuor Evangeliorum* (the boundary table is
  the SynoptiQ project's own encoding).

Before making this Space public, confirm the redistribution terms of the MACULA / BOL
gloss data suit your intended license.

## Run locally

```bash
pip install gradio "synoptiq @ git+https://github.com/abderahmane-ai/SynoptiQ.git"
python scripts/build_reader_index.py          # writes data/n1904_index.json.gz
python spaces/koine-reader/app.py
```
