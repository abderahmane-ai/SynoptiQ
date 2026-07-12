# Release artifacts & publish guide

Everything here is **built and validated** — the steps below are the parts that need *your*
Hugging Face / Zenodo / GitHub accounts. Nothing publishes automatically.

| Artifact | Source in repo | Publishes to |
|---|---|---|
| Live demo | [`spaces/koine-t5-demo/`](../spaces/koine-t5-demo/) | HF Space `ainouche-abderahmane/koine-t5-demo` |
| Citable DOI | [`CITATION.cff`](../CITATION.cff), [`.zenodo.json`](../.zenodo.json) | Zenodo (via a GitHub Release) |
| Polished cards | [`release/hf_cards/`](hf_cards/) | the three HF model/dataset repos |

---

## 1. Gradio demo Space

Files: `spaces/koine-t5-demo/{app.py, requirements.txt, README.md}`. Runs on the **free CPU tier**
(first request is slow — it downloads GreTa + the adapter, then caches).

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create koine-t5-demo --type space --space_sdk gradio

git clone https://huggingface.co/spaces/ainouche-abderahmane/koine-t5-demo
cp spaces/koine-t5-demo/{app.py,requirements.txt,README.md} koine-t5-demo/
cd koine-t5-demo && git add . && git commit -m "Koine-T5 interactive demo" && git push
```

The Space name **`koine-t5-demo`** matches the "Try it live" links already in the three cards. If HF
rejects `sdk_version: 4.44.1` (README front-matter), bump it to a version the platform offers — the
app uses only stable `gr.Blocks` components, so any Gradio 4.x/5.x works.

## 2. Zenodo DOI (makes the whole project citable)

1. Sign in to <https://zenodo.org> **with GitHub**.
2. Zenodo → *GitHub* settings → toggle the **`abderahmane-ai/SynoptiQ`** repo **ON**.
3. On GitHub, cut a **Release** (e.g. tag `v0.1.0`). Zenodo archives that tag and mints a DOI.
   `.zenodo.json` supplies the metadata (title, author, license, related HF links) automatically.
4. Copy the DOI. Then:
   - **`CITATION.cff`** — uncomment the `identifiers:` block and paste the DOI.
   - **`README.md`** — add the DOI badge next to the other badges.
   - Optionally add the badge to the three HF cards.

> Zenodo mints a *concept* DOI (always latest) and a *version* DOI (this release). Cite the concept DOI.

## 3. Polish the HF cards

The polished READMEs are in `release/hf_cards/`. Diffs from what's live: a live-demo link on all three,
a `model-index` on KoineFormer, a fixed GitHub link (was `ainouche-abderahmane/SynoptiQ`), sibling
cross-links, and — on the corpus card — removal of the "detect copying direction" use-case (it
contradicts the project's own negative result) plus a note pointing to `DIRECTION_NEGATIVE_RESULT.md`.

Update each repo's `README.md` (edit in the HF web UI, or clone + push):

| File | Target repo |
|---|---|
| `hf_cards/koine-t5.md` | `ainouche-abderahmane/koine-t5` |
| `hf_cards/koineformer.md` | `ainouche-abderahmane/koineformer` |
| `hf_cards/synoptiq-corpus.md` | `datasets/ainouche-abderahmane/synoptiq-corpus` |

```bash
git clone https://huggingface.co/ainouche-abderahmane/koine-t5
cp release/hf_cards/koine-t5.md koine-t5/README.md
cd koine-t5 && git add README.md && git commit -m "Polish card: live demo, links" && git push
# repeat for koineformer and datasets/…/synoptiq-corpus
```

## Order

Space first (its URL is referenced by the cards) → update cards → cut the GitHub Release for the DOI →
paste the DOI back into `CITATION.cff` / `README.md`.
