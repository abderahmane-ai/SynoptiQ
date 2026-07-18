"""Koine Reader — an interlinear reading assistant for Ancient Greek (Hugging Face Space).

Two engines over one word-analysis shape (see :mod:`synoptiq.reader`):

* **Read (gold):** look up any GNT reference and get, per word, the lemma, full
  morphology, English gloss, and Strong's number — human-curated, instant, exact —
  plus the **synoptic parallels** (via the Aland table) that no other reader offers.
* **Analyse (neural):** paste *any* Koine — papyri, apocrypha, patristics — and the
  published Koine-T5 model predicts POS + lemma, glossed against the GNT lexicon.

Gold data is the bundled ~2 MB index (``data/n1904_index.json.gz``); the neural model
loads lazily on first use so the Space starts instantly and the reader works even
where torch is unavailable. The reading output is drawn on a fixed light "parchment"
surface so contrast holds under both light and dark app themes.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import gradio as gr

from synoptiq.reader import GoldReader, IndexReader
from synoptiq.reader.parallels import find_pericope, format_range, parallel_ranges

_HERE = Path(__file__).parent
_GOSPELS = {"Matthew", "Mark", "Luke", "John"}


# ── data loading (bundled artifact, with a local raw-TF fallback for dev) ────────


def _load_reader() -> GoldReader | IndexReader:
    artifact = _HERE / "data" / "n1904_index.json.gz"
    if artifact.exists():
        return IndexReader.from_file(artifact)
    for candidate in (
        Path("data/raw/n1904/tf/1.0.0"),
        _HERE.parent.parent / "data" / "raw" / "n1904" / "tf" / "1.0.0",
    ):
        if candidate.exists():
            return GoldReader.from_dir(candidate)
    msg = "No gold index (data/n1904_index.json.gz) or raw TF data found."
    raise FileNotFoundError(msg)


READER = _load_reader()

_NEURAL = None  # lazily constructed NeuralReader


def _neural():  # noqa: ANN202 - lazy import keeps torch out of startup
    global _NEURAL
    if _NEURAL is None:
        from synoptiq.reader.neural import NeuralReader

        _NEURAL = NeuralReader(gloss_lookup=READER.gloss_for)
    return _NEURAL


# ── HTML rendering (everything sits inside a fixed light .kbox surface) ──────────


def _esc(text: str) -> str:
    return html.escape(text or "")


def _kbox(inner: str) -> str:
    return f'<div class="kbox">{inner}</div>'


def _word_card(word) -> str:  # noqa: ANN001 - WordAnalysis
    strong = f"G{word.strong}" if word.strong else ""
    pos_label = _esc(word.pos)
    morph_label = "(predicted)" if word.predicted else _esc(word.morphology)
    return (
        '<div class="kw">'
        f'<div class="kw-grk">{_esc(word.surface)}</div>'
        f'<div class="kw-gloss">{_esc(word.gloss) or "—"}</div>'
        f'<div class="kw-lem">{_esc(word.lemma)}</div>'
        f'<div class="kw-pos">{pos_label}</div>'
        f'<div class="kw-mor">{morph_label}</div>'
        f'<div class="kw-str">{_esc(strong)}</div>'
        "</div>"
    )


def _interlinear_inner(result) -> str:  # noqa: ANN001 - ReadResult
    if not result.words:
        return '<div class="empty">No text found for that reference.</div>'
    cards = "".join(_word_card(w) for w in result.words)
    return (
        f'<div class="ref-label">{_esc(result.ref)}</div>'
        f'<div class="reading">{_esc(result.text)}</div>'
        f'<div class="kw-grid">{cards}</div>'
    )


def _render_interlinear(result) -> str:  # noqa: ANN001 - ReadResult
    return _kbox(_interlinear_inner(result))


def _render_parallels(book: str, chapter: int, start_verse: int) -> str:
    if book not in _GOSPELS:
        return ""
    pid = find_pericope(book, chapter, start_verse)
    ranges = parallel_ranges(book, chapter, start_verse)
    if pid is None or not ranges:
        return _kbox('<div class="par-none">No catalogued synoptic parallel here.</div>')
    blocks = []
    for other, rng in ranges.items():
        passage = READER.passage(other, rng[0], rng[1])
        blocks.append(
            '<div class="par">'
            f'<div class="par-ref">{_esc(other)} {format_range(rng)}</div>'
            f'<div class="par-text">{_esc(passage.text)}</div>'
            "</div>"
        )
    return _kbox(
        f'<div class="par-title">Synoptic parallels · Aland §{_esc(pid)}</div>{"".join(blocks)}'
    )


# ── event handlers ──────────────────────────────────────────────────────────────


def update_chapters(book: str):  # noqa: ANN201
    chapters = [str(c) for c in READER.chapters(book)]
    return gr.update(choices=chapters, value=chapters[0] if chapters else None)


def do_read(book: str, chapter: str, verses: str) -> tuple[str, str]:
    if not book or not chapter:
        return _kbox('<div class="empty">Choose a book and chapter.</div>'), ""
    ch = int(chapter)
    vtext = (verses or "").strip()
    if vtext:
        ref = f"{book} {ch}:{vtext}"
        m = re.match(r"(\d+)", vtext)
        start_verse = int(m.group(1)) if m else 1
    else:
        ref = f"{book} {ch}"
        present = READER.verses(book, ch)
        start_verse = present[0] if present else 1
    try:
        result = READER.read(ref)
    except ValueError:
        return _kbox(f'<div class="empty">Could not parse “{_esc(ref)}”.</div>'), ""
    return _render_interlinear(result), _render_parallels(book, ch, start_verse)


def do_analyze(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return _kbox('<div class="empty">Paste some Koine Greek to analyse.</div>')
    try:
        reader = _neural()
        reader.load()
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
        return _kbox(
            '<div class="empty">Neural model unavailable in this environment '
            f"({_esc(type(exc).__name__)}). The Read tab still works fully.</div>"
        )
    try:
        result = reader.analyze(text)
    except Exception as exc:  # noqa: BLE001
        return _kbox(f'<div class="empty">Analysis failed: {_esc(str(exc))}</div>')
    if not result.words:
        return _kbox('<div class="empty">No Greek words detected.</div>')
    note = (
        '<div class="note">Predicted by <b>Koine-T5</b> — part of speech + lemma only '
        "(no full morphology); glosses looked up from the GNT lexicon. Predictions can "
        "err on rare forms.</div>"
    )
    return _kbox(note + _interlinear_inner(result))


# ── UI ──────────────────────────────────────────────────────────────────────────

# The reading output lives on a fixed papyrus surface (.kbox) with dark ink, so it is
# legible regardless of the surrounding Gradio light/dark theme. Accents: marine
# #123C48, terracotta #B75B36, papyrus #F7F2E7.
_CSS = """
.kbox { background: #F7F2E7; color: #3B342A; border-radius: 12px; padding: 16px 20px;
  margin-top: 6px; }
.ref-label { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
  color: #B75B36; font-weight: 600; margin-bottom: 6px; }
.reading { font-size: 1.35rem; line-height: 1.9; color: #123C48;
  padding-bottom: 14px; margin-bottom: 14px; border-bottom: 2px solid #E4DBC7; }
.kw-grid { display: flex; flex-wrap: wrap; gap: 10px; }
.kw { min-width: 104px; background: #FFFFFF; border: 1px solid #E4DBC7;
  border-radius: 10px; padding: 10px 12px; text-align: center; }
.kw-grk { font-size: 1.25rem; font-weight: 600; color: #123C48; }
.kw-gloss { font-size: .8rem; color: #4A4A4A; margin-top: 3px; min-height: 1.1em; }
.kw-lem { font-size: .95rem; color: #B75B36; margin-top: 6px; }
.kw-pos { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
  color: #8A7F63; margin-top: 4px; }
.kw-mor { font-size: .72rem; color: #6B6250; margin-top: 2px; }
.kw-str { font-size: .68rem; color: #9A8F73; margin-top: 4px;
  font-variant-numeric: tabular-nums; }
.par-title { font-weight: 600; color: #123C48; margin-bottom: 10px; }
.par { background: #FFFFFF; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
  border-left: 3px solid #123C48; }
.par-ref { font-size: .85rem; font-weight: 600; color: #B75B36; }
.par-text { font-size: 1.05rem; line-height: 1.6; margin-top: 3px; color: #123C48; }
.par-none, .empty { color: #6B6250; font-style: italic; }
.note { font-size: .85rem; color: #6B6250; margin-bottom: 12px;
  border-left: 3px solid #B75B36; padding-left: 10px; }
"""

_HEADER = """
# 🏛️ Koine Reader

Read the Greek New Testament with **lemma · morphology · gloss · Strong's** for every word —
and the **synoptic parallels** for each Gospel passage. Or paste *any* Koine and let
**[Koine-T5](https://huggingface.co/ainouche-abderahmane/koine-t5)** analyse it.

*Gospel text & morphology: Nestle 1904 via the [MACULA Greek](https://github.com/Clear-Bible/macula-greek)
Text-Fabric edition (glosses © Bible OnLine Learner). Part of the
[SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ) project.*
"""

_EXAMPLES = [
    ["John", "1", "1"], ["Mark", "1", "9-11"],
    ["Matthew", "5", "3-12"], ["Luke", "15", ""],
]

with gr.Blocks(title="Koine Reader") as demo:  # theme/css passed to launch() (Gradio 6.0)
    gr.Markdown(_HEADER)

    with gr.Tab("Read (New Testament)"):
        with gr.Row():
            book = gr.Dropdown(choices=READER.books(), value="John", label="Book", scale=2)
            chapter = gr.Dropdown(
                choices=[str(c) for c in READER.chapters("John")],
                value="1", label="Chapter", scale=1,
            )
            verses = gr.Textbox(
                value="1", label="Verse(s) — optional",
                placeholder="1   ·   3-12   ·   blank = whole chapter", scale=2,
            )
            read_btn = gr.Button("Read", variant="primary", scale=1)
        gr.Examples(examples=_EXAMPLES, inputs=[book, chapter, verses], label="Try")
        interlinear = gr.HTML()
        parallels = gr.HTML()

        book.change(update_chapters, inputs=book, outputs=chapter)
        read_btn.click(do_read, inputs=[book, chapter, verses], outputs=[interlinear, parallels])
        verses.submit(do_read, inputs=[book, chapter, verses], outputs=[interlinear, parallels])
        demo.load(do_read, inputs=[book, chapter, verses], outputs=[interlinear, parallels])

    with gr.Tab("Analyse any Koine (Koine-T5)"):
        gr.Markdown(
            "Paste Koine Greek that isn't in the GNT — a papyrus, an apocryphal gospel, "
            "the Apostolic Fathers, or your own sentence. The model loads on first use "
            "(a few seconds)."
        )
        neural_in = gr.Textbox(
            lines=3, label="Koine Greek",
            placeholder="π.χ.  Ἐν ἀρχῇ ἐποίησεν ὁ θεὸς τὸν οὐρανὸν καὶ τὴν γῆν",
        )
        analyze_btn = gr.Button("Analyse", variant="primary")
        gr.Examples(
            examples=[
                ["Ἐν ἀρχῇ ἐποίησεν ὁ θεὸς τὸν οὐρανὸν καὶ τὴν γῆν"],
                ["Διδαχὴ κυρίου τοῖς ἔθνεσιν διὰ τῶν δώδεκα ἀποστόλων"],
            ],
            inputs=neural_in,
        )
        neural_out = gr.HTML()
        analyze_btn.click(do_analyze, inputs=neural_in, outputs=neural_out)
        neural_in.submit(do_analyze, inputs=neural_in, outputs=neural_out)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=_CSS)
