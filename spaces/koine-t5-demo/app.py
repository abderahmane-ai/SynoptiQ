"""Koine-T5 interactive demo (Hugging Face Space).

Loads the Koine-T5 LoRA adapter on GreTa and exposes its four tasks:
POS tagging, lemmatization, text infilling, and synoptic style transfer.

The inference recipe (tokenizer setup, per-task decoding, POS upper-casing)
mirrors the model card exactly — https://huggingface.co/ainouche-abderahmane/koine-t5
"""
from __future__ import annotations

import gradio as gr
from peft import PeftModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "bowphs/GreTa"
ADAPTER = "ainouche-abderahmane/koine-t5"
MAX_SEQ_LEN = 256


def load_model() -> tuple:
    """Load GreTa + the Koine-T5 adapter. No embedding resize — the 100 T5
    sentinels map onto GreTa's pre-trained ghost slots (32003-32102)."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]}
    )
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return tokenizer, model


TOKENIZER, MODEL = load_model()

TASK_PREFIX = {
    "POS tagging": "pos: ",
    "Lemmatization": "lemma: ",
    "Text infilling": "",  # no prefix — pass text with <extra_id_N> masks
    "Synoptic: Mark → Matthew": "synoptic mark_to_matt: ",
    "Synoptic: Mark → Luke": "synoptic mark_to_luke: ",
}


def run(task: str, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "(enter some Greek text)"
    prefix = TASK_PREFIX[task]
    full = prefix + text if not text.startswith(prefix) else text
    inputs = TOKENIZER(full, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)

    if task in ("POS tagging", "Lemmatization"):
        gen_kwargs = {"max_new_tokens": 256, "num_beams": 1, "do_sample": False}
    elif task == "Text infilling":
        gen_kwargs = {"max_new_tokens": 64, "num_beams": 4, "early_stopping": True}
    else:  # synoptic style transfer
        gen_kwargs = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    with torch.no_grad():
        out = MODEL.generate(**inputs, **gen_kwargs)

    # Keep the <extra_id_N> sentinels visible for infilling; strip them otherwise.
    decoded = TOKENIZER.decode(out[0], skip_special_tokens=(task != "Text infilling"))
    # GreTa lowercases; MorphGNT POS codes are case-unique, so upper-case restores them losslessly.
    if task == "POS tagging":
        decoded = decoded.upper()
    return decoded.strip() or "(empty output)"


EXAMPLES = [
    ["POS tagging", "καὶ φωνὴ ἐγένετο ἐκ τῶν οὐρανῶν"],
    ["Lemmatization", "καὶ φωνὴ ἐγένετο ἐκ τῶν οὐρανῶν"],
    [
        "Text infilling",
        "Ἀρχὴ τοῦ εὐαγγελίου Ἰησοῦ <extra_id_0> καθὼς γέγραπται ἐν τῷ <extra_id_1> τῷ προφήτῃ",
    ],
    [
        "Synoptic: Mark → Luke",
        "καὶ πρωῒ ἔννυχα λίαν ἀναστὰς ἐξῆλθεν καὶ ἀπῆλθεν εἰς ἔρημον τόπον κἀκεῖ προσηύχετο.",
    ],
]

DESCRIPTION = """
# 🏛️ Koine-T5 — one model, four tasks for Ancient Greek

A **104 MB LoRA adapter** turning [GreTa](https://huggingface.co/bowphs/GreTa) into a multitask
Ancient-Greek workhorse: **POS tagging · lemmatization · text infilling · synoptic style transfer**
— 96.6% POS token accuracy on New Testament Koine.

**Model:** [`ainouche-abderahmane/koine-t5`](https://huggingface.co/ainouche-abderahmane/koine-t5) ·
**Project:** [SynoptiQ](https://github.com/abderahmane-ai/SynoptiQ)

*Tips.* For **Text infilling**, mark gaps with `<extra_id_0>`, `<extra_id_1>`, … (as in T5).
Synoptic transfer is **experimental** (155 training pairs) — expect evocative pastiche, not faithful
translation. Output is lowercase (GreTa case-folds); POS codes are upper-cased for you.
"""

with gr.Blocks(title="Koine-T5 Demo") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            task = gr.Dropdown(
                choices=list(TASK_PREFIX.keys()), value="POS tagging", label="Task"
            )
            text = gr.Textbox(
                lines=4, label="Greek input",
                placeholder="Paste Koine or Classical Greek here…",
            )
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=1):
            output = gr.Textbox(lines=6, label="Output", buttons=["copy"])
    gr.Examples(
        examples=EXAMPLES, inputs=[task, text], outputs=output, fn=run, cache_examples=False
    )
    btn.click(run, inputs=[task, text], outputs=output)
    text.submit(run, inputs=[task, text], outputs=output)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
