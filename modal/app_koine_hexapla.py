"""Koine-T5-Hexapla: the MAX edition — powerful generation, zero analysis regression.

A generation-focused evolution of ``app_koine_t5.py`` (named after Origen's Hexapla, the
six-column parallel-scripture alignment — the ancient precedent for SynoptiQ). Same GreTa
base + LoRA, but built to fix the discourse-level failures seen in
``docs/gospel_of_the_savior.md`` (speaker/pericope bleed, mode-collapse) WITHOUT sacrificing
the POS/lemma/synoptic competence the model already has. Five levers over Koine-T5:

  1. A ~17M-word raw-Koine diet (LXX + First1KGreek + Apostolic Fathers + SBLGNT-minus-
     synoptics), prepared by ``scripts/prepare_koine_maxi_corpus.py`` (was ~263K tokens).
  2. A new **continuation (prefix-LM)** task that teaches autoregressive fluency — the signal
     span-infill denoise never provides.
  3. **Passage-level** units (whole windows) + **512-token** context (was 256) for
     cross-sentence state tracking.
  4. More adapter capacity (LoRA **r=128 α=256**) so added tasks stop competing with POS.
  5. A **two-stage curriculum** (generative backbone → rebalanced multitask) + a
     **regression-gated** ``evaluate_all``: a checkpoint is only "best" if POS/lemma stay at
     or above the published Koine-T5 gates, then the generation score is maximized. "No
     sacrifice" is enforced, not hoped for.

The script is standalone (no ``synoptiq`` imports), mirroring ``app_koine_t5.py``. It reads
the prepared corpus from the ``synoptiq-data`` volume (``/data/koine_maxi``); if absent it
falls back to the Gospel+PROIEL pools so a run is always possible.

Usage:
    modal run modal/app_koine_hexapla.py::train        # A10G/A100 GPU, auto-resumes
    modal app logs koine-t5-hexapla                    # live logs
    python modal/app_koine_hexapla.py demo [adapter]   # local A/B vs GreTa base
"""

from __future__ import annotations

import json
import math
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

try:
    import modal  # type: ignore[import-untyped]
except ImportError:
    modal = None  # type: ignore[assignment]

# ── Constants ────────────────────────────────────────────────────────────────

DATA_VOLUME = "synoptiq-data"
OUTPUT_VOLUME = "koine-hexapla-outputs"
GPU_TYPE = "A10G"
TIMEOUT = 86_400

BASE_MODEL_ID = "bowphs/GreTa"

# LoRA / model — 2× Koine-T5's rank so the new generation tasks don't crowd out POS.
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
MAX_SEQ_LEN = 512          # 2× Koine-T5 — a whole pericope fits (state-tracking)
BATCH_SIZE = 2            # smaller micro-batch to fit 512-len on A10G …
GRAD_ACCUM = 16           # … × accum → effective batch 32 (unchanged)
LR = 1e-4

MAX_STEPS = 40_000
WARMUP_STEPS = 2_000
SAVE_STEPS = 1_000
EVAL_STEPS = 1_000
LOG_STEPS = 100
CKPT_KEEP = 2

# Two-stage curriculum: first STAGE_A_FRAC of steps build the generative backbone
# (denoise + continuation dominate); then rebalance to protect the analysis tasks.
STAGE_A_FRAC = 0.4
STAGE_A_WEIGHTS = {"denoise": 4.0, "continuation": 4.0, "pos": 1.0, "lemma": 0.5, "synoptic": 0.5}
STAGE_B_WEIGHTS = {"denoise": 2.0, "continuation": 2.0, "pos": 3.0, "lemma": 1.0, "synoptic": 1.5}
_TASK_ORDER = ("pos", "lemma", "synoptic", "denoise", "continuation")

# Koine style must not be swamped by First1KGreek's 16M Classical words — sample the
# generative pools register-first (≈ DAPT's 70/30 Koine/Classical replay ratio).
REGISTER_WEIGHTS = {"koine": 0.7, "classical": 0.3}
_KOINE_SOURCES = {"lxx", "apostolic", "sblgnt"}

# Regression gates: a checkpoint may only become "best" if it holds these (published
# Koine-T5 dev numbers). Lemma gate is conservative; raise to the measured baseline once known.
GATE_POS_NT = 0.966
GATE_POS_CL = 0.877
GATE_LEMMA = 0.80

# Eval sizing (kept small so evaluate_all stays cheap at every checkpoint).
MAX_POS_WORDS = 60
EVAL_MAX_PER_SUBSET = 250
EVAL_BATCH_SIZE = 32
PPL_EVAL_N = 200
GEN_EVAL_N = 100
MORPH_EVAL_N = 40

# Span corruption (T5 §3.1).
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LEN = 3.0

# Contrastive-search generation (anti-degeneration for free-form Greek).
GEN_PENALTY_ALPHA = 0.6
GEN_TOP_K = 4
GEN_MAX_NEW_TOKENS = 256
GEN_REP_PENALTY = 1.25
GEN_NO_REPEAT_NGRAM = 3

# Prepared-corpus location on the data volume (see prepare_koine_maxi_corpus.py).
MAXI_CORPUS_DIR = "/data/koine_maxi"

# The 13 MorphGNT POS codes — used for the morphological self-consistency metric.
VALID_POS_TAGS = frozenset(
    {"V-", "N-", "RA", "A-", "D-", "P-", "C-", "RP", "RR", "RD", "RI", "I-", "X-"}
)

# ── PROIEL treebank configuration ────────────────────────────────────────────
PROIEL_SPLITS = ("train", "dev", "test")
PROIEL_URL_TMPL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Ancient_Greek-PROIEL/master/grc_proiel-ud-{split}.conllu"
)
PROIEL_REMOTE_DIR = "/data/proiel"
PROIEL_LOCAL_DIR = "data/raw/proiel"
PROIEL_CACHE_DIR = "/tmp/proiel"


def _proiel_file(dir_: str, split: str) -> Path:
    return Path(dir_) / f"grc_proiel-ud-{split}.conllu"


def _proiel_files_present(dir_: str | None) -> bool:
    if not dir_:
        return False
    return all(
        (p := _proiel_file(dir_, s)).exists() and p.stat().st_size > 0 for s in PROIEL_SPLITS
    )


def _download_proiel(dest: str) -> bool:
    import urllib.request

    try:
        Path(dest).mkdir(parents=True, exist_ok=True)
        for split in PROIEL_SPLITS:
            out = _proiel_file(dest, split)
            if out.exists() and out.stat().st_size > 0:
                continue
            print(f"  [proiel] downloading {split} → {out}")
            urllib.request.urlretrieve(PROIEL_URL_TMPL.format(split=split), out)
        return _proiel_files_present(dest)
    except Exception as exc:
        print(f"  [proiel] download failed ({type(exc).__name__}: {exc})")
        return False


def resolve_proiel_dir(allow_download: bool = True) -> str | None:
    for candidate in (os.environ.get("PROIEL_DIR"), PROIEL_REMOTE_DIR, PROIEL_LOCAL_DIR):
        if _proiel_files_present(candidate):
            return candidate
    if allow_download:
        for target in (PROIEL_LOCAL_DIR, PROIEL_CACHE_DIR):
            if _download_proiel(target):
                return target
    return None


# ── PROIEL XPOS → MorphGNT POS mapping (see app_koine_t5.py for the rationale) ─────
PROIEL_XPOS_TO_MORPHGNT: dict[str, str] = {
    "V-": "V-",
    "Nb": "N-", "Ne": "N-", "F-": "N-",
    "S-": "RA",
    "A-": "A-", "Ma": "A-", "Mo": "A-", "Ps": "A-",
    "Df": "D-", "Du": "D-",
    "R-": "P-",
    "C-": "C-", "G-": "C-", "Dq": "C-",
    "Pp": "RP", "Pk": "RP", "Pc": "RP",
    "Pr": "RR", "Pd": "RD", "Pi": "RI",
    "I-": "I-",
}
FALLBACK_MORPHGNT = "X-"
INDEF_PRONOUN_LEMMAS = {"τις"}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    ).lower()


def map_xpos_to_morphgnt(xpos: str, lemma: str) -> str:
    if xpos == "Px":
        return "RI" if _strip_accents(lemma) in INDEF_PRONOUN_LEMMAS else "A-"
    return PROIEL_XPOS_TO_MORPHGNT.get(xpos, FALLBACK_MORPHGNT)


# ── Modal image / app ────────────────────────────────────────────────────────
_REQUIREMENTS = [
    "torch>=2.6.0", "transformers>=4.51.0", "peft>=0.15.0", "safetensors>=0.4.0",
    "sentencepiece>=0.2.0", "datasets>=3.3.0", "accelerate>=1.3.0",
    "pandas>=2.2", "pyarrow>=16.0", "tqdm>=4.67", "numpy>=1.26",
]


def _build_image() -> Any:
    if modal is None:
        raise RuntimeError("Modal not installed")
    image = modal.Image.debian_slim(python_version="3.12")
    for req in _REQUIREMENTS:
        image = image.pip_install(req)
    return image


app = modal.App("koine-t5-hexapla") if modal is not None else None

_VOLUMES = {
    "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
} if modal is not None else {}


def _commit() -> None:
    if modal is not None:
        modal.Volume.from_name(OUTPUT_VOLUME).commit()


# ── Tokenizer + model (identical to Koine-T5 except LoRA rank) ────────────────

def build_tokenizer():
    """Load GreTa's tokenizer and register the 100 T5 sentinels into the ghost slots.

    Do NOT add a [PAD] token or resize — that desyncs the decoder-start id and collapses
    seq2seq generation (see CLAUDE.md tokenizer notes).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    sentinels = [f"<extra_id_{i}>" for i in range(100)]
    tokenizer.add_special_tokens({"additional_special_tokens": sentinels})
    assert tokenizer.convert_tokens_to_ids("<extra_id_0>") == 32003, "Sentinel ID mismatch"
    assert tokenizer.convert_tokens_to_ids("<extra_id_99>") == 32102, "Sentinel ID mismatch"
    return tokenizer


def load_model_with_lora(tokenizer, device: str = "cpu"):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, dtype=torch.bfloat16)
    if hasattr(base.config, "tie_word_embeddings"):
        base.config.tie_word_embeddings = False
    base.config.vocab_size = len(tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=["q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1"],
    )
    model = get_peft_model(base, lora_config)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Koine-T5-Hexapla LoRA params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


# ── T5 span corruption (verbatim from Koine-T5) ──────────────────────────────

def _random_spans_noise_mask(length, noise_density, mean_span_len, rng) -> list[bool]:
    num_noise = max(1, round(length * noise_density))
    num_noise = min(num_noise, length - 1)
    span_lengths: list[int] = []
    remaining = num_noise
    while remaining > 0:
        span_len = max(1, int(rng.geometric(p=1.0 / mean_span_len)))
        span_len = min(span_len, remaining)
        span_lengths.append(span_len)
        remaining -= span_len
    num_nonnoise = length - num_noise
    gap_lengths = _split_into_k_parts(num_nonnoise, len(span_lengths) + 1, rng)
    mask: list[bool] = []
    for g, gap in enumerate(gap_lengths):
        mask.extend([False] * gap)
        if g < len(span_lengths):
            mask.extend([True] * span_lengths[g])
    return mask[:length]


def _split_into_k_parts(total: int, k: int, rng) -> list[int]:
    if k == 1:
        return [total]
    cuts = sorted(rng.integers(0, total + 1, size=k - 1).tolist())
    cuts = [0, *cuts, total]
    return [cuts[i + 1] - cuts[i] for i in range(k)]


def apply_span_corruption(input_ids, tokenizer, rng,
                          noise_density=NOISE_DENSITY, mean_span_len=MEAN_NOISE_SPAN_LEN):
    mask = _random_spans_noise_mask(len(input_ids), noise_density, mean_span_len, rng)
    corrupted: list[int] = []
    target: list[int] = []
    sentinel_idx = 0
    in_noise_span = False
    for token_id, is_noise in zip(input_ids, mask, strict=True):
        if is_noise:
            if not in_noise_span:
                sentinel_id = tokenizer.convert_tokens_to_ids(f"<extra_id_{sentinel_idx}>")
                corrupted.append(sentinel_id)
                target.append(sentinel_id)
                sentinel_idx += 1
                in_noise_span = True
            target.append(token_id)
        else:
            in_noise_span = False
            corrupted.append(token_id)
    target.append(tokenizer.eos_token_id)
    return corrupted, target


# ── PROIEL CoNLL-U loading (verbatim from Koine-T5) ───────────────────────────

def iter_conllu_sentences(path):
    source = ""
    toks: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("# source"):
                source = line.split("=", 1)[1].strip() if "=" in line else ""
            elif line.startswith("#"):
                continue
            elif not line.strip():
                if toks:
                    yield source, toks
                toks = []
            else:
                cols = line.split("\t")
                if len(cols) < 5 or "-" in cols[0] or "." in cols[0]:
                    continue
                toks.append((cols[1], cols[2], cols[4]))
        if toks:
            yield source, toks


def _is_nt_source(source: str) -> bool:
    return "new testament" in source.lower()


def build_proiel_training(proiel_dir: str) -> tuple[list[dict], list[dict], list[str]]:
    pos_examples: list[dict] = []
    lemma_examples: list[dict] = []
    raw_texts: list[str] = []
    path = _proiel_file(proiel_dir, "train")
    if not path.exists():
        return pos_examples, lemma_examples, raw_texts
    for source, toks in iter_conllu_sentences(path):
        forms = [t[0] for t in toks]
        text = " ".join(forms)
        if len(toks) >= 2:
            raw_texts.append(text)
        if 2 <= len(toks) <= MAX_POS_WORDS:
            tags = [map_xpos_to_morphgnt(t[2], t[1]) for t in toks]
            lemmas = [t[1] for t in toks]
            pos_examples.append({"task": "pos", "input_text": f"pos: {text}",
                                 "target_text": " ".join(tags)})
            lemma_examples.append({"task": "lemma", "input_text": f"lemma: {text}",
                                   "target_text": " ".join(lemmas)})
    print(f"  PROIEL train: {len(pos_examples):,} pos/lemma, {len(raw_texts):,} denoise texts")
    return pos_examples, lemma_examples, raw_texts


def build_proiel_eval(proiel_dir: str) -> tuple[list[dict], list[dict]]:
    """POS + lemma eval records from PROIEL dev: ([{text,gold,subset}], [{text,gold,subset}])."""
    pos_records: list[dict] = []
    lemma_records: list[dict] = []
    path = _proiel_file(proiel_dir, "dev")
    if not path.exists():
        return pos_records, lemma_records
    for source, toks in iter_conllu_sentences(path):
        if not (2 <= len(toks) <= MAX_POS_WORDS):
            continue
        subset = "nt" if _is_nt_source(source) else "classical"
        text = " ".join(t[0] for t in toks)
        pos_records.append({"text": text,
                            "gold": [map_xpos_to_morphgnt(t[2], t[1]) for t in toks],
                            "subset": subset})
        lemma_records.append({"text": text, "gold": [t[1] for t in toks], "subset": subset})
    return pos_records, lemma_records


# ── Gospel corpus loading (verbatim from Koine-T5) ────────────────────────────

def load_synoptic_pairs(tokens_path, pericopes_path) -> tuple[list[dict], list[str]]:
    import pandas as pd
    from collections import defaultdict

    tokens_df = pd.read_parquet(tokens_path)
    _ = pd.read_parquet(pericopes_path)
    pericope_book_tokens: dict[tuple, list] = defaultdict(list)
    for row in tokens_df.itertuples():
        pid = getattr(row, "pericope_id", None)
        book = getattr(row, "book", None)
        if pid is not None and book is not None:
            pericope_book_tokens[(pid, book)].append(row)

    examples: list[dict] = []
    raw_texts: list[str] = []

    def _text(rows) -> str:
        return " ".join(str(getattr(r, "text", "")) for r in rows)

    def _pos(rows) -> str:
        return " ".join(str(getattr(r, "pos", "?")) for r in rows)

    def _lemma(rows) -> str:
        return " ".join(str(getattr(r, "lemma", getattr(r, "text", ""))) for r in rows)

    for pid in {pid for (pid, _) in pericope_book_tokens}:
        if (pid, "Mark") not in pericope_book_tokens:
            continue
        mk_text = _text(pericope_book_tokens[(pid, "Mark")])
        for book in ("Mark", "Matthew", "Luke"):
            if (pid, book) not in pericope_book_tokens:
                continue
            r = pericope_book_tokens[(pid, book)]
            text = _text(r)
            raw_texts.append(text)
            examples.append({"task": "pos", "input_text": f"pos: {text}", "target_text": _pos(r)})
            examples.append({"task": "lemma", "input_text": f"lemma: {text}",
                             "target_text": _lemma(r)})
        if (pid, "Matthew") in pericope_book_tokens:
            examples.append({"task": "synoptic_mk_to_mt",
                             "input_text": f"synoptic mark_to_matt: {mk_text}",
                             "target_text": _text(pericope_book_tokens[(pid, "Matthew")])})
        if (pid, "Luke") in pericope_book_tokens:
            examples.append({"task": "synoptic_mk_to_lk",
                             "input_text": f"synoptic mark_to_luke: {mk_text}",
                             "target_text": _text(pericope_book_tokens[(pid, "Luke")])})
    print(f"  Gospel corpus: {len(examples)} instruction examples, {len(raw_texts)} denoise texts")
    return examples, raw_texts


# ── Prepared MAXI corpus (the new generative fuel) ────────────────────────────

def _register_of(source: str) -> str:
    return "koine" if source in _KOINE_SOURCES else "classical"


def load_maxi_corpus(corpus_dir: str) -> tuple[dict[str, list], list[dict], list[dict]]:
    """Read the prepared koine_maxi artifact.

    Returns (denoise_by_register, continuation_by_register, continuation_eval):
      * denoise_by_register / continuation_by_register: {"koine": [...], "classical": [...]}
      * continuation_eval: held-out {input_text, target_text} rows for evaluate_all.
    Empty structures if the artifact is absent (caller falls back to Gospel+PROIEL).
    """
    cdir = Path(corpus_dir)
    denoise: dict[str, list] = {"koine": [], "classical": []}
    continuation: dict[str, list] = {"koine": [], "classical": []}
    cont_eval: list[dict] = []

    corpus_jsonl = cdir / "corpus.jsonl"
    if corpus_jsonl.exists():
        with open(corpus_jsonl, encoding="utf-8") as fh:
            for raw_line in fh:
                r = json.loads(raw_line)
                reg = r.get("register") or _register_of(r.get("source", ""))
                if r["task"] == "denoise":
                    denoise[reg].append({"task": "denoise", "raw_text": r["raw_text"]})
                elif r["task"] == "continuation":
                    continuation[reg].append({"task": "continuation",
                                              "input_text": r["input_text"],
                                              "target_text": r["target_text"]})
    eval_path = cdir / "eval_continuation.jsonl"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as fh:
            cont_eval = [json.loads(line) for line in fh]

    n_dn = sum(len(v) for v in denoise.values())
    n_ct = sum(len(v) for v in continuation.values())
    print(f"  MAXI corpus: denoise={n_dn:,} (koine {len(denoise['koine']):,}/"
          f"classical {len(denoise['classical']):,}), continuation={n_ct:,}, "
          f"cont-eval={len(cont_eval):,}")
    return denoise, continuation, cont_eval


def build_task_pools(tokens_path, pericopes_path, proiel_dir, corpus_dir):
    """Assemble the five task pools. denoise/continuation are register-split dicts."""
    corpus_examples, corpus_raw = load_synoptic_pairs(tokens_path, pericopes_path)
    pos_pool = [e for e in corpus_examples if e["task"] == "pos"]
    lemma_pool = [e for e in corpus_examples if e["task"] == "lemma"]
    synoptic_pool = [e for e in corpus_examples if e["task"].startswith("synoptic")]

    denoise: dict[str, list] = {"koine": [], "classical": []}
    continuation: dict[str, list] = {"koine": [], "classical": []}
    cont_eval: list[dict] = []
    if corpus_dir and (Path(corpus_dir) / "corpus.jsonl").exists():
        denoise, continuation, cont_eval = load_maxi_corpus(corpus_dir)
    # Gospel raw text + PROIEL raw text join the Koine denoise pool (small but on-domain).
    denoise["koine"].extend({"task": "denoise", "raw_text": t} for t in corpus_raw)

    if proiel_dir:
        p_pos, p_lemma, p_raw = build_proiel_training(proiel_dir)
        pos_pool += p_pos
        lemma_pool += p_lemma
        denoise["koine"].extend({"task": "denoise", "raw_text": t} for t in p_raw)
    else:
        print("  [warn] PROIEL unavailable — pos/lemma from Gospel corpus only")

    pools = {
        "pos": pos_pool,
        "lemma": lemma_pool,
        "synoptic": synoptic_pool,
        "denoise": denoise,
        "continuation": continuation,
    }
    return pools, cont_eval


# ── Register-aware, two-stage weighted sampling ───────────────────────────────

def _pool_nonempty(pool) -> bool:
    if isinstance(pool, dict):
        return any(pool.values())
    return bool(pool)


def weights_for_step(step: int) -> dict[str, float]:
    """Stage-A weights for the first STAGE_A_FRAC of training, then Stage-B."""
    return STAGE_A_WEIGHTS if step < STAGE_A_FRAC * MAX_STEPS else STAGE_B_WEIGHTS


def _draw_example(task: str, pools: dict, rng_py):
    """Draw one example from a task pool, register-first for the generative pools."""
    pool = pools[task]
    if isinstance(pool, dict):
        regs = [r for r in ("koine", "classical") if pool.get(r)]
        reg = rng_py.choices(regs, weights=[REGISTER_WEIGHTS[r] for r in regs])[0] \
            if len(regs) > 1 else regs[0]
        return rng_py.choice(pool[reg])
    return rng_py.choice(pool)


def sample_balanced_batch(pools, batch_size, step, rng_py) -> list[dict]:
    weights = weights_for_step(step)
    active = [t for t in _TASK_ORDER if _pool_nonempty(pools.get(t)) and weights.get(t, 0) > 0]
    if not active:
        return []
    tasks = rng_py.choices(active, weights=[weights[t] for t in active], k=batch_size)
    return [_draw_example(t, pools, rng_py) for t in tasks]


# ── Collation (denoise = online corruption; everything else = seq2seq) ────────

def _collate_batch(examples, tokenizer, rng, max_len: int = MAX_SEQ_LEN):
    import torch

    batch_input_ids: list[Any] = []
    batch_attn_masks: list[Any] = []
    batch_labels: list[Any] = []
    pad_id = 0

    for ex in examples:
        if ex["task"] == "denoise":
            raw = ex.get("raw_text") or ex.get("input_text", "")
            raw_ids = tokenizer.encode(raw, add_special_tokens=False,
                                       max_length=max_len - 10, truncation=True)
            if len(raw_ids) < 8:
                continue
            corrupted, target = apply_span_corruption(raw_ids, tokenizer, rng)
            input_ids = torch.tensor(corrupted[:max_len], dtype=torch.long)
            label_ids = torch.tensor(target[:max_len], dtype=torch.long)
        else:
            src = tokenizer(ex["input_text"], max_length=max_len, truncation=True,
                            return_tensors="pt")
            tgt = tokenizer(ex["target_text"], max_length=max_len, truncation=True,
                            return_tensors="pt")
            input_ids = src["input_ids"][0]
            label_ids = tgt["input_ids"][0].clone()

        label_ids[label_ids == pad_id] = -100
        if (label_ids != -100).sum() == 0:
            continue
        batch_input_ids.append(input_ids)
        batch_attn_masks.append(torch.ones(len(input_ids), dtype=torch.long))
        batch_labels.append(label_ids)

    if not batch_input_ids:
        return None

    def _pad(tensors, pad_value):
        m = max(t.size(0) for t in tensors)
        return torch.stack([torch.nn.functional.pad(t, (0, m - t.size(0)), value=pad_value)
                            for t in tensors])

    return {
        "input_ids": _pad(batch_input_ids, pad_id),
        "attention_mask": _pad(batch_attn_masks, 0),
        "labels": _pad(batch_labels, -100),
    }


# ── Prediction + evaluation ───────────────────────────────────────────────────

def _batch_tag_predict(model, tokenizer, texts, device, prefix, *, upper: bool):
    """Batched greedy decode of `<prefix><text>` for pos/lemma. Right-padded (T5 encoder)."""
    import torch

    max_words = max(len(t.split()) for t in texts)
    max_new = min(220, max_words * 4 + 10)
    encoded = tokenizer([prefix + t for t in texts], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        out = model.generate(input_ids=encoded["input_ids"],
                             attention_mask=encoded["attention_mask"],
                             max_new_tokens=max_new, num_beams=1, do_sample=False)
    preds = []
    for seq in out:
        toks = tokenizer.decode(seq, skip_special_tokens=True).split()
        preds.append([t.upper() for t in toks] if upper else toks)
    return preds


def evaluate_tagging(model, tokenizer, records, device, prefix, *, upper, batch_size=EVAL_BATCH_SIZE):
    """Token-accuracy + EM on PROIEL-dev records, split NT / Classical. Shared by pos & lemma."""
    was_training = model.training
    model.eval()
    agg = {"nt": [0, 0, 0, 0], "classical": [0, 0, 0, 0]}
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        preds = _batch_tag_predict(model, tokenizer, [r["text"] for r in chunk], device,
                                   prefix, upper=upper)
        for r, pred in zip(chunk, preds, strict=True):
            gold = r["gold"]
            a = agg[r["subset"]]
            a[0] += sum(1 for g, p in zip(gold, pred, strict=False) if g == p)
            a[1] += len(gold)
            a[2] += int(pred == gold)
            a[3] += 1
    if was_training:
        model.train()

    def pack(a):
        return {"tok": a[0] / max(1, a[1]), "em": a[2] / max(1, a[3]), "n": a[3]}

    nt, cl = agg["nt"], agg["classical"]
    overall = [nt[0] + cl[0], nt[1] + cl[1], nt[2] + cl[2], nt[3] + cl[3]]
    return {"nt": pack(nt), "classical": pack(cl), "overall": pack(overall)}


def _select_eval_subset(records, per_subset, seed: int = 1234):
    import random as _random

    rng = _random.Random(seed)
    out: list[dict] = []
    for subset in ("nt", "classical"):
        pool = [r for r in records if r["subset"] == subset]
        rng.shuffle(pool)
        out.extend(pool[:per_subset])
    return out


def _f1_tokens(pred: str, gold: str) -> float:
    from collections import Counter

    p = [_strip_accents(t) for t in pred.split()]
    g = [_strip_accents(t) for t in gold.split()]
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _generate_continuations(model, tokenizer, prefixes, device):
    """Free-form contrastive-search continuation for a list of 'continue: …' inputs."""
    import torch

    outs = []
    for pfx in prefixes:
        inputs = tokenizer(pfx, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=GEN_MAX_NEW_TOKENS, penalty_alpha=GEN_PENALTY_ALPHA,
                top_k=GEN_TOP_K, repetition_penalty=GEN_REP_PENALTY,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM, early_stopping=True,
                trust_remote_code=True,  # transformers ≥4.62 fetches contrastive-search via custom_generate
            )
        outs.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return outs


def evaluate_generation(model, tokenizer, cont_eval, device):
    """Generation metrics on held-out continuation eval:

      * ppl   — teacher-forced perplexity of the gold continuation (fluency).
      * f1    — token-F1 of a free-form continuation vs gold (right-direction lexical overlap).
      * morph — morphological self-consistency: run the model's OWN `pos:` on its generation;
                fraction of generated words that receive a valid, length-aligned MorphGNT tag.
                Degenerate/garbled Greek fails to self-tag → low score.
    """
    import torch

    if not cont_eval:
        return {"ppl": float("inf"), "f1": 0.0, "morph": 0.0, "n": 0}
    was_training = model.training
    model.eval()

    # Perplexity (teacher-forced) over PPL_EVAL_N examples, batched.
    ppl_rows = cont_eval[:PPL_EVAL_N]
    total_nll, total_tok = 0.0, 0
    for start in range(0, len(ppl_rows), EVAL_BATCH_SIZE):
        chunk = ppl_rows[start:start + EVAL_BATCH_SIZE]
        enc = tokenizer([r["input_text"] for r in chunk], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LEN).to(device)
        lab = tokenizer([r["target_text"] for r in chunk], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LEN)["input_ids"].to(device)
        lab[lab == 0] = -100
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        labels=lab)
        n_tok = int((lab != -100).sum().item())
        total_nll += float(out.loss.item()) * n_tok
        total_tok += n_tok
    ppl = math.exp(total_nll / max(1, total_tok))

    # Free-generation token-F1 over GEN_EVAL_N examples.
    gen_rows = cont_eval[:GEN_EVAL_N]
    gens = _generate_continuations(model, tokenizer, [r["input_text"] for r in gen_rows], device)
    f1 = sum(_f1_tokens(g, r["target_text"]) for g, r in zip(gens, gen_rows, strict=True)) \
        / max(1, len(gen_rows))

    # Morphological self-consistency over MORPH_EVAL_N generations.
    morph_gens = gens[:MORPH_EVAL_N]
    nonempty = [g for g in morph_gens if g.split()]
    morph = 0.0
    if nonempty:
        tag_preds = _batch_tag_predict(model, tokenizer, nonempty, device, "pos: ", upper=True)
        scores = []
        for g, tags in zip(nonempty, tag_preds, strict=True):
            words = g.split()
            m = min(len(words), len(tags))
            valid = sum(1 for t in tags[:m] if t in VALID_POS_TAGS)
            scores.append(valid / len(words))
        morph = sum(scores) / len(scores)

    if was_training:
        model.train()
    return {"ppl": ppl, "f1": f1, "morph": morph, "n": len(gen_rows)}


def passes_gates(pos_m, lemma_m) -> bool:
    return (pos_m["nt"]["tok"] >= GATE_POS_NT
            and pos_m["classical"]["tok"] >= GATE_POS_CL
            and lemma_m["overall"]["tok"] >= GATE_LEMMA)


def evaluate_all(model, tokenizer, pos_eval, lemma_eval, cont_eval, device):
    """Score every task; return metrics + a selection key that enforces no-regression.

    ``select_key`` = (1, gen_score) once the POS/lemma gates pass, else (0, pos_tok). Any
    gate-passing checkpoint outranks any non-passing one; among passers the generation score
    (token-F1 + self-consistency) decides — i.e. "maximize generation SUBJECT TO no analysis
    regression". Before the gates are met, POS token-acc still tracks early progress.
    """
    pos_m = evaluate_tagging(model, tokenizer, pos_eval, device, "pos: ", upper=True)
    lemma_m = evaluate_tagging(model, tokenizer, lemma_eval, device, "lemma: ", upper=False)
    gen_m = evaluate_generation(model, tokenizer, cont_eval, device)
    gated = passes_gates(pos_m, lemma_m)
    gen_score = gen_m["f1"] + gen_m["morph"]
    select_key = (1.0, gen_score) if gated else (0.0, pos_m["overall"]["tok"])
    return {"pos": pos_m, "lemma": lemma_m, "gen": gen_m, "gated": gated,
            "gen_score": gen_score, "select_key": select_key}


# ── Checkpoint pruning + training loop ────────────────────────────────────────

def _prune_checkpoints(output_dir: Path, keep: int, fingerprint: str) -> None:
    import shutil

    if keep <= 0:
        return
    mine = []
    for d in output_dir.iterdir():
        if not (d.is_dir() and d.name.startswith("step-") and d.name.split("-")[1].isdigit()):
            continue
        fp = d / "run_fp.txt"
        if fp.exists() and fp.read_text().strip() == fingerprint:
            mine.append(d)
    mine.sort(key=lambda d: int(d.name.split("-")[1]))
    for d in mine[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


def _training_loop(model, tokenizer, pools, pos_eval, lemma_eval, cont_eval,
                   output_dir: Path, device: str, volume=None) -> None:
    import hashlib
    import random

    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    rng = np.random.default_rng(42)
    py_rng = random.Random(42)
    torch.manual_seed(42)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)

    warmup_opt = max(1, WARMUP_STEPS // GRAD_ACCUM)
    total_opt = max(warmup_opt + 1, MAX_STEPS // GRAD_ACCUM)

    def lr_lambda(opt_step: int) -> float:
        if opt_step < warmup_opt:
            return opt_step / warmup_opt
        progress = min(1.0, (opt_step - warmup_opt) / (total_opt - warmup_opt))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    pos_subset = _select_eval_subset(pos_eval, EVAL_MAX_PER_SUBSET) if pos_eval else []
    lemma_subset = _select_eval_subset(lemma_eval, EVAL_MAX_PER_SUBSET) if lemma_eval else []
    best_key = (-1.0, -1.0)

    fingerprint = hashlib.sha1(json.dumps({
        "max_steps": MAX_STEPS, "lr": LR, "batch": BATCH_SIZE, "accum": GRAD_ACCUM,
        "warmup": WARMUP_STEPS, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "seq": MAX_SEQ_LEN,
        "stage_a_frac": STAGE_A_FRAC, "stage_a": STAGE_A_WEIGHTS, "stage_b": STAGE_B_WEIGHTS,
        "register": REGISTER_WEIGHTS,
        "rev": 1,  # Hexapla rev 1: continuation task + 512 ctx + r128 + gated eval
        "pools": {t: (sum(len(v) for v in pools[t].values()) if isinstance(pools[t], dict)
                      else len(pools.get(t, []))) for t in _TASK_ORDER},
    }, sort_keys=True).encode()).hexdigest()[:12]

    def _stamp(d: Path) -> None:
        (d / "run_fp.txt").write_text(fingerprint)

    force_fresh = os.environ.get("KF_NO_RESUME") == "1"
    step = 0
    step_dirs = [d for d in output_dir.iterdir()
                 if d.is_dir() and d.name.startswith("step-") and d.name.split("-")[1].isdigit()] \
        if output_dir.exists() else []

    if not force_fresh:
        compatible = sorted(
            ((int(d.name.split("-")[1]), d) for d in step_dirs
             if (d / "run_fp.txt").exists()
             and (d / "run_fp.txt").read_text().strip() == fingerprint
             and int(d.name.split("-")[1]) < MAX_STEPS),
            key=lambda t: t[0],
        )
        if compatible:
            latest_step, latest = compatible[-1]
            adapter_file = latest / "adapter_model.safetensors"
            if adapter_file.exists():
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                set_peft_model_state_dict(model, load_file(str(adapter_file)))
                step = latest_step
                state_file = latest / "training_state.pt"
                if state_file.exists():
                    state = torch.load(str(state_file), map_location=device, weights_only=False)
                    optimizer.load_state_dict(state["optimizer"])
                    scheduler.load_state_dict(state["scheduler"])
                    step = int(state.get("step", latest_step))
                    best_key = tuple(state.get("best_key", (-1.0, -1.0)))
                    try:
                        torch.set_rng_state(state["torch_rng"].cpu())
                        rng.bit_generator.state = state["numpy_rng"]
                        py_rng.setstate(state["python_rng"])
                    except Exception:
                        pass
                    print(f"Resuming {latest.name} (fp={fingerprint}): step={step}, "
                          f"best_key={best_key}, lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"Starting Hexapla training from step {step} → {MAX_STEPS}")
    for t in _TASK_ORDER:
        p = pools.get(t)
        n = sum(len(v) for v in p.values()) if isinstance(p, dict) else len(p or [])
        print(f"    pool {t:12s} {n:,}")
    print(f"  GPU: {device}  |  Stage-A until step {int(STAGE_A_FRAC * MAX_STEPS):,}  |  "
          f"pos-eval {len(pos_subset)}  cont-eval {len(cont_eval)}")
    print("=" * 70)

    model.train()
    accum_loss, valid_steps = 0.0, 0
    optimizer.zero_grad()

    while step < MAX_STEPS:
        batch_examples = sample_balanced_batch(pools, BATCH_SIZE, step, py_rng)
        batch = _collate_batch(batch_examples, tokenizer, rng)
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = out.loss
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = out.loss

        if not math.isfinite(raw_loss.item()):
            print(f"  [skip] non-finite loss at step {step}")
            optimizer.zero_grad()
            step += 1
            continue

        (raw_loss / GRAD_ACCUM).backward()
        accum_loss += raw_loss.item()
        valid_steps += 1

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1

        if step % LOG_STEPS == 0:
            stage = "A" if step < STAGE_A_FRAC * MAX_STEPS else "B"
            print(f"  step {step:>6}/{MAX_STEPS} [{stage}] loss={accum_loss / max(1, valid_steps):.4f}"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}")
            accum_loss, valid_steps = 0.0, 0

        if pos_subset and step % EVAL_STEPS == 0:
            m = evaluate_all(model, tokenizer, pos_subset, lemma_subset, cont_eval, device)
            g = m["gen"]
            print(f"  [eval {step}] POS nt={m['pos']['nt']['tok']:.3f} "
                  f"cl={m['pos']['classical']['tok']:.3f} | lemma={m['lemma']['overall']['tok']:.3f} "
                  f"| ppl={g['ppl']:.2f} f1={g['f1']:.3f} morph={g['morph']:.3f} "
                  f"| gated={m['gated']} gen_score={m['gen_score']:.3f}")
            if m["select_key"] > best_key:
                best_key = m["select_key"]
                model.save_pretrained(str(best_dir))
                (best_dir / "metrics.json").write_text(json.dumps({"step": step, **m}, indent=2))
                _stamp(best_dir)
                print(f"  [best] select_key={best_key} (gated={m['gated']}) → saved {best_dir}")
                if volume is not None:
                    _commit()

        if step % SAVE_STEPS == 0:
            ckpt = output_dir / f"step-{step}"
            model.save_pretrained(str(ckpt))
            _stamp(ckpt)
            torch.save({
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "best_key": best_key, "torch_rng": torch.get_rng_state(),
                "numpy_rng": rng.bit_generator.state, "python_rng": py_rng.getstate(),
            }, str(ckpt / "training_state.pt"))
            _prune_checkpoints(output_dir, CKPT_KEEP, fingerprint)
            print(f"  [checkpoint] {ckpt} (kept last {CKPT_KEEP})")
            if volume is not None:
                _commit()

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    _stamp(final_dir)
    print(f"\nTraining complete. Final: {final_dir}  |  best_key={best_key}")
    if volume is not None:
        _commit()


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(model, tokenizer, input_text: str, task: str = "denoise", device: str = "cpu") -> str:
    import torch

    task_prefixes = {
        "denoise": "", "pos": "pos: ", "lemma": "lemma: ",
        "continuation": "continue: ",
        "synoptic_mk_to_mt": "synoptic mark_to_matt: ",
        "synoptic_mk_to_lk": "synoptic mark_to_luke: ",
    }
    prefix = task_prefixes.get(task, "")
    full_input = prefix + input_text if not input_text.startswith(prefix) else input_text
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)

    if task in ("pos", "lemma"):
        gen_kwargs = {"max_new_tokens": GEN_MAX_NEW_TOKENS, "num_beams": 1, "do_sample": False}
    else:
        gen_kwargs = {
            "max_new_tokens": GEN_MAX_NEW_TOKENS, "penalty_alpha": GEN_PENALTY_ALPHA,
            "top_k": GEN_TOP_K, "repetition_penalty": GEN_REP_PENALTY,
            "no_repeat_ngram_size": GEN_NO_REPEAT_NGRAM, "early_stopping": True,
            "trust_remote_code": True,
        }
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"], **gen_kwargs)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded if decoded.strip() else input_text


# ── Modal entrypoints ─────────────────────────────────────────────────────────

@app.function(gpu=GPU_TYPE, image=_build_image(), volumes=_VOLUMES, timeout=TIMEOUT) \
    if modal is not None else None
def train() -> None:
    """Full Koine-T5-Hexapla training run on Modal GPU."""
    device = "cuda"
    output_dir = Path("/outputs/koine_hexapla")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = "/data/processed/tokens.parquet"
    pericopes_path = "/data/processed/pericopes.parquet"

    print("Phase 1: tokenizer")
    tokenizer = build_tokenizer()
    print(f"Phase 2: model (LoRA r={LORA_R}, seq={MAX_SEQ_LEN})")
    model = load_model_with_lora(tokenizer, device=device)

    print("Phase 3: PROIEL")
    proiel_dir = resolve_proiel_dir(allow_download=True)
    print(f"  PROIEL: {proiel_dir or 'UNAVAILABLE (Gospel-only pos/lemma)'}")

    print("Phase 4: task pools")
    corpus_dir = MAXI_CORPUS_DIR if Path(MAXI_CORPUS_DIR).exists() else ""
    if not corpus_dir:
        print(f"  [warn] {MAXI_CORPUS_DIR} not on volume — falling back to Gospel+PROIEL "
              "denoise (run scripts/prepare_koine_maxi_corpus.py --upload for the full diet)")
    pools, cont_eval = build_task_pools(tokens_path, pericopes_path, proiel_dir, corpus_dir)
    pos_eval, lemma_eval = build_proiel_eval(proiel_dir) if proiel_dir else ([], [])
    print(f"  Eval: pos/lemma={len(pos_eval)}  continuation={len(cont_eval)}")

    print("Phase 5: train")
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME) if modal is not None else None
    _training_loop(model, tokenizer, pools, pos_eval, lemma_eval, cont_eval,
                   output_dir, device, volume=output_vol)
    print("\nDownload:")
    print(f"  modal volume get {OUTPUT_VOLUME} koine_hexapla/best models/koine_hexapla/best")


def _run_demo(adapter_path: str | None = None) -> None:
    """Local A/B demo: GreTa base vs Koine-T5-Hexapla on the failure-mode prompts."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Koine-T5-Hexapla demo on {device}\n")
    tokenizer = build_tokenizer()

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID, local_files_only=True, dtype=torch.bfloat16).to(device)
    if hasattr(base_model.config, "tie_word_embeddings"):
        base_model.config.tie_word_embeddings = False
    base_model.config.vocab_size = len(tokenizer)
    base_model.eval()

    hex_model = None
    if adapter_path and Path(adapter_path).exists():
        hb = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID, local_files_only=True, dtype=torch.bfloat16).to(device)
        if hasattr(hb.config, "tie_word_embeddings"):
            hb.config.tie_word_embeddings = False
        hb.config.vocab_size = len(tokenizer)
        hex_model = PeftModel.from_pretrained(hb, adapter_path, local_files_only=True).to(device)

    test_cases = [
        {"task": "continuation", "name": "Continuation — Lukan opening",
         "text": "καὶ ἐγένετο ἐν ταῖς ἡμέραις ἐκείναις"},
        {"task": "denoise", "name": "Mark 1:1 — span completion",
         "text": "Ἀρχὴ τοῦ <extra_id_0> Ἰησοῦ Χριστοῦ <extra_id_1>."},
        {"task": "pos", "name": "Luke 1:46 — POS", "text": "μεγαλύνει ἡ ψυχή μου τὸν κύριον"},
        {"task": "lemma", "name": "Matthew 6:9 — lemma", "text": "Πάτερ ἡμῶν ὁ ἐν τοῖς οὐρανοῖς"},
    ]
    print("=" * 80)
    for case in test_cases:
        print(f"\n  {case['name']}\n  Input: {case['text']}")
        print(f"  GreTa base:  {generate(base_model, tokenizer, case['text'], case['task'], device)}")
        if hex_model is not None:
            print(f"  Hexapla:     {generate(hex_model, tokenizer, case['text'], case['task'], device)}")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        _run_demo(adapter_path=sys.argv[2] if len(sys.argv) > 2 else "models/koine_hexapla/best")
    else:
        print(__doc__)
        print("\nTrain:  modal run modal/app_koine_hexapla.py::train")
        print("Demo:   python modal/app_koine_hexapla.py demo [adapter_path]")
