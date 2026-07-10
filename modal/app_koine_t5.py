"""Koine-T5: a general-purpose multitask Ancient Greek seq2seq model.

Standalone Modal training script — no imports from the synoptiq package. It trains a
single GreTa+LoRA model on FOUR balanced task pools simultaneously:

  1. denoise  — True T5 span corruption (applied online, on RAW Greek prose)
  2. pos      — part-of-speech tagging  (MorphGNT tagset, seq2seq text-to-text)
  3. lemma    — lemmatization           (seq2seq text-to-text)
  4. synoptic — Mark→Matthew and Mark→Luke style transfer (the small, curated pool)

The `pos`/`lemma`/`denoise` pools are fed by BOTH the Synoptic Gospel corpus AND the
**UD_Ancient_Greek-PROIEL** treebank (New Testament Koine + Herodotus Classical, ~214K
tokens). PROIEL is what cures the POS "task collapse": with only ~401 Gospel examples the
pre-trained language prior dominated and `pos:` produced natural-language prose instead of
tags. Every batch now samples every task pool (§ balanced sampling), so the ~401-example
synoptic pool can never be starved out (catastrophic forgetting), and POS gets tens of
thousands of examples.

Validation uses the PROIEL **dev** set with GREEDY decoding and reports token-level
accuracy + sentence-level Exact Match (EM), split into the Koine-NT and Classical subsets.
The best-EM adapter is saved separately to `best/` for independent deployment.

PROIEL background (verified against the real CoNLL-U, 20180408 release):
  * Composition: ~52–59% Koine NT (Matthew/Mark/Luke/John/Acts/Revelation/Romans…),
    ~41–48% Classical (Herodotus *Histories*). This is the right treebank for a Koine model —
    UD-Perseus has NO New Testament and is Classical poetry only.
  * License CC BY-NC-SA 3.0 (NonCommercial; fine for research training).
  * POS mapping uses the **XPOS** column, NOT UPOS/FEATS. The article ὁ/ἡ/τό is UPOS=DET with
    FEATS `PronType=Dem` (a trap that would mislabel it a demonstrative); its XPOS is `S-`,
    which maps cleanly to MorphGNT `RA`. See PROIEL_XPOS_TO_MORPHGNT below.

Usage:
    # Upload the processed Gospel corpus to the Modal volume (once):
    modal run modal/app_koine_t5.py::upload_corpus

    # (Optional) pre-stage PROIEL onto the Modal data volume for offline runs:
    modal run modal/app_koine_t5.py::upload_proiel

    # Start training (A10G GPU, auto-downloads PROIEL if absent, survives laptop sleep):
    modal run modal/app_koine_t5.py::train

    # Monitor live logs:
    modal app logs koine-t5

    # Download the BEST adapter (model-selected on POS EM) and the final adapter:
    modal volume get koine-t5-outputs koine_t5/best  models/koine_t5/best
    modal volume get koine-t5-outputs koine_t5/final models/koine_t5/final

    # Run local demo / inference:
    python modal/app_koine_t5.py demo
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

try:
    import modal  # type: ignore[import-untyped]
except ImportError:
    modal = None  # type: ignore[assignment]

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_VOLUME   = "synoptiq-data"          # reuse existing volume with the processed parquets
OUTPUT_VOLUME = "koine-t5-outputs"    # separate volume — never overwrites old KoineFormer
GPU_TYPE      = "A10G"
TIMEOUT       = 86_400                   # 24 hours

BASE_MODEL_ID = "bowphs/GreTa"           # T5-base fine-tuned on Ancient Greek

# LoRA / model hyper-parameters
LORA_R          = 32
LORA_ALPHA      = 64
LORA_DROPOUT    = 0.05
MAX_SEQ_LEN     = 256   # 512 caused OOM on A10G; 256 gives same quality for Koine verse length
BATCH_SIZE      = 4     # 1 example per task pool per micro-batch (see sample_balanced_batch)
GRAD_ACCUM      = 8     # effective batch = 32 (8 per task per optimizer step)
LR              = 1e-4

# ── Hyper-parameter recalibration for the ~214K-token PROIEL corpus ──────────────
# The old MAX_STEPS=30_000 was calibrated for ~401 examples. The dominant new pools
# (pos / lemma / denoise) each hold ~15K PROIEL sentences. With balanced sampling the
# optimizer sees 8 examples PER TASK per step (effective batch 32 split across 4 pools),
# so one epoch over the ~15K POS pool ≈ 15000/8 ≈ 1875 steps. We target ~6–7 epochs:
#   12_000 steps × 8 pos-examples/step ÷ ~15K pool ≈ 6.4 epochs over POS/lemma/denoise.
# That is ample for a low-capacity LoRA (r=32) to learn the abstract tag-output format and
# unlearn the prose-collapse, without over-fitting. Warmup is a healthy ~5% of training.
MAX_STEPS           = 12_000
WARMUP_STEPS        = 600     # ~5% of MAX_STEPS
SAVE_STEPS          = 1_000   # checkpoint cadence
EVAL_STEPS          = 1_000   # POS-EM validation + best-checkpoint selection cadence
LOG_STEPS           = 100

# POS/eval example shaping
MAX_POS_WORDS       = 60      # cap PROIEL sentence length so pos/lemma input↔target stay aligned
EVAL_MAX_PER_SUBSET = 250     # dev sentences per subset (NT / Classical) per eval — keeps eval fast

# Span-corruption hyper-parameters (T5 paper §3.1)
NOISE_DENSITY         = 0.15
MEAN_NOISE_SPAN_LEN   = 3.0

# Contrastive-search generation defaults (used for denoise/synoptic demo inference)
GEN_PENALTY_ALPHA  = 0.6
GEN_TOP_K          = 4
GEN_MAX_NEW_TOKENS = 256
GEN_REP_PENALTY    = 1.25
GEN_NO_REPEAT_NGRAM = 3

# ── PROIEL treebank configuration ────────────────────────────────────────────────
# Paths are configurable for BOTH the remote Modal volume and a local checkout, with a
# best-effort auto-download fallback. Precedence: $PROIEL_DIR → remote volume → local dir
# → download. If nothing resolves, training gracefully falls back to Synoptic-only.
PROIEL_SPLITS     = ("train", "dev", "test")
PROIEL_URL_TMPL   = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Ancient_Greek-PROIEL/master/grc_proiel-ud-{split}.conllu"
)
PROIEL_REMOTE_DIR = "/data/proiel"      # optional pre-upload target on the Modal data volume
PROIEL_LOCAL_DIR  = "data/raw/proiel"   # local checkout / offline
PROIEL_CACHE_DIR  = "/tmp/proiel"       # ephemeral download target (Modal container scratch)


def _proiel_file(dir_: str, split: str) -> Path:
    return Path(dir_) / f"grc_proiel-ud-{split}.conllu"


def _proiel_files_present(dir_: str | None) -> bool:
    """True iff all three split files exist and are non-empty under `dir_`."""
    if not dir_:
        return False
    return all(
        (p := _proiel_file(dir_, s)).exists() and p.stat().st_size > 0
        for s in PROIEL_SPLITS
    )


def _download_proiel(dest: str) -> bool:
    """Best-effort download of the three PROIEL CoNLL-U splits into `dest`. Returns success."""
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
    except Exception as exc:  # network down, offline, GitHub hiccup — non-fatal
        print(f"  [proiel] download failed ({type(exc).__name__}: {exc})")
        return False


def resolve_proiel_dir(allow_download: bool = True) -> str | None:
    """Locate a directory holding the three PROIEL splits, downloading if needed.

    Returns the directory path, or None if PROIEL is unavailable (caller then falls
    back to Synoptic-only training — see `train`).
    """
    for candidate in (os.environ.get("PROIEL_DIR"), PROIEL_REMOTE_DIR, PROIEL_LOCAL_DIR):
        if _proiel_files_present(candidate):
            return candidate
    if allow_download:
        for target in (PROIEL_LOCAL_DIR, PROIEL_CACHE_DIR):
            if _download_proiel(target):
                return target
    return None


# ── PROIEL XPOS → MorphGNT POS mapping ───────────────────────────────────────────
# The rest of the pipeline (and the Gospel corpus `pos` field) uses the 13-code MorphGNT
# tagset. PROIEL's XPOS column carries an equally granular tagset that maps onto it almost
# 1:1 — decisively better than UPOS (which collapses RA/RD/RI/RP/RR into DET/PRON) or FEATS
# (whose PronType=Dem is set even on the plain article). Every mapping below was validated
# against how the Gospel corpus itself tags the underlying lemma.
PROIEL_XPOS_TO_MORPHGNT: dict[str, str] = {
    # verbs (finite / infinite / participle) + copula εἰμί (PROIEL UPOS=AUX)
    "V-": "V-",
    # nouns
    "Nb": "N-",   # common noun
    "Ne": "N-",   # proper noun  (MorphGNT has no PROPN category → N-)
    "F-": "N-",   # foreign word (very rare; default to noun)
    # article — the decisive case: XPOS `S-` cleanly separates it from demonstratives
    "S-": "RA",   # definite article ὁ/ἡ/τό   [FEATS wrongly says PronType=Dem — do not use it]
    # adjectives & numerals (numerals & possessives are adjectival in MorphGNT)
    "A-": "A-",   # adjective
    "Ma": "A-",   # cardinal numeral (εἷς, δύο, δώδεκα …)
    "Mo": "A-",   # ordinal numeral  (πρῶτος, δεύτερος …)
    "Ps": "A-",   # possessive adjective (ἐμός, σός) — corpus tags these A-
    # adverbs
    "Df": "D-",   # adverb
    "Du": "D-",   # interrogative adverb (πῶς, ποῦ, πότε) — corpus D-
    # prepositions / adpositions
    "R-": "P-",
    # conjunctions: coordinating, subordinating, and correlative subordinators
    "C-": "C-",   # coordinating conjunction (καί, δέ …)
    "G-": "C-",   # subjunction (ὅτι, ἵνα, εἰ …)
    "Dq": "C-",   # relative/correlative "adverb" (ὡς, καθώς, ὅταν, ὅπου) — corpus tags C-
    # pronouns — the subtypes MorphGNT distinguishes
    "Pp": "RP",   # personal pronoun (ἐγώ, σύ, αὐτός)
    "Pk": "RP",   # reflexive personal pronoun (ἑαυτοῦ, σεαυτοῦ) — corpus RP
    "Pc": "RP",   # reciprocal pronoun (ἀλλήλων) — corpus RP
    "Pr": "RR",   # relative pronoun (ὅς, ὅστις, ὅσος)
    "Pd": "RD",   # demonstrative pronoun (οὗτος, ἐκεῖνος, ὅδε)
    "Pi": "RI",   # interrogative pronoun (τίς;)
    # interjection
    "I-": "I-",
    # NOTE: "Px" (indefinite/quantifier) is resolved by map_xpos_to_morphgnt() below.
}
FALLBACK_MORPHGNT = "X-"           # particle/other bucket for any unseen XPOS
INDEF_PRONOUN_LEMMAS = {"τις"}     # accent-stripped; the indefinite τὶς → RI (else quantifier → A-)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    ).lower()


def map_xpos_to_morphgnt(xpos: str, lemma: str) -> str:
    """Map a PROIEL XPOS tag (+ lemma) to the MorphGNT POS code the pipeline expects.

    `Px` is the one genuinely split class: PROIEL lumps the indefinite pronoun τις together
    with quantifiers (πᾶς, οὐδείς, ἄλλος, ἕκαστος). The Gospel corpus tags τις as `RI` but
    the quantifiers as `A-`, so we key on the lemma to reproduce that distinction.
    """
    if xpos == "Px":
        return "RI" if _strip_accents(lemma) in INDEF_PRONOUN_LEMMAS else "A-"
    return PROIEL_XPOS_TO_MORPHGNT.get(xpos, FALLBACK_MORPHGNT)


# ── Modal image ────────────────────────────────────────────────────────────────

_REQUIREMENTS = [
    "torch>=2.6.0",
    "transformers>=4.51.0",
    "peft>=0.15.0",
    "safetensors>=0.4.0",
    "sentencepiece>=0.2.0",
    "datasets>=3.3.0",
    "accelerate>=1.3.0",
    "pandas>=2.2",
    "pyarrow>=16.0",
    "tqdm>=4.67",
    "numpy>=1.26",
]


def _build_image() -> Any:
    if modal is None:
        raise RuntimeError("Modal not installed")
    image = modal.Image.debian_slim(python_version="3.12")
    for req in _REQUIREMENTS:
        image = image.pip_install(req)
    return image


# ── Modal App ──────────────────────────────────────────────────────────────────

app = modal.App("koine-t5") if modal is not None else None

_VOLUMES = {
    "/data":    modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
} if modal is not None else {}


def _commit() -> None:
    if modal is not None:
        modal.Volume.from_name(OUTPUT_VOLUME).commit()


# ── Phase 1: Bulletproof Tokenizer Setup ──────────────────────────────────────

def build_tokenizer():
    """Load the Koine-T5 tokenizer (bowphs/GreTa) and register the 100 T5 sentinel tokens.

    bowphs/GreTa already ships with 32103 embedding slots in the model weights
    but only 32003 tokens in the tokenizer vocabulary. By adding exactly 100
    sentinel tokens (<extra_id_0> … <extra_id_99>), we map them to the existing
    pre-trained ghost slots (indices 32003–32102) WITHOUT resizing the model
    embeddings and WITHOUT introducing any random-weight noise.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # Bind existing pad/eos — do NOT add a new [PAD] token (breaks decoder start id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"   # id 0 in GreTa
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"    # id 1 in GreTa

    # Register the 100 T5 sentinel tokens into the ghost slots
    sentinels = [f"<extra_id_{i}>" for i in range(100)]
    tokenizer.add_special_tokens({"additional_special_tokens": sentinels})

    # Sanity-check: the ghost slots must land exactly here
    assert tokenizer.convert_tokens_to_ids("<extra_id_0>")  == 32003, \
        "Sentinel ID mismatch — vocab size changed unexpectedly"
    assert tokenizer.convert_tokens_to_ids("<extra_id_99>") == 32102, \
        "Sentinel ID mismatch — vocab size changed unexpectedly"

    return tokenizer


def load_model_with_lora(tokenizer, device: str = "cpu"):
    """Load the Koine-T5 base (bowphs/GreTa) in bfloat16 with maximalist LoRA.

    bfloat16 halves VRAM vs float32 (same exponent range so no overflow risk),
    which is the primary fix for A10G OOM at batch=4, seq_len=256.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from peft import LoraConfig, TaskType, get_peft_model

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Use 'dtype' (new API) instead of deprecated 'torch_dtype'
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.bfloat16
    )

    # Silence the tied-weights warning (known GreTa quirk)
    if hasattr(base.config, "tie_word_embeddings"):
        base.config.tie_word_embeddings = False

    # Tokenizer now has 32103 tokens which matches the embedding table exactly.
    # Update vocab_size so generate() uses the correct ceiling.
    base.config.vocab_size = len(tokenizer)  # 32103

    # Maximalist LoRA on encoder + decoder (attention + FFN).
    # Includes both T5-v1.0 (wi/wo) and T5-v1.1 (wi_0/wi_1) FFN names.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q", "k", "v", "o",    # Attention: encoder + decoder self + cross-attn
            "wi", "wo",             # T5 v1.0 FFN
            "wi_0", "wi_1",         # T5 v1.1 FFN (Gated Linear Unit)
        ],
    )

    model = get_peft_model(base, lora_config)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Koine-T5 LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# ── Phase 2: True T5 Span Corruption ──────────────────────────────────────────

def _random_spans_noise_mask(length: int, noise_density: float, mean_span_len: float,
                              rng) -> list[bool]:
    """Return a boolean mask for which token positions to corrupt (True = corrupt).

    Follows the T5 paper §3.1 exactly:
      - Sample the number of noise tokens: num_noise = round(length * noise_density)
      - Arrange them as spans drawn from a geometric distribution with mean mean_span_len
    """
    num_noise = max(1, round(length * noise_density))
    num_noise = min(num_noise, length - 1)

    # Draw span lengths from geometric distribution
    span_lengths: list[int] = []
    remaining = num_noise
    while remaining > 0:
        span_len = max(1, int(rng.geometric(p=1.0 / mean_span_len)))
        span_len = min(span_len, remaining)
        span_lengths.append(span_len)
        remaining -= span_len

    # Place spans at random non-overlapping positions
    num_nonnoise = length - num_noise
    # interleave spans with gaps
    num_spans = len(span_lengths)
    num_gaps  = num_spans + 1
    gap_lengths = _split_into_k_parts(num_nonnoise, num_gaps, rng)

    mask = []
    for g, gap in enumerate(gap_lengths):
        mask.extend([False] * gap)
        if g < len(span_lengths):
            mask.extend([True] * span_lengths[g])

    return mask[:length]


def _split_into_k_parts(total: int, k: int, rng) -> list[int]:
    """Split `total` into `k` non-negative integer parts via sorted uniform samples."""
    if k == 1:
        return [total]
    cuts = sorted(rng.integers(0, total + 1, size=k - 1).tolist())
    cuts = [0] + cuts + [total]
    return [cuts[i + 1] - cuts[i] for i in range(k)]


def apply_span_corruption(
    input_ids: list[int],
    tokenizer,
    rng,
    noise_density: float = NOISE_DENSITY,
    mean_span_len: float = MEAN_NOISE_SPAN_LEN,
) -> tuple[list[int], list[int]]:
    """Apply T5 span corruption and return (corrupted_input_ids, target_ids).

    Example (tokenized):
      input_ids  = [Ἀρχὴ, τοῦ, εὐαγγελίου, Ἰησοῦ, Χριστοῦ]
      corrupted  = [Ἀρχὴ, <extra_id_0>, Ἰησοῦ, <extra_id_1>]
      target     = [<extra_id_0>, τοῦ, εὐαγγελίου, <extra_id_1>, Χριστοῦ, </s>]
    """
    mask = _random_spans_noise_mask(len(input_ids), noise_density, mean_span_len, rng)

    corrupted: list[int] = []
    target: list[int]    = []
    sentinel_idx = 0
    in_noise_span = False

    for token_id, is_noise in zip(input_ids, mask):
        if is_noise:
            if not in_noise_span:
                # Start of a new noise span — emit sentinel into both
                sentinel_id = tokenizer.convert_tokens_to_ids(f"<extra_id_{sentinel_idx}>")
                corrupted.append(sentinel_id)
                target.append(sentinel_id)
                sentinel_idx += 1
                in_noise_span = True
            target.append(token_id)  # real token goes only into target
        else:
            if in_noise_span:
                in_noise_span = False
            corrupted.append(token_id)

    target.append(tokenizer.eos_token_id)  # terminate target with </s>
    return corrupted, target


# ── Phase 3a: PROIEL CoNLL-U loading ──────────────────────────────────────────

def iter_conllu_sentences(path: str | Path):
    """Yield (source, tokens) per sentence from a CoNLL-U file.

    `source` is the `# source = …` metadata; `tokens` is a list of (form, lemma, xpos).
    Multiword-token ranges (id "n-m") and empty nodes (id "n.m") are skipped.
    """
    source = ""
    toks: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
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
                toks.append((cols[1], cols[2], cols[4]))  # FORM, LEMMA, XPOS
        if toks:
            yield source, toks


def _is_nt_source(source: str) -> bool:
    """True for the Koine New Testament portion; False for Classical (Herodotus)."""
    # Every PROIEL NT sentence's `# source` begins "The Greek New Testament, <Book> <ch>";
    # the only Classical source is "Histories".
    return "new testament" in source.lower()


def build_proiel_training(proiel_dir: str) -> tuple[list[dict], list[dict], list[str]]:
    """Build (pos_examples, lemma_examples, raw_texts) from the PROIEL TRAIN split.

    Sentences longer than MAX_POS_WORDS are excluded from pos/lemma (to keep input↔target
    token counts inside MAX_SEQ_LEN) but still contribute their raw Greek to the denoise pool.
    """
    pos_examples: list[dict] = []
    lemma_examples: list[dict] = []
    raw_texts: list[str] = []

    path = _proiel_file(proiel_dir, "train")
    if not path.exists():
        return pos_examples, lemma_examples, raw_texts

    n_nt = n_cl = 0
    for source, toks in iter_conllu_sentences(path):
        forms = [t[0] for t in toks]
        text = " ".join(forms)
        if len(toks) >= 2:
            raw_texts.append(text)  # raw Greek prose → online span corruption
        if 2 <= len(toks) <= MAX_POS_WORDS:
            tags = [map_xpos_to_morphgnt(t[2], t[1]) for t in toks]
            lemmas = [t[1] for t in toks]
            pos_examples.append({
                "task": "pos",
                "input_text": f"pos: {text}",
                "target_text": " ".join(tags),
            })
            lemma_examples.append({
                "task": "lemma",
                "input_text": f"lemma: {text}",
                "target_text": " ".join(lemmas),
            })
            if _is_nt_source(source):
                n_nt += 1
            else:
                n_cl += 1

    print(f"  PROIEL train: {len(pos_examples):,} pos/lemma examples "
          f"({n_nt:,} NT + {n_cl:,} Classical), {len(raw_texts):,} denoise texts")
    return pos_examples, lemma_examples, raw_texts


def build_proiel_eval(proiel_dir: str) -> list[dict]:
    """Build POS eval records from the PROIEL DEV split: [{text, gold(list[str]), subset}]."""
    records: list[dict] = []
    path = _proiel_file(proiel_dir, "dev")
    if not path.exists():
        return records
    for source, toks in iter_conllu_sentences(path):
        if not (2 <= len(toks) <= MAX_POS_WORDS):
            continue
        records.append({
            "text": " ".join(t[0] for t in toks),
            "gold": [map_xpos_to_morphgnt(t[2], t[1]) for t in toks],
            "subset": "nt" if _is_nt_source(source) else "classical",
        })
    return records


# ── Phase 3b: Gospel corpus loading ───────────────────────────────────────────

def load_synoptic_pairs(tokens_path: str, pericopes_path: str) -> tuple[list[dict], list[str]]:
    """Load parallel Synoptic pericopes from the processed corpus parquets.

    Returns (examples, raw_texts):
      * examples  — instruction dicts (task, input_text, target_text) covering
                    pos / lemma / synoptic_mk_to_mt / synoptic_mk_to_lk
      * raw_texts — the raw Greek surface text of every book's pericope, for the
                    denoise pool (Problem 4: denoise on RAW prose, never on POS/lemma strings)
    """
    import pandas as pd  # noqa: F401  (kept explicit for the Modal image)

    tokens_df   = pd.read_parquet(tokens_path)
    _pericopes_df = pd.read_parquet(pericopes_path)

    # Build a dict: (pericope_id, book) -> list of token rows in order
    from collections import defaultdict
    pericope_book_tokens: dict[tuple, list] = defaultdict(list)
    for row in tokens_df.itertuples():
        pid  = getattr(row, "pericope_id", None)
        book = getattr(row, "book", None)
        if pid is not None and book is not None:
            pericope_book_tokens[(pid, book)].append(row)

    examples: list[dict] = []
    raw_texts: list[str] = []

    def tokens_to_text(rows) -> str:
        return " ".join(str(getattr(r, "text", "")) for r in rows)

    def tokens_to_pos(rows) -> str:
        return " ".join(str(getattr(r, "pos", "?")) for r in rows)

    def tokens_to_lemma(rows) -> str:
        return " ".join(str(getattr(r, "lemma", getattr(r, "text", ""))) for r in rows)

    # Identify full-triple pericopes (Mark present)
    all_pids = {pid for (pid, _) in pericope_book_tokens}
    for pid in all_pids:
        if (pid, "Mark") not in pericope_book_tokens:
            continue
        mk_rows = pericope_book_tokens[(pid, "Mark")]
        mk_text = tokens_to_text(mk_rows)

        # --- pos + lemma + raw text (all three books) ---
        for book in ("Mark", "Matthew", "Luke"):
            if (pid, book) not in pericope_book_tokens:
                continue
            r = pericope_book_tokens[(pid, book)]
            text = tokens_to_text(r)
            raw_texts.append(text)
            examples.append({
                "task": "pos",
                "input_text": f"pos: {text}",
                "target_text": tokens_to_pos(r),
            })
            examples.append({
                "task": "lemma",
                "input_text": f"lemma: {text}",
                "target_text": tokens_to_lemma(r),
            })

        # --- synoptic style transfer (Mark → Matthew / Luke) ---
        if (pid, "Matthew") in pericope_book_tokens:
            examples.append({
                "task": "synoptic_mk_to_mt",
                "input_text": f"synoptic mark_to_matt: {mk_text}",
                "target_text": tokens_to_text(pericope_book_tokens[(pid, "Matthew")]),
            })
        if (pid, "Luke") in pericope_book_tokens:
            examples.append({
                "task": "synoptic_mk_to_lk",
                "input_text": f"synoptic mark_to_luke: {mk_text}",
                "target_text": tokens_to_text(pericope_book_tokens[(pid, "Luke")]),
            })

    print(f"  Gospel corpus: {len(examples)} instruction examples, {len(raw_texts)} denoise texts")
    return examples, raw_texts


def build_task_pools(tokens_path: str, pericopes_path: str,
                     proiel_dir: str | None) -> dict[str, list[dict]]:
    """Assemble the FOUR balanced task pools (denoise / pos / lemma / synoptic).

    Gospel corpus feeds all four; PROIEL feeds pos / lemma / denoise. The synoptic pool is
    Gospel-only and tiny (~401) — it is protected at sample time by balanced upsampling.
    """
    corpus_examples, corpus_raw = load_synoptic_pairs(tokens_path, pericopes_path)

    pos_pool      = [e for e in corpus_examples if e["task"] == "pos"]
    lemma_pool    = [e for e in corpus_examples if e["task"] == "lemma"]
    synoptic_pool = [e for e in corpus_examples if e["task"].startswith("synoptic")]
    denoise_texts = list(corpus_raw)

    if proiel_dir:
        p_pos, p_lemma, p_raw = build_proiel_training(proiel_dir)
        pos_pool   += p_pos
        lemma_pool += p_lemma
        denoise_texts += p_raw
    else:
        print("  [warn] PROIEL unavailable — pos/lemma/denoise from Gospel corpus only")

    denoise_pool = [{"task": "denoise", "raw_text": t} for t in denoise_texts]

    return {
        "denoise":  denoise_pool,
        "pos":      pos_pool,
        "lemma":    lemma_pool,
        "synoptic": synoptic_pool,
    }


# ── Phase 4: Balanced multi-task sampling ─────────────────────────────────────

# Fixed task order → deterministic round-robin over the batch slots.
_TASK_ORDER = ("pos", "lemma", "synoptic", "denoise")


def sample_balanced_batch(pools: dict[str, list[dict]], batch_size: int, rng_py) -> list[dict]:
    """Draw a batch that contains EVERY non-empty task pool, regardless of pool sizes.

    Batch slots are assigned round-robin across the (non-empty) task pools and each slot is
    filled by sampling WITH REPLACEMENT (`random.choice`). With-replacement is the correct
    primitive here: a tiny pool (synoptic, ~401) must be able to fill its slot in every one of
    hundreds of thousands of batches — `random.sample` (without replacement) draws distinct
    items and cannot upsample a pool beyond its own size, so a minority task would be starved
    and catastrophically forgotten. `random.choice` gives each item equal probability on every
    draw (an unbiased estimate of the task's gradient) and never runs out.
    """
    active = [t for t in _TASK_ORDER if pools.get(t)]
    if not active:
        return []
    batch: list[dict] = []
    for i in range(batch_size):
        task = active[i % len(active)]
        batch.append(rng_py.choice(pools[task]))
    return batch


def _collate_batch(examples: list[dict], tokenizer, rng, max_len: int = MAX_SEQ_LEN):
    """Build a training batch from a list of task examples.

    For 'denoise' the corruption is applied online on the RAW Greek text (so every batch sees
    a fresh masking). All other tasks tokenize input/target directly.
    """
    import torch

    batch_input_ids:   list[torch.Tensor] = []
    batch_attn_masks:  list[torch.Tensor] = []
    batch_labels:      list[torch.Tensor] = []

    # GreTa's SentencePiece backend may leave pad_token_id unset on the object even after
    # tokenizer.pad_token = "<pad>". Hard-code id=0 (verified: <pad> is always id 0 in GreTa).
    PAD_ID = 0

    for ex in examples:
        task = ex["task"]

        if task == "denoise":
            # True T5 span corruption — applied online on raw Greek prose
            raw = ex.get("raw_text") or ex.get("input_text", "")
            raw_ids = tokenizer.encode(raw, add_special_tokens=False,
                                       max_length=max_len - 10, truncation=True)
            if len(raw_ids) < 8:
                continue
            corrupted, target = apply_span_corruption(raw_ids, tokenizer, rng)
            corrupted = corrupted[:max_len]
            target    = target[:max_len]
            input_ids = torch.tensor(corrupted, dtype=torch.long)
            label_ids = torch.tensor(target,    dtype=torch.long)
        else:
            # Seq2seq instruction tasks (POS, lemma, synoptic)
            src = tokenizer(ex["input_text"],  max_length=max_len, truncation=True,
                            return_tensors="pt")
            tgt = tokenizer(ex["target_text"], max_length=max_len, truncation=True,
                            return_tensors="pt")
            input_ids = src["input_ids"][0]
            label_ids = tgt["input_ids"][0].clone()

        # Mask pad positions (id=0) in labels so they don't contribute to loss
        label_ids[label_ids == PAD_ID] = -100

        # Guard: skip any example where ALL labels are masked — would produce NaN loss
        if (label_ids != -100).sum() == 0:
            continue

        batch_input_ids.append(input_ids)
        batch_attn_masks.append(torch.ones(len(input_ids), dtype=torch.long))
        batch_labels.append(label_ids)

    if not batch_input_ids:
        return None

    # Pad to max length in batch
    def _pad(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len_in_batch = max(t.size(0) for t in tensors)
        out = []
        for t in tensors:
            p = max_len_in_batch - t.size(0)
            out.append(torch.nn.functional.pad(t, (0, p), value=pad_value))
        return torch.stack(out)

    return {
        "input_ids":      _pad(batch_input_ids,  PAD_ID),
        "attention_mask": _pad(batch_attn_masks, 0),
        "labels":         _pad(batch_labels,    -100),
    }


# ── Phase 4b: POS Exact-Match evaluation (greedy decoding) ────────────────────

def _greedy_pos_predict(model, tokenizer, greek_text: str, device: str) -> list[str]:
    """Greedy-decode `pos: <text>` → list of predicted tag strings.

    Greedy (num_beams=1, do_sample=False) with NO repetition penalties: POS tag sequences
    legitimately repeat (e.g. `RA N- RA N-`), and contrastive search / no_repeat_ngram would
    corrupt them.
    """
    import torch

    n_words = len(greek_text.split())
    inputs = tokenizer("pos: " + greek_text, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=min(200, n_words * 4 + 10),
            num_beams=1,
            do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).split()


def evaluate_pos_em(model, tokenizer, records: list[dict], device: str) -> dict[str, dict]:
    """Compute token-level accuracy + sentence-level Exact Match on PROIEL dev.

    Returns {"nt": {...}, "classical": {...}, "overall": {...}} where each inner dict is
    {"tok": token_accuracy, "em": exact_match, "n": n_sentences}.
    """
    was_training = model.training
    model.eval()

    # [tok_correct, tok_total, em_count, n_sentences] per subset
    agg = {"nt": [0, 0, 0, 0], "classical": [0, 0, 0, 0]}
    for r in records:
        pred = _greedy_pos_predict(model, tokenizer, r["text"], device)
        gold = r["gold"]
        tok_correct = sum(1 for g, p in zip(gold, pred) if g == p)
        a = agg[r["subset"]]
        a[0] += tok_correct
        a[1] += len(gold)
        a[2] += int(pred == gold)
        a[3] += 1

    if was_training:
        model.train()

    def pack(a: list[int]) -> dict:
        return {"tok": a[0] / max(1, a[1]), "em": a[2] / max(1, a[3]), "n": a[3]}

    nt, cl = agg["nt"], agg["classical"]
    overall = [nt[0] + cl[0], nt[1] + cl[1], nt[2] + cl[2], nt[3] + cl[3]]
    return {"nt": pack(nt), "classical": pack(cl), "overall": pack(overall)}


def _select_eval_subset(records: list[dict], per_subset: int, seed: int = 1234) -> list[dict]:
    """Deterministically pick up to `per_subset` sentences from each subset (stable across evals)."""
    import random as _random
    rng = _random.Random(seed)
    out: list[dict] = []
    for subset in ("nt", "classical"):
        pool = [r for r in records if r["subset"] == subset]
        rng.shuffle(pool)
        out.extend(pool[:per_subset])
    return out


# ── Training loop ──────────────────────────────────────────────────────────────

def _training_loop(
    model,
    tokenizer,
    pools: dict[str, list[dict]],
    eval_records: list[dict],
    output_dir: Path,
    device: str,
    volume=None,
) -> None:
    """Main training loop: balanced 4-task sampling + POS-EM eval + best-checkpoint selection."""
    import hashlib
    import json
    import math
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

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    # Linear warmup + cosine decay to zero
    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    # bfloat16 needs no GradScaler (that was for float16 underflow); bf16 autocast has the
    # same memory benefit with no overflow risk (float16 autocast previously caused NaN losses).
    USE_BF16 = True

    # Deterministic eval subset (stable across evals → comparable EM for model selection)
    eval_subset = _select_eval_subset(eval_records, EVAL_MAX_PER_SUBSET) if eval_records else []
    best_em = -1.0

    # Run fingerprint: a short hash of the config that defines "the same run". Auto-resume ONLY
    # from checkpoints stamped with this exact fingerprint, so a stale/foreign checkpoint (e.g.
    # the old 30k-step, pre-PROIEL adapter sitting on the persistent volume) can never hijack the
    # run or — worse — silently no-op it because its step number already exceeds MAX_STEPS.
    fingerprint = hashlib.sha1(json.dumps({
        "max_steps": MAX_STEPS, "lr": LR, "batch": BATCH_SIZE, "accum": GRAD_ACCUM,
        "warmup": WARMUP_STEPS, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "seq": MAX_SEQ_LEN,
        "pools": {t: len(pools.get(t, [])) for t in _TASK_ORDER},
    }, sort_keys=True).encode()).hexdigest()[:12]

    def _stamp(d: Path) -> None:
        (d / "run_fp.txt").write_text(fingerprint)

    force_fresh = os.environ.get("KF_NO_RESUME") == "1"
    step = 0
    step_dirs = [d for d in output_dir.iterdir()
                 if d.is_dir() and d.name.startswith("step-") and d.name.split("-")[1].isdigit()] \
        if output_dir.exists() else []

    if not force_fresh:
        # compatible = same fingerprint AND still short of MAX_STEPS (i.e. genuine unfinished work)
        compatible = sorted(
            ((int(d.name.split("-")[1]), d) for d in step_dirs
             if (d / "run_fp.txt").exists() and (d / "run_fp.txt").read_text().strip() == fingerprint
             and int(d.name.split("-")[1]) < MAX_STEPS),
            key=lambda t: t[0],
        )
        if compatible:
            latest_step, latest = compatible[-1]
            adapter_file = latest / "adapter_model.safetensors"
            if adapter_file.exists():
                # Load LoRA weights INTO the existing "default" adapter (avoid PeftModel.load_adapter,
                # which registers a *new* named adapter and collides on "default").
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                set_peft_model_state_dict(model, load_file(str(adapter_file)))
                step = latest_step
                print(f"Resuming compatible checkpoint: {latest.name}  (fp={fingerprint})")
                if (best_dir / "metrics.json").exists():
                    try:
                        best_em = json.loads(
                            (best_dir / "metrics.json").read_text()
                        )["overall"]["em"]
                        print(f"  Prior best EM (resumed): {best_em:.4f}")
                    except Exception:
                        best_em = -1.0

    # Surface (but do not touch) checkpoints we are deliberately ignoring
    if step == 0:
        foreign = [d.name for d in step_dirs
                   if not (d / "run_fp.txt").exists()
                   or (d / "run_fp.txt").read_text().strip() != fingerprint]
        if foreign:
            shown = ", ".join(sorted(foreign)[:6]) + ("…" if len(foreign) > 6 else "")
            print(f"Fresh start (fp={fingerprint}); ignoring {len(foreign)} checkpoint(s) from a "
                  f"different run config: {shown}")

    # Report pool sizes + the effective-epoch math (documents Problem 6)
    print(f"Starting training from step {step} → {MAX_STEPS}")
    print("  Task pools: " + " | ".join(f"{t}={len(pools.get(t, [])):,}" for t in _TASK_ORDER))
    per_task_per_step = GRAD_ACCUM * max(1, BATCH_SIZE // len([t for t in _TASK_ORDER if pools.get(t)]))
    largest = max((len(pools.get(t, [])) for t in ("pos", "lemma", "denoise")), default=1)
    if largest:
        print(f"  ≈{MAX_STEPS * per_task_per_step / largest:.1f} effective epochs over the largest pool "
              f"({largest:,} examples; ~{per_task_per_step} examples/task/step)")
    print(f"  GPU: {device}  |  bfloat16: {USE_BF16}  |  eval sentences/round: {len(eval_subset)}")
    print("=" * 70)

    model.train()
    accum_loss  = 0.0
    valid_steps = 0
    optimizer.zero_grad()

    while step < MAX_STEPS:
        batch_examples = sample_balanced_batch(pools, BATCH_SIZE, py_rng)
        batch = _collate_batch(batch_examples, tokenizer, rng)
        if batch is None:
            continue

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        if USE_BF16 and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = out.loss
        else:
            out      = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = out.loss

        # Guard: skip NaN/Inf batches without crashing training
        if not math.isfinite(raw_loss.item()):
            print(f"  [skip] non-finite loss={raw_loss.item():.4f} at step {step}")
            optimizer.zero_grad()
            step += 1
            continue

        loss = raw_loss / GRAD_ACCUM
        loss.backward()

        accum_loss  += raw_loss.item()
        valid_steps += 1

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()   # after optimizer — prevents the PyTorch ordering warning
            optimizer.zero_grad()

        step += 1

        if step % LOG_STEPS == 0:
            avg_loss = accum_loss / max(1, valid_steps)
            accum_loss  = 0.0
            valid_steps = 0
            print(f"  step {step:>6}/{MAX_STEPS}  loss={avg_loss:.4f}"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}")

        # POS-EM validation + best-checkpoint selection
        if eval_subset and step % EVAL_STEPS == 0:
            m = evaluate_pos_em(model, tokenizer, eval_subset, device)
            print(f"  [eval step {step}] POS EM  overall={m['overall']['em']:.3f} "
                  f"(tok={m['overall']['tok']:.3f})  |  NT em={m['nt']['em']:.3f} "
                  f"tok={m['nt']['tok']:.3f} (n={m['nt']['n']})  |  "
                  f"Classical em={m['classical']['em']:.3f} tok={m['classical']['tok']:.3f} "
                  f"(n={m['classical']['n']})")
            if m["overall"]["em"] > best_em:
                best_em = m["overall"]["em"]
                model.save_pretrained(str(best_dir))
                (best_dir / "metrics.json").write_text(json.dumps({"step": step, **m}, indent=2))
                _stamp(best_dir)
                print(f"  [best] new best EM={best_em:.4f} → saved {best_dir}")
                if volume is not None:
                    _commit()

        if step % SAVE_STEPS == 0:
            ckpt = output_dir / f"step-{step}"
            model.save_pretrained(str(ckpt))
            _stamp(ckpt)
            print(f"  [checkpoint] saved → {ckpt}")
            if volume is not None:
                _commit()

    # Final save
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    _stamp(final_dir)
    print(f"\nTraining complete. Final adapters: {final_dir}")
    if best_em >= 0:
        print(f"Best POS-EM adapter (EM={best_em:.4f}): {best_dir}")
    if volume is not None:
        _commit()


# ── Phase 5: Bulletproof Inference ────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    input_text: str,
    task: str = "denoise",
    device: str = "cpu",
) -> str:
    """Generate a response for any task.

    Decoding strategy is task-conditional:
      * pos / lemma        → GREEDY (their outputs legitimately repeat tokens; penalties would
                             corrupt the tag/lemma sequence).
      * denoise / synoptic → Contrastive Search (SOTA anti-degeneration for free-form Greek).

    Tasks:
      - denoise:           prefix `denoise: <text with masks>`
      - pos:               prefix `pos: <greek text>`
      - lemma:             prefix `lemma: <greek text>`
      - synoptic_mk_to_mt: prefix `synoptic mark_to_matt: <mark text>`
      - synoptic_mk_to_lk: prefix `synoptic mark_to_luke: <mark text>`
    """
    import torch

    task_prefixes = {
        "denoise":           "denoise: ",
        "pos":               "pos: ",
        "lemma":             "lemma: ",
        "synoptic_mk_to_mt": "synoptic mark_to_matt: ",
        "synoptic_mk_to_lk": "synoptic mark_to_luke: ",
    }

    prefix = task_prefixes.get(task, "")
    full_input = prefix + input_text if not input_text.startswith(prefix) else input_text

    inputs = tokenizer(full_input, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)

    if task in ("pos", "lemma"):
        gen_kwargs = dict(max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=1, do_sample=False)
    else:
        gen_kwargs = dict(
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            penalty_alpha=GEN_PENALTY_ALPHA,
            top_k=GEN_TOP_K,
            repetition_penalty=GEN_REP_PENALTY,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
            encoder_no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
            early_stopping=True,
        )

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Fallback: if model returned empty, return the input as-is
    if not decoded.strip():
        decoded = input_text

    return decoded


# ── Modal entrypoints ──────────────────────────────────────────────────────────

@app.local_entrypoint()  # type: ignore[misc]
def upload_corpus() -> None:
    """Upload the local processed corpus to the Modal data volume (run once).

    Reuses the existing 'synoptiq-data' volume which already contains the
    processed SynoptiQ corpus (tokens.parquet + pericopes.parquet).
    If the files are already there, this is a no-op — skip it.

    Requires: data/processed/tokens.parquet and data/processed/pericopes.parquet
    """
    import subprocess

    processed = Path("data/processed")
    if not processed.exists():
        print("ERROR: data/processed/ not found. Run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    print(f"Uploading {processed} → Modal volume '{DATA_VOLUME}:/processed' (--force) ...")
    result = subprocess.run(
        # --force overwrites existing files; safe to re-run
        ["modal", "volume", "put", "--force", DATA_VOLUME, str(processed), "/processed"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Upload failed:\n{result.stderr}")
        sys.exit(1)
    print(f"Corpus uploaded to Modal volume '{DATA_VOLUME}'.")
    print("Contents: tokens.parquet, pericopes.parquet (49,061 tokens, 170 pericopes, Mt/Mk/Lk)")


@app.local_entrypoint()  # type: ignore[misc]
def upload_proiel() -> None:
    """(Optional) pre-stage the PROIEL treebank onto the Modal data volume.

    Downloads the three CoNLL-U splits locally (into data/raw/proiel) and uploads them to
    '{DATA_VOLUME}:/proiel'. Not required — `train` auto-downloads PROIEL at runtime
    — but useful for fully offline/reproducible runs. Safe to re-run (skips present files).
    """
    import subprocess

    local = Path(PROIEL_LOCAL_DIR)
    if not _proiel_files_present(PROIEL_LOCAL_DIR):
        print(f"Downloading PROIEL → {local} ...")
        if not _download_proiel(PROIEL_LOCAL_DIR):
            print("ERROR: could not download PROIEL (offline?). Place the three "
                  "grc_proiel-ud-*.conllu files under data/raw/proiel/ manually.")
            sys.exit(1)

    print(f"Uploading {local} → Modal volume '{DATA_VOLUME}:/proiel' (--force) ...")
    result = subprocess.run(
        ["modal", "volume", "put", "--force", DATA_VOLUME, str(local), "/proiel"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Upload failed:\n{result.stderr}")
        sys.exit(1)
    print(f"PROIEL uploaded to '{DATA_VOLUME}:/proiel' "
          "(New Testament Koine + Herodotus Classical, ~214K tokens).")


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes=_VOLUMES,
    timeout=TIMEOUT,
) if modal is not None else None
def train() -> None:
    """Full Koine-T5 training run on Modal A10G GPU.

    Trains on four balanced task pools (denoise / pos / lemma / synoptic), fed by the Gospel
    corpus + the UD_Ancient_Greek-PROIEL treebank, with POS-EM validation and best-checkpoint
    selection. Outputs LoRA adapters to /outputs/koine_t5/ (step-N + best/ + final/).
    """
    device     = "cuda"
    output_dir = Path("/outputs/koine_t5")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path    = "/data/processed/tokens.parquet"
    pericopes_path = "/data/processed/pericopes.parquet"

    # Phase 1: tokenizer (sentinel hack)
    print("Phase 1: Building Koine-T5 tokenizer with sentinel tokens...")
    tokenizer = build_tokenizer()
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    print(f"  Sentinel <extra_id_0> -> id {tokenizer.convert_tokens_to_ids('<extra_id_0>')}")

    # Phase 2: model (bfloat16 + maximalist LoRA)
    print("\nPhase 2: Loading Koine-T5 (bfloat16, LoRA r=32, encoder+decoder)...")
    model = load_model_with_lora(tokenizer, device=device)

    # Phase 3: resolve PROIEL (graceful fallback if unavailable)
    print("\nPhase 3: Resolving PROIEL treebank...")
    proiel_dir = resolve_proiel_dir(allow_download=True)
    if proiel_dir:
        print(f"  PROIEL available at: {proiel_dir}")
    else:
        print("  [warn] PROIEL NOT available — falling back to Synoptic-only training "
              "(no POS-EM eval, no best-checkpoint selection).")

    # Phase 4: build the four task pools + POS eval set
    print("\nPhase 4: Building task pools...")
    pools = build_task_pools(tokens_path, pericopes_path, proiel_dir)
    eval_records = build_proiel_eval(proiel_dir) if proiel_dir else []
    print(f"  Eval records (PROIEL dev): {len(eval_records)}")

    # Phase 5: train
    print("\nStarting Koine-T5 training loop...")
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME) if modal is not None else None
    _training_loop(model, tokenizer, pools, eval_records, output_dir, device, volume=output_vol)

    print("\nDone! Download adapters:")
    print(f"  modal volume get {OUTPUT_VOLUME} koine_t5/best  models/koine_t5/best")
    print(f"  modal volume get {OUTPUT_VOLUME} koine_t5/final models/koine_t5/final")
    print("\nLocal demo:")
    print("  python modal/app_koine_t5.py demo models/koine_t5/best")


# ── Local Demo / Inference ─────────────────────────────────────────────────────

def _run_demo(adapter_path: str | None = None) -> None:
    """Run a local inference demo comparing GreTa base vs Koine-T5."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Koine-T5 demo on {device}\n")

    tokenizer = build_tokenizer()

    print("Loading GreTa Base (pre-Koine-T5)...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    if hasattr(base_model.config, "tie_word_embeddings"):
        base_model.config.tie_word_embeddings = False
    base_model.config.vocab_size = len(tokenizer)
    base_model.eval()

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading Koine-T5 adapters from {adapter_path}...")
        ult_base = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID, local_files_only=True, dtype=torch.bfloat16
        ).to(device)
        if hasattr(ult_base.config, "tie_word_embeddings"):
            ult_base.config.tie_word_embeddings = False
        ult_base.config.vocab_size = len(tokenizer)
        ult_model = PeftModel.from_pretrained(
            ult_base, adapter_path, local_files_only=True
        ).to(device)
    else:
        print("No adapter found — showing base model only")
        ult_model = None

    test_cases = [
        {
            "task": "denoise",
            "name": "Mark 1:1 — Span Completion",
            "text": "Ἀρχὴ τοῦ <extra_id_0> Ἰησοῦ Χριστοῦ <extra_id_1>.",
            "note": "Target: εὐαγγελίου ... υἱοῦ θεοῦ",
        },
        {
            "task": "pos",
            "name": "Luke 1:46 — POS Tagging",
            "text": "μεγαλύνει ἡ ψυχή μου τὸν κύριον",
            "note": "Expected: V- RA N- RP RA N-",
        },
        {
            "task": "lemma",
            "name": "Matthew 6:9 — Lemmatization",
            "text": "Πάτερ ἡμῶν ὁ ἐν τοῖς οὐρανοῖς",
            "note": "Expected: πατήρ ἐγώ ὁ ἐν ὁ οὐρανός",
        },
        {
            "task": "synoptic_mk_to_mt",
            "name": "Mark→Matthew Style Transfer",
            "text": "καὶ εὐθὺς ἀνέβη ἐκ τοῦ ὕδατος",
            "note": "Expected Matthew paraphrase of Mark 1:10",
        },
    ]

    print("\n" + "=" * 80)
    print("GRETA BASE  vs  KOINE-T5")
    print("=" * 80)

    for case in test_cases:
        print(f"\n  {case['name']}")
        print(f"  Input:    {case['text']}")
        print(f"  Expected: {case['note']}")

        base_out = generate(base_model, tokenizer, case["text"],
                                     task=case["task"], device=device)
        print(f"  GreTa Base:           {base_out}")

        if ult_model is not None:
            ult_out = generate(ult_model, tokenizer, case["text"],
                                        task=case["task"], device=device)
            print(f"  Koine-T5: {ult_out}")

        print("-" * 80)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        adapter = sys.argv[2] if len(sys.argv) > 2 else "models/koine_t5/best"
        _run_demo(adapter_path=adapter)
    else:
        print(__doc__)
        print("\nTo run the demo locally:  python modal/app_koine_t5.py demo [adapter_path]")
        print("To train on Modal:        modal run modal/app_koine_t5.py::train")
        print("To upload corpus first:   modal run modal/app_koine_t5.py::upload_corpus")
        print("To pre-stage PROIEL:      modal run modal/app_koine_t5.py::upload_proiel")
