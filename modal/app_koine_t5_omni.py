"""Koine-T5 Omni: a general-purpose multitask Ancient Greek seq2seq model.

Standalone Modal training script — no imports from the synoptiq package. Trains a
single GreTa+LoRA model on SEVEN balanced task pools simultaneously:

  1. denoise    — T5 span corruption (online, on raw Greek prose)
  2. pos        — part-of-speech tagging  (MorphGNT tagset)
  3. lemma      — lemmatization
  4. morphology — full parse code per token, compact encoding (MorphGNT col 2+3)
  5. normalize  — crasis/itacism resolution → standard polytonic form
  6. restore    — uncial/scriptio-continua → polytonic with diacritics (synthetic)
  7. synoptic   — Mark→Matthew and Mark→Luke style transfer (small curated pool)

Data sources:
  * PROIEL CoNLL-U (NT Koine + Herodotus, ~214K tokens) → pos/lemma/denoise
  * MorphGNT all 27 NT books (on disk: data/raw/morphgnt/) → morphology/pos/lemma/denoise
  * Real crasis expansions + lexicon-gated itacism on raw texts → normalize
  * Synthetic diacritic stripping on existing raw texts → restore
  * Synoptic Gospel corpus (existing processed parquets) → synoptic/pos/lemma/denoise

All new pools are graceful-fallback: if any download fails, the pool is skipped
and training continues on the remaining pools.

TWO English-output pools were tried and removed. `translate` (Greek→English verse) never ran: no
English pretraining plus ~6K verse pairs is the shelved Koine-T5-Hexapla bet again. `gloss`
(word-level English) DID run for 54,000 steps and reached 0.10 token accuracy — see TASK_WEIGHTS
for the measured post-mortem. This backbone does not cross into English at this data scale, by
either route.

Validation runs every task with greedy decoding. `pos` is scored on PROIEL **dev**, split into
Koine-NT and Classical subsets; the remaining tasks are scored on held-out slices carved out of
their own pools. Checkpoint selection is GATED (see `evaluate_all`): a checkpoint must hold POS-NT
and lemma at threshold before its secondary-task score can win, so `best/` is the best *omni*
model rather than the best tagger. Headline numbers come from PROIEL **test** via `run_test`.

PROIEL background (verified against the real CoNLL-U, 20180408 release):
  * Composition: ~52–59% Koine NT (Matthew/Mark/Luke/John/Acts/Revelation/Romans…),
    ~41–48% Classical (Herodotus *Histories*). This is the right treebank for a Koine model —
    UD-Perseus has NO New Testament and is Classical poetry only.
  * License CC BY-NC-SA 3.0 (NonCommercial; fine for research training).
  * Point of speech (POS) mapping uses the **XPOS** column, NOT UPOS/FEATS. The article ὁ/ἡ/τό is UPOS=DET with
    FEATS `PronType=Dem` (which would mislabel it as a demonstrative); its XPOS is `S-`,
    which maps cleanly to MorphGNT `RA`. See PROIEL_XPOS_TO_MORPHGNT below.

Usage:
    # Upload the processed Gospel corpus to the Modal volume (once):
    modal run modal/app_koine_t5_omni.py::upload_corpus

    # Upload MorphGNT to the Modal volume:
    modal run modal/app_koine_t5_omni.py::upload_morphgnt

    # (Optional) pre-stage PROIEL onto the Modal data volume for offline runs:
    modal run modal/app_koine_t5_omni.py::upload_proiel

    # Start training (A10G GPU, auto-downloads PROIEL if absent, survives laptop sleep):
    modal run modal/app_koine_t5_omni.py::train

    # Monitor live logs:
    modal app logs koine-t5-omni

    # Download the best adapter (model-selected on POS EM) and the final adapter:
    modal volume get koine-t5-omni-outputs koine_t5_omni/best  models/koine_t5_omni/best
    modal volume get koine-t5-omni-outputs koine_t5_omni/final models/koine_t5_omni/final

    # Run local demo / inference:
    python modal/app_koine_t5_omni.py demo
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
OUTPUT_VOLUME = "koine-t5-omni-outputs"    # dedicated output volume for Koine-T5 adapters (best/ + final/)
GPU_TYPE      = "A10G"
TIMEOUT       = 86_400                   # 24 hours

BASE_MODEL_ID = "bowphs/GreTa"           # T5-base fine-tuned on Ancient Greek

# LoRA / model hyper-parameters
LORA_R          = 64    # LoRA rank for the shared 4-task adapter
LORA_ALPHA      = 128   # kept at 2×r (standard LoRA scaling)
LORA_DROPOUT    = 0.05
MAX_SEQ_LEN     = 256   # 512 OOM'd on A10G; 256 fits Koine verse length
BATCH_SIZE      = 4     # micro-batch slots; filled ∝ TASK_WEIGHTS (see sample_balanced_batch)
GRAD_ACCUM      = 8     # effective batch = 32
LR              = 1e-4

# ── Step budget & task balance ──────────────────────────────────────────────────
# READ THIS BEFORE CHANGING TASK_WEIGHTS OR ADDING A TASK.
#
# Sampling is WEIGHTED, not one-slot-per-task: each of the BATCH_SIZE slots independently picks a
# task ∝ TASK_WEIGHTS. That makes every task's effective epoch count depend on the SUM of all
# weights — so adding tasks silently starves the existing ones unless MAX_STEPS rises to match.
#
# This is exactly how the 9-task revision regressed. It inherited v1's MAX_STEPS=30_000 while the
# weight total went 8 → 17 and the pos pool grew 15,041 → 22,968, which cut POS from
#     v1:   120,000 × 3/8  ÷ 15,041 = 2.99 epochs   → 0.966 NT
#   to  omni: 120,000 × 3/17 ÷ 22,968 = 0.92 epochs   → 0.678 NT
# and the best checkpoint landed on the FINAL step, i.e. POS never plateaued. Never edit this
# block without recomputing the table below.
#
# Current budget: 75,000 micro-steps × BATCH_SIZE 4 = 300,000 draws, weights summing to 17.5:
#     pos        4.5  →  77,143 draws / 22,856  =  3.38 epochs   (the reported/validated task)
#     lemma      3.5  →  60,000 draws / 22,711  =  2.64 epochs
#     denoise    3.0  →  51,429 draws / 23,011  =  2.23 epochs   (the general-LM backbone)
#     morphology 2.5  →  42,857 draws /  7,727  =  5.55 epochs   (hardest: 10 joint axes)
#     restore    2.0  →  34,286 draws /  9,114  =  3.76 epochs
#     normalize  1.5  →  25,714 draws / ~7,900  =  3.25 epochs
#     synoptic   0.5  →   8,571 draws /     98  = 87×           (tiny pool, upsampled)
#
# 3 epochs is the proven POS budget (0.8-0.9 lands below a majority-class baseline of 0.215).
# _training_loop prints this table live at startup from the ACTUAL pool sizes — read it and confirm
# it matches before letting a run proceed. best/ is selected on the eval curve, so over-shooting
# late steps is harmless. All counts are MICRO-steps; lr_lambda converts to optimizer steps.
MAX_STEPS           = 75_000
WARMUP_STEPS        = 3_750   # micro-steps; ~5% of MAX_STEPS (converted to opt-steps in lr_lambda)
SAVE_STEPS          = 1_000   # checkpoint + resumable training-state cadence
# Eval cadence is a WALL-CLOCK decision, not just a resolution one. Each eval generates
# EVAL_MAX_PER_SUBSET*2 POS sentences plus HOLDOUT_PER_TASK for every secondary task; at
# EVAL_STEPS=1000 with 5 secondary tasks the rev-6 run spent several of its ~8 hours generating
# rather than training (~3,500 generate() calls vs v1's ~480). 2,500 keeps 30 evals across the run
# — ample for picking a checkpoint — at ~40% of the eval cost.
EVAL_STEPS          = 2_500   # multi-task validation + best-checkpoint selection cadence
LOG_STEPS           = 100
CKPT_KEEP           = 2       # retain only the N newest step-N checkpoints (older ones pruned)

# POS/eval example shaping
MAX_POS_WORDS       = 60      # cap PROIEL sentence length so pos/lemma input↔target stay aligned
EVAL_MAX_PER_SUBSET = 250     # dev sentences per subset (NT / Classical) per eval — 500 total
HOLDOUT_PER_TASK    = 150     # held-out examples per secondary task (was 200; see EVAL_STEPS)
EVAL_BATCH_SIZE     = 32      # sentences per generate() call; ~15x faster than batch=1 on A10G

# Span-corruption hyper-parameters (T5 paper §3.1)
NOISE_DENSITY         = 0.15
MEAN_NOISE_SPAN_LEN   = 3.0

# Tasks decoded greedily: every one emits exactly one output unit per input word, so beam search
# buys nothing and any repetition penalty is actively harmful (tags, lemmas and near-copy Greek all
# repeat by design). See generate() for the full rationale.
_GREEDY_TASKS = frozenset({"pos", "lemma", "morphology", "normalize", "restore"})

# Contrastive-search generation defaults (denoise inference only)
GEN_PENALTY_ALPHA  = 0.6
GEN_TOP_K          = 4
GEN_MAX_NEW_TOKENS = 256
GEN_REP_PENALTY    = 1.25
GEN_NO_REPEAT_NGRAM = 3

# ── PROIEL treebank configuration ────────────────────────────────────────────────
# Paths are configurable for both the remote Modal volume and a local checkout, with a
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

MORPHGNT_LOCAL_DIR  = "data/raw/morphgnt"   # all 27 NT books, already on disk
MORPHGNT_REMOTE_DIR = "/data/morphgnt"      # remote volume path


def resolve_morphgnt_dir() -> str | None:
    """Locate MorphGNT directory on remote volume or local disk."""
    for candidate in (MORPHGNT_REMOTE_DIR, MORPHGNT_LOCAL_DIR):
        p = Path(candidate)
        if p.exists() and list(p.glob("*-morphgnt.txt")):
            return candidate
    return None

# Synthetic task hyper-parameters
NORMALIZE_PROB  = 0.4    # fraction of raw texts to use for normalize examples
RESTORE_PROB    = 0.4    # fraction of raw texts to use for restore examples


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
# 1:1 — unlike UPOS (which collapses RA/RD/RI/RP/RR into DET/PRON) or FEATS
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

# ── PROIEL→MorphGNT lemma overrides (tagset-convention reconciliation) ───────────
# The XPOS table above is right about *classes* but the two treebanks disagree about a handful of
# very high-frequency lemmas. MorphGNT calls the postpositive particles conjunctions (δέ/γάρ/οὖν
# → C-) and buckets ἄν/δή/ἰδού into X-; PROIEL tags all of them XPOS=Df, which the class table maps
# to D-. Since the `pos` pool mixes PROIEL, MorphGNT, and the Gospel corpus, that disagreement is
# not a nuance — it is *contradictory supervision on the most frequent words in the corpus*, and it
# lands on the eval set too. Measured on PROIEL dev before these overrides: the MorphGNT-majority
# tag disagreed with PROIEL gold on 1,133/13,652 tokens = 8.30% (δέ alone 443).
#
# The project's canonical tagset is MorphGNT (`synoptiq/utils/constants.py`, the Gospel corpus `pos`
# column, the Koine Reader), so PROIEL is conformed to MorphGNT — not the reverse.
#
# Derivation (not hand-picked): every lemma where MorphGNT is ≥90% pure on one tag, the XPOS class
# mapping is ≥90% pure on a *different* tag, and PROIEL-train has ≥20 tokens. μέν is included at
# 87.6% MorphGNT purity — PROIEL is 1,154/1,154 Df, an unambiguous convention difference.
# Effect on PROIEL dev: NT conflict 5.06% → 1.58%, Classical 12.75% → 4.54%, pooled 8.30% → ~2.8%.
# The residual (ὁ as relative/personal pronoun in Herodotus, adverbial καί, αὐτός as demonstrative)
# is genuinely context-dependent and cannot be fixed at the lemma level.
PROIEL_LEMMA_OVERRIDES: dict[str, str] = {
    # postpositive particles: MorphGNT = conjunction, PROIEL XPOS = Df (adverb)
    "δε": "C-", "γαρ": "C-", "ουν": "C-", "μεν": "C-",
    "αρα": "C-", "μεντοι": "C-", "διο": "C-",
    # particles MorphGNT buckets as X- (PROIEL: Df adverb, or I- interjection)
    "δη": "X-", "αν": "X-", "γε": "X-",
    "ιδου": "X-", "αμην": "X-", "ναι": "X-",
    # gentilic nouns: MorphGNT treats as proper nouns, PROIEL as adjectives
    "κυρηναιος": "N-", "κορινθιος": "N-",
    # χήρα "widow": MorphGNT substantival adjective, PROIEL common noun
    "χηρα": "A-",
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    ).lower()


def map_xpos_to_morphgnt(xpos: str, lemma: str) -> str:
    """Map a PROIEL XPOS tag (+ lemma) to the MorphGNT POS code the pipeline expects.

    Two lemma-keyed corrections sit on top of the XPOS class table:

      * `Px` is the one genuinely split class: PROIEL lumps the indefinite pronoun τις together
        with quantifiers (πᾶς, οὐδείς, ἄλλος, ἕκαστος). The Gospel corpus tags τις as `RI` but
        the quantifiers as `A-`, so we key on the lemma to reproduce that distinction.
      * `PROIEL_LEMMA_OVERRIDES` reconciles the two treebanks' tagset *conventions* on the
        high-frequency particles (δέ/γάρ/οὖν/μέν → C-, ἄν/δή/ἰδού → X-). Without it, ~8.3% of
        PROIEL-dev tokens carry a label that contradicts the MorphGNT half of the `pos` pool.

    The override is applied first: it is a statement about the lemma, not about the XPOS class.
    """
    lem = _strip_accents(lemma)
    if lem in PROIEL_LEMMA_OVERRIDES:
        return PROIEL_LEMMA_OVERRIDES[lem]
    if xpos == "Px":
        return "RI" if lem in INDEF_PRONOUN_LEMMAS else "A-"
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

app = modal.App("koine-t5-omni") if modal is not None else None

_VOLUMES = {
    "/data":    modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
} if modal is not None else {}


def _commit() -> None:
    if modal is not None:
        modal.Volume.from_name(OUTPUT_VOLUME).commit()


# ── Tokenizer setup ──────────────────────────────────────

def build_tokenizer(local_files_only: bool = False):
    """Load the Koine-T5 tokenizer (bowphs/GreTa) and register the 100 T5 sentinel tokens.

    bowphs/GreTa already ships with 32103 embedding slots in the model weights
    but only 32003 tokens in the tokenizer vocabulary. By adding exactly 100
    sentinel tokens (<extra_id_0> … <extra_id_99>), we map them to the existing
    pre-trained ghost slots (indices 32003–32102) without resizing the model
    embeddings and without introducing any random-weight noise.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=local_files_only)

    # Bind existing pad/eos — do not add a new [PAD] token (breaks decoder start id)
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
    """Load the Koine-T5 base (bowphs/GreTa) in bfloat16 with LoRA on all attention + FFN projections.

    bfloat16 halves VRAM vs float32 (same exponent range so no overflow risk),
    which avoids A10G OOM at batch=4, seq_len=256.
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

    # LoRA on encoder + decoder (attention + FFN).
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


# ── T5 span corruption ──────────────────────────────────────────

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


# ── PROIEL CoNLL-U loading ──────────────────────────────────────────

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


def _xpos_to_morphgnt_no_overrides(xpos: str, lemma: str) -> str:
    """The XPOS class mapping WITHOUT PROIEL_LEMMA_OVERRIDES — i.e. the convention Koine-T5 v1 learned."""
    if xpos == "Px":
        return "RI" if _strip_accents(lemma) in INDEF_PRONOUN_LEMMAS else "A-"
    return PROIEL_XPOS_TO_MORPHGNT.get(xpos, FALLBACK_MORPHGNT)


def build_proiel_eval(proiel_dir: str, split: str = "dev") -> list[dict]:
    """Build POS eval records from a PROIEL split: [{text, gold(list[str]), subset}].

    `dev` drives checkpoint selection during training; `test` is scored ONCE on the final best/
    adapter for the headline numbers, so the reported figure is not selection-contaminated.

    Each record also carries `neutral`: a per-token mask that is True where the gold label is the
    SAME with and without PROIEL_LEMMA_OVERRIDES. This exists to make cross-model comparison
    honest. The overrides encode the MorphGNT tagset convention, which omni was trained on and
    Koine-T5 v1 was not, so scoring v1 on the overridden tokens marks it wrong for answering in the
    convention it was actually taught. On the test split that touches 5.6% of tokens but **39.3% of
    NT sentences and 85.5% of Classical sentences** — enough to make a full-sentence exact-match
    comparison between the two models meaningless. Score with `neutral_only=True` to compare.
    """
    records: list[dict] = []
    path = _proiel_file(proiel_dir, split)
    if not path.exists():
        return records
    for source, toks in iter_conllu_sentences(path):
        if not (2 <= len(toks) <= MAX_POS_WORDS):
            continue
        gold = [map_xpos_to_morphgnt(t[2], t[1]) for t in toks]
        old  = [_xpos_to_morphgnt_no_overrides(t[2], t[1]) for t in toks]
        records.append({
            "text": " ".join(t[0] for t in toks),
            "gold": gold,
            "neutral": [g == o for g, o in zip(gold, old)],
            "subset": "nt" if _is_nt_source(source) else "classical",
        })
    return records


# ── Gospel corpus loading ───────────────────────────────────────────

def load_synoptic_pairs(tokens_path: str, pericopes_path: str) -> tuple[list[dict], list[str]]:
    """Load parallel Synoptic pericopes from the processed corpus parquets.

    Returns (examples, raw_texts):
      * examples  — instruction dicts (task, input_text, target_text) covering
                    pos / lemma / synoptic_mk_to_mt / synoptic_mk_to_lk
      * raw_texts — the raw Greek surface text of every book's pericope, for the
                    denoise pool (denoise trains on raw Greek prose, never POS/lemma target strings)
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


# ── MorphGNT (all 27 NT books) ────────────────────────────────────────────────

# ── Compact morphology tag encoding ────────────────────────────────────────────
# The native MorphGNT tag is POS(2) + PARSE(8) = 10 chars, e.g. "N----NSF-" / "V-3PAI-S--", where
# the hyphens are positional padding for unfilled axes. GreTa's SentencePiece has no notion of that
# structure and shreds each run of hyphens into single characters:
#     "N----NSF-" → ['▁','n','-','-','-','-','ns','f','-']      (9 tokens for one tag)
# At 9.2 tokens/word a 60-word verse needs ~550 target tokens — well past MAX_SEQ_LEN=256 — so
# 10.9% of morphology targets were being silently truncated mid-sequence by _collate_batch, which
# is precisely what destroys tag↔word alignment.
#
# Dropping the padding hyphens ("N----NSF-" → "N-NSF", "V-3PAI-S--" → "V-3PAIS") costs 2.26× fewer
# tokens (p50 163 → 72, p95 292 → 132, max 575 → 242) and brings truncation to 0%.
#
# This is lossless: across all 602 distinct tags attested in MorphGNT the compact form produces
# ZERO collisions, so `build_morph_decode_table()` inverts it exactly. The table is written next to
# the adapter as `morph_tags.json` so any consumer can decode back to the canonical 10-char code.

def compact_morph_tag(tag: str) -> str:
    """Compress a 10-char MorphGNT tag to its non-padding characters ("N----NSF-" → "N-NSF")."""
    import re
    return re.sub(r"-+$", "", tag[:2].rstrip("-") + "-" + tag[2:].replace("-", ""))


def build_morph_decode_table(morphgnt_dir: str) -> dict[str, str]:
    """Map every attested compact tag back to its canonical 10-char MorphGNT form.

    Raises if the compaction is not injective over the observed tagset — that invariant is what
    makes the compact target lossless, so it is checked rather than assumed.
    """
    full: set[str] = set()
    for txt_file in sorted(Path(morphgnt_dir).glob("*-morphgnt.txt")):
        with open(txt_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 7:
                    full.add(parts[1] + parts[2])

    table: dict[str, str] = {}
    for tag in sorted(full):
        key = compact_morph_tag(tag)
        if key in table and table[key] != tag:
            raise ValueError(
                f"compact morphology encoding is not injective: {key!r} ← "
                f"{table[key]!r} and {tag!r}"
            )
        table[key] = tag
    return table


def decode_morph_tag(compact: str, table: dict[str, str]) -> str:
    """Invert `compact_morph_tag` via the table; unknown forms pass through unchanged."""
    return table.get(compact.upper(), compact.upper())

# Internal 2-digit book code → OSIS book abbreviation (for verse-key alignment)
_MORPHGNT_BOOK_TO_OSIS: dict[str, str] = {
    "01": "Matt", "02": "Mark", "03": "Luke", "04": "John", "05": "Acts",
    "06": "Rom",  "07": "1Cor", "08": "2Cor", "09": "Gal",  "10": "Eph",
    "11": "Phil", "12": "Col",  "13": "1Thess","14": "2Thess","15": "1Tim",
    "16": "2Tim", "17": "Titus","18": "Phlm", "19": "Heb",  "20": "Jas",
    "21": "1Pet", "22": "2Pet", "23": "1John","24": "2John","25": "3John",
    "26": "Jude", "27": "Rev",
}


def build_morphgnt_training(
    morphgnt_dir: str,
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Build morphology/pos/lemma/denoise examples from all 27 MorphGNT books.

    MorphGNT line format (7 space-separated fields):
      BCVW   POS  PARSE     WORD        NORMALIZED  DICTIONARY  LEMMA
      010101 N-   ----NSF-  Βίβλος      Βίβλος      βίβλος      βίβλος

    BCVW is a 6-digit string: BB=book (01-27), CC=chapter, VV=verse.
    The full morphological tag per word = POS (2 chars) + PARSE (8 chars) = 10 chars.

    Returns: (morphology_examples, pos_examples, lemma_examples, raw_texts)
    """
    from collections import defaultdict

    morphology_examples: list[dict] = []
    pos_examples: list[dict] = []
    lemma_examples: list[dict] = []
    raw_texts: list[str] = []

    morphgnt_path = Path(morphgnt_dir)
    if not morphgnt_path.exists():
        print(f"  [morphgnt] directory not found: {morphgnt_dir} — skipping")
        return morphology_examples, pos_examples, lemma_examples, raw_texts

    txt_files = sorted(morphgnt_path.glob("*-morphgnt.txt"))
    if not txt_files:
        print(f"  [morphgnt] no *-morphgnt.txt files found in {morphgnt_dir} — skipping")
        return morphology_examples, pos_examples, lemma_examples, raw_texts

    # Group tokens by verse key (BCVW[:6])
    verse_tokens: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for txt_file in txt_files:
        with open(txt_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                # Use the NORMALIZED column (5th), not the raw WORD column (4th). The raw column
                # carries SBLGNT apparatus markers — ⸀ (5,114 tokens), ⸂/⸃ (1,764/1,765), ⸁ (46) —
                # which are textual-variant sigla, not Greek. Feeding them in puts ~8,700 junk
                # characters into the inputs of every MorphGNT-derived task. `normalized` is the
                # same token with the sigla and trailing punctuation already removed.
                bcvw, pos, parse, _word, normalized, _dict, lemma = parts[:7]
                verse_key = bcvw[:6]   # BB CC VV (ignore word index)
                surface = normalized.rstrip(",.:;·!?()")
                if not surface:
                    continue
                verse_tokens[verse_key].append((surface, pos, parse, lemma))

    n_morphology = n_pos = n_lemma = 0
    for verse_key, tokens in sorted(verse_tokens.items()):
        if len(tokens) < 2:
            continue
        # Cap at MAX_POS_WORDS to keep sequences inside MAX_SEQ_LEN
        if len(tokens) > MAX_POS_WORDS:
            raw_texts.append(" ".join(t[0] for t in tokens))
            continue

        surface_seq = " ".join(t[0] for t in tokens)
        raw_texts.append(surface_seq)

        # morphology: full POS+PARSE tag per word, emitted in the compact (padding-free) encoding
        # so the target stays inside MAX_SEQ_LEN — see compact_morph_tag() for why.
        morph_tags = [compact_morph_tag(t[1] + t[2]) for t in tokens]   # "N-NSF", "V-3AAIS"
        morphology_examples.append({
            "task": "morphology",
            "input_text": f"morphology: {surface_seq}",
            "target_text": " ".join(morph_tags),
        })
        n_morphology += 1

        # pos: 2-char POS tag per word (compatible with PROIEL pos format)
        pos_examples.append({
            "task": "pos",
            "input_text": f"pos: {surface_seq}",
            "target_text": " ".join(t[1] for t in tokens),
        })
        n_pos += 1

        # lemma: lemma sequence
        lemma_examples.append({
            "task": "lemma",
            "input_text": f"lemma: {surface_seq}",
            "target_text": " ".join(t[3] for t in tokens),
        })
        n_lemma += 1

    print(f"  MorphGNT: {n_morphology:,} morphology | {n_pos:,} pos | {n_lemma:,} lemma "
          f"| {len(raw_texts):,} denoise texts  (from {len(txt_files)} books)")
    return morphology_examples, pos_examples, lemma_examples, raw_texts


# ── Synthetic normalize pool (crasis/itacism) ──────────────────────────────────

# Koine crasis forms, keyed by ACCENT-STRIPPED crasis form → expansion (also accent-stripped).
# Matching is accent-insensitive because Greek accents shift with context: κἀγώ appears as κἀγὼ
# (grave) before an enclitic-less following word, and an exact-string rule silently misses those.
#
# The counts below are real occurrences in the MorphGNT NT surface text. They matter: the previous
# implementation searched for the EXPANSION and rewrote it into crasis, but "καὶ ἐγώ" occurs
# exactly ONCE in the whole NT, and κοὐ/κοὐκ (which were in this table) occur ZERO times — they are
# Attic, not Koine. So the model essentially never saw the κἀγώ ↔ καὶ ἐγώ mapping it is asked for.
# Building in the natural direction (find a crasis form that is actually there, expand it) yields
# 130 genuine anchors instead of ~1.
# Keys are accent-stripped (matching is accent-insensitive); VALUES CARRY THEIR REAL ACCENTS.
# The values must be properly accented Greek: an earlier revision stored them accent-stripped, which
# made every crasis target read like "και εγω δέ σοι λέγω" — unaccented words spliced into accented
# text. The model was being asked to produce something incoherent, and unsurprisingly never did.
_CRASIS_EXPANSIONS: dict[str, str] = {
    "καγω":       "καὶ ἐγὼ",        # 71 occurrences
    "καν":        "καὶ ἐάν",        # 14
    "κακει":      "καὶ ἐκεῖ",       # 10
    "κακειθεν":   "καὶ ἐκεῖθεν",    #  8
    "κακεινος":   "καὶ ἐκεῖνος",    #  7
    "κακεινοι":   "καὶ ἐκεῖνοι",    #  7
    "καμοι":      "καὶ ἐμοὶ",       #  5
    "καμε":       "καὶ ἐμὲ",        #  3
    "τουναντιον": "τὸ ἐναντίον",    #  3
    "κακεινους":  "καὶ ἐκείνους",   #  1
    "τουνομα":    "τὸ ὄνομα",       #  1
    # deliberately absent: κοὐ / κοὐκ (0 occurrences in the NT — Attic, not Koine)
}

# Itacism confusions as (correct_spelling, scribal_error). Applied correct → error to build the
# corrupted input; the model learns the inverse.
_ITACISMS: list[tuple[str, str]] = [
    ("ει", "ι"),   # ει written ι (the most common NT itacism)
    ("η",  "ι"),   # η  written ι
    ("οι", "υ"),   # οι written υ
    ("αι", "ε"),   # αι written ε
    ("ω",  "ο"),   # ω  written ο
]

# Fraction of eligible words to corrupt per example. This is the single most important number in
# the normalize task, and the first revision got it wrong: 1-3 edits per sentence left ~90% of
# tokens already correct, so "echo the input" was a near-optimal policy and the model learned
# exactly that (measured lift over copying: −0.011, i.e. worse than a no-op).
#
# `restore` is the control that shows why: it corrupts EVERY token (strips all diacritics), copying
# earns 0.052, and the model genuinely learned the task (+0.665 lift). Dense corruption is what
# forces learning. 0.40 puts normalize in the same regime while keeping sentences readable.
ITACISM_EDIT_RATE = 0.40
MIN_ITACISM_EDITS = 2


def build_normalize_lexicon(morphgnt_dir: str | None) -> set[str]:
    """Accent-stripped set of every surface form attested in MorphGNT (~20,487 forms).

    Used to keep the normalize task WELL-POSED. A corruption is only accepted if the result is not
    itself a real word: replacing αι→ε inside a form that happens to yield another attested word
    leaves the original unrecoverable, so the model would be trained on an input with two equally
    valid targets. Measured over all candidate edits, 16,980/17,440 (97.4%) pass this gate — the
    cost of the check is small and it removes the ill-posed remainder.
    """
    lexicon: set[str] = set()
    if not morphgnt_dir:
        return lexicon
    for txt_file in sorted(Path(morphgnt_dir).glob("*-morphgnt.txt")):
        with open(txt_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 7:
                    lexicon.add(_strip_accents(parts[3].rstrip(",.:;·!?()")))
    return lexicon


def build_normalize_examples(raw_texts: list[str], rng_py,
                             lexicon: set[str] | None = None) -> list[dict]:
    """Build normalize examples: crasis/itacism-corrupted Greek → standard polytonic form.

      Input:  "normalize: κἀγὼ εἶπον αὐτῷ"
      Output: "καὶ ἐγὼ εἶπον αὐτῷ"

    Two strategies, applied per word rather than per raw substring so edits land at varied
    positions instead of always at the start of the sentence:

      1. Crasis expansion — a word whose accent-stripped form is a known crasis becomes the
         corrupted input, and its expansion becomes the target.
      2. Itacism injection — up to MAX_ITACISM_EDITS vowel confusions, each gated on the corrupted
         form not being a real word (see build_normalize_lexicon).
    """
    lexicon = lexicon or set()
    examples: list[dict] = []
    n_crasis = n_itacism = 0

    for text in raw_texts:
        words = text.split()
        if not words:
            continue

        # ── Strategy 1: crasis. The raw text already contains the crasis form; the TARGET is the
        # expanded reading, so the pair is real data, not an injected artefact.
        # Crasis anchors are NOT subject to NORMALIZE_PROB: there are only ~130 in the entire NT,
        # so sampling them at 0.4 would throw away 60% of the scarcest signal in this pool. The
        # probability gate exists to keep the abundant itacism examples from swamping the mix.
        crasis_idx = [i for i, w in enumerate(words)
                      if _strip_accents(w.rstrip(",.:;·!?()")) in _CRASIS_EXPANSIONS]
        if crasis_idx:
            i = rng_py.choice(crasis_idx)
            target_words = list(words)
            target_words[i] = _CRASIS_EXPANSIONS[_strip_accents(words[i].rstrip(",.:;·!?()"))]
            examples.append({
                "task": "normalize",
                "input_text":  f"normalize: {text}",
                "target_text": " ".join(target_words),
            })
            n_crasis += 1
            continue

        # ── Strategy 2: itacism. Corrupt 1..MAX_ITACISM_EDITS words at random positions.
        if rng_py.random() > NORMALIZE_PROB:
            continue
        order = list(range(len(words)))
        rng_py.shuffle(order)
        corrupted = list(words)
        edits = 0
        budget = max(MIN_ITACISM_EDITS, int(len(words) * ITACISM_EDIT_RATE))
        for i in order:
            if edits >= budget:
                break
            word = words[i]
            for correct, wrong in rng_py.sample(_ITACISMS, len(_ITACISMS)):
                # Match the vowel sequence LITERALLY (i.e. only where it carries no diacritic) and
                # rewrite in place, so every other accent on the word survives. Corrupting the
                # accent-stripped form instead would make `normalize` a second `restore` task and
                # teach the model to conflate the two.
                if correct not in word:
                    continue
                candidate = word.replace(correct, wrong, 1)
                # Reject if the corruption produces another attested word — the target would then
                # be ambiguous and the example unlearnable.
                if candidate == word or _strip_accents(candidate) in lexicon:
                    continue
                corrupted[i] = candidate
                edits += 1
                break

        if edits:
            examples.append({
                "task": "normalize",
                "input_text":  f"normalize: {' '.join(corrupted)}",
                "target_text": text,
            })
            n_itacism += 1

    print(f"  Normalize (synthetic): {len(examples):,} examples "
          f"({n_crasis:,} crasis + {n_itacism:,} itacism, p={NORMALIZE_PROB})")
    return examples


# ── Synthetic restore pool (uncial → polytonic) ────────────────────────────────

def _strip_diacritics(text: str) -> str:
    """Return uppercase text with all Greek diacritical marks removed.

    Simulates ancient Greek uncial script (ALL-CAPS, no accents or breathings).
    Steps:
      1. NFD decompose (splits base char from combining diacritic).
      2. Drop all combining characters (accents, breathings, iota subscripts, etc.).
      3. Upper-case the base characters.
    """
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    return stripped.upper()


def build_restore_examples(raw_texts: list[str], rng_py) -> list[dict]:
    """Build restore examples by stripping diacritics from clean polytonic text.

    restore task: uncial/stripped form → fully accented polytonic form.
      Input:  "restore: ΕΝ ΑΡΧΗ ΗΝ Ο ΛΟΓΟΣ"
      Output: "ἐν ἀρχῇ ἦν ὁ λόγος"
    """
    examples: list[dict] = []
    for text in raw_texts:
        if not text.strip() or rng_py.random() > RESTORE_PROB:
            continue
        uncial = _strip_diacritics(text)
        if uncial == text.upper():
            # Text was already unaccented — skip (no restoration to learn)
            continue
        examples.append({
            "task": "restore",
            "input_text":  f"restore: {uncial}",
            "target_text": text,
        })

    print(f"  Restore (synthetic): {len(examples):,} examples "
          f"(uncial stripping, p={RESTORE_PROB})")
    return examples


# ── Master task pool builder ───────────────────────────────────────────────────

def build_task_pools(
    tokens_path: str,
    pericopes_path: str,
    proiel_dir: str | None,
) -> dict[str, list[dict]]:
    """Assemble all EIGHT balanced task pools.

    Sources (in order of addition to each pool):
      pos/lemma/denoise : Gospel corpus → PROIEL → MorphGNT all-27-books
      morphology        : MorphGNT all-27-books (compact tag encoding)
      normalize         : Real crasis expansions + dense lexicon-gated itacism on raw texts
      restore           : Synthetic uncial stripping on combined raw texts
      synoptic          : Gospel pericope pairs (tiny, upsampled)
      denoise           : All raw texts (Gospel + PROIEL + MorphGNT)

Both English-output pools (`translate`, `gloss`) were removed — see TASK_WEIGHTS for the measured
    post-mortem. Neither is a budget problem; this backbone has no English prior.
    """
    import random as _random
    rng_py = _random.Random(42)   # deterministic synthetic data

    # ── Gospel corpus (existing) ───────────────────────────────────────────────
    corpus_examples, corpus_raw = load_synoptic_pairs(tokens_path, pericopes_path)
    pos_pool      = [e for e in corpus_examples if e["task"] == "pos"]
    lemma_pool    = [e for e in corpus_examples if e["task"] == "lemma"]
    synoptic_pool = [e for e in corpus_examples if e["task"].startswith("synoptic")]
    denoise_texts = list(corpus_raw)

    # ── PROIEL CoNLL-U ─────────────────────────────────────────────────────────
    if proiel_dir:
        p_pos, p_lemma, p_raw = build_proiel_training(proiel_dir)
        pos_pool      += p_pos
        lemma_pool    += p_lemma
        denoise_texts += p_raw
    else:
        print("  [warn] PROIEL unavailable — pos/lemma/denoise from Gospel corpus only")

    # ── MorphGNT all 27 NT books ───────────────────────────────────────────────
    morphgnt_dir = resolve_morphgnt_dir()
    if morphgnt_dir:
        mgnt_morph, mgnt_pos, mgnt_lemma, mgnt_raw = build_morphgnt_training(morphgnt_dir)
        morphology_pool = mgnt_morph
        pos_pool        += mgnt_pos
        lemma_pool      += mgnt_lemma
        denoise_texts   += mgnt_raw
        normalize_lex   = build_normalize_lexicon(morphgnt_dir)
    else:
        print("  [warn] MorphGNT unavailable — morphology pool will be empty")
        morphology_pool = []
        normalize_lex   = set()

    # ── Synthetic normalize + restore (on combined raw texts) ──────────────────
    all_raw        = denoise_texts   # all raw Greek text accumulated so far
    normalize_pool = build_normalize_examples(all_raw, rng_py, normalize_lex)
    restore_pool   = build_restore_examples(all_raw, rng_py)

    # ── Denoise pool (raw texts → online span corruption) ─────────────────────
    denoise_pool = [{"task": "denoise", "raw_text": t} for t in denoise_texts]

    # Summary
    pools = {
        "pos":        pos_pool,
        "lemma":      lemma_pool,
        "morphology": morphology_pool,
        "normalize":  normalize_pool,
        "restore":    restore_pool,
        "synoptic":   synoptic_pool,
        "denoise":    denoise_pool,
    }
    active = {k: len(v) for k, v in pools.items() if v}
    empty  = [k for k, v in pools.items() if not v]
    print("  ── Task pools assembled: "
          + ", ".join(f"{k}={n:,}" for k, n in active.items()))
    if empty:
        print(f"  [warn] Empty pools (skipped due to unavailable data): {empty}")
    return pools


# ── Weighted multi-task sampling ─────────────────────────────────────

# Task order + per-task sampling weights. pos and denoise are the two hard, high-value pools
# (pos is the reported metric; denoise is the general-LM backbone) so they take the lion's share
# of every micro-batch; lemma is easy and synoptic is tiny, so they get one share each. Weights
# are relative — normalized against the non-empty pools at sample time.
# `gloss` was REMOVED after the rev-6 run measured it at 0.10 token accuracy and still climbing at
# ~0.001/1000 steps — it was never going to arrive. Two compounding reasons, both now understood:
# GreTa has no English pretraining, and joining multi-word MACULA glosses with "_" (needed to make
# the task word-aligned at all) turned the output into open-vocabulary generation over 19,710 types
# of which 63% are hapax, at 5.1 subword pieces each. Word-level English glossing needs an
# English-capable backbone; it is not a budget problem. Its 1.5 share is redistributed below.
_TASK_ORDER  = ("pos", "lemma", "morphology", "normalize", "restore", "synoptic", "denoise")
TASK_WEIGHTS = {
    "pos":        4.5,   # primary eval metric — must clear ~3 epochs (see step-budget block)
    "lemma":      3.5,   # large pool
    "morphology": 2.5,   # full parse codes — hardest task; still climbing at 75K, so +0.5
    "normalize":  1.5,   # crasis expansion + dense lexicon-gated itacism
    "restore":    2.0,   # diacritic restoration from uncial — proven learnable, +0.5
    "synoptic":   0.5,   # tiny pool (98 after length filtering) — already ~55× upsampled
    "denoise":    3.0,   # backbone LM — keep high
}   # sum = 17.5; see the step-budget block above for the resulting per-task epoch table


def sample_balanced_batch(pools: dict[str, list[dict]], batch_size: int, rng_py) -> list[dict]:
    """Draw a weighted multi-task micro-batch (sampling WITH REPLACEMENT within each task).

    Each of the `batch_size` slots independently picks a task ∝ TASK_WEIGHTS (restricted to the
    non-empty pools), then fills it via `random.choice` on that pool. Two invariants matter:

      * With replacement per task: a tiny pool (synoptic, ~155) must fill its slot across
        hundreds of thousands of batches; `random.sample` cannot upsample beyond a pool's size
        and would under-sample a minority task. `random.choice` gives every item equal probability
        on every draw and never runs out.
      * Weighted, not one-slot-each: pos/denoise draw at 3/8 each, lemma/synoptic at 1/8 each.
        Even the least-weighted task (synoptic, 1/8) appears ~4× per optimizer step (32 draws) —
        far above the frequency at which a task is catastrophically forgotten — so up-weighting
        pos triples its gradient at no cost to the minority tasks.
    """
    active = [t for t in _TASK_ORDER if pools.get(t)]
    if not active:
        return []
    weights = [TASK_WEIGHTS[t] for t in active]
    tasks   = rng_py.choices(active, weights=weights, k=batch_size)
    return [rng_py.choice(pools[t]) for t in tasks]


# Running count of examples dropped mid-training for exceeding MAX_SEQ_LEN, per task. Should stay
# at zero once filter_overlong_examples has run — a non-zero value at the end of a run means a pool
# is producing sequences the pre-filter did not see, and is printed rather than swallowed.
_OVERLONG_DROPS: dict[str, int] = {}


def filter_overlong_examples(
    pools: dict[str, list[dict]], tokenizer, max_len: int = MAX_SEQ_LEN
) -> dict[str, list[dict]]:
    """Drop examples whose input or target exceeds `max_len` tokens, and report what was removed.

    Doing this at pool-build time (rather than truncating in the collator) has two benefits: the
    model never sees a half-finished target, and the per-task epoch table printed at startup counts
    only examples that can actually be trained on.

    Measured against the real pools at MAX_SEQ_LEN=256, this removes ~24.5% of the `synoptic` pool
    (whole-pericope pairs run to 1,576 target tokens) and nothing else — `morphology` is already
    inside the budget thanks to the compact tag encoding.
    """
    out: dict[str, list[dict]] = {}
    for task, pool in pools.items():
        if task == "denoise":
            out[task] = pool          # denoise is corrupted online and clipped by construction
            continue
        kept = []
        for ex in pool:
            n_in  = len(tokenizer.encode(ex["input_text"]))
            n_tgt = len(tokenizer.encode(ex["target_text"]))
            if n_in <= max_len and n_tgt <= max_len:
                kept.append(ex)
        dropped = len(pool) - len(kept)
        if dropped:
            print(f"  [len-filter] {task}: dropped {dropped:,}/{len(pool):,} "
                  f"({dropped / len(pool):.1%}) over {max_len} tokens")
        out[task] = kept
    return out


def _collate_batch(examples: list[dict], tokenizer, rng, max_len: int = MAX_SEQ_LEN):
    """Build a training batch from a list of task examples.

    For 'denoise' the corruption is applied online on the raw Greek text (so every batch sees
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
            # Seq2seq instruction tasks (pos, lemma, morphology, normalize, restore, synoptic).
            # These are all ALIGNED tasks — one output unit per input word — so a
            # truncated target does not merely lose the tail, it teaches the model to stop early
            # and destroys the word↔output correspondence for every subsequent example. Drop the
            # example instead; `filter_overlong_examples` removes these at pool-build time, so
            # anything reaching here is a straggler worth counting rather than silently cutting.
            src = tokenizer(ex["input_text"],  return_tensors="pt")
            tgt = tokenizer(ex["target_text"], return_tensors="pt")
            input_ids = src["input_ids"][0]
            label_ids = tgt["input_ids"][0].clone()
            if input_ids.size(0) > max_len or label_ids.size(0) > max_len:
                _OVERLONG_DROPS[task] = _OVERLONG_DROPS.get(task, 0) + 1
                continue

        # Mask pad positions (id=0) in labels so they don't contribute to loss
        label_ids[label_ids == PAD_ID] = -100

        # Guard: skip any example where all labels are masked — would produce NaN loss
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


# ── POS Exact-Match evaluation (batched greedy decoding) ────────────

def _batch_pos_predict(model, tokenizer, texts: list[str], device: str) -> list[list[str]]:
    """Batched greedy-decode `pos: <text>` for a list of sentences.

    Right-padded (the tokenizer default): T5 is encoder-decoder — its encoder consumes
    input_ids with an attention_mask, so left-padding (a decoder-only requirement) is
    unnecessary and actively corrupts T5's relative-position encoding, producing garbage
    generations. Do NOT left-pad here.
    NO repetition penalties — POS tags legitimately repeat (e.g. `RA N- RA N-`).

    Returns a list of predicted-tag lists, one per input sentence.
    """
    import torch

    # Determine a per-batch max_new_tokens from the longest sentence in this chunk
    max_words = max(len(t.split()) for t in texts)
    max_new = min(200, max_words * 4 + 10)

    encoded = tokenizer(
        ["pos: " + t for t in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new,
            num_beams=1,
            do_sample=False,
        )

    # GreTa's SentencePiece tokenizer case-folds everything (Ἀρχὴ→ἀρχὴ, N-→n-), so the model
    # is trained on — and emits — lowercase tags. Upper-case them back: MorphGNT codes are
    # case-unique, so this is lossless and lets us score against the upper-case gold.
    return [
        [t.upper() for t in tokenizer.decode(seq, skip_special_tokens=True).split()]
        for seq in out
    ]


def evaluate_pos_em(model, tokenizer, records: list[dict], device: str,
                   batch_size: int = EVAL_BATCH_SIZE,
                   neutral_only: bool = False) -> dict[str, dict]:
    """Compute token-level accuracy + sentence-level Exact Match on a PROIEL split.

    Generates in mini-batches of `batch_size` for ~15x speedup over batch=1 on A10G.
    Returns {"nt": {...}, "classical": {...}, "overall": {...}} where each inner dict is
    {"tok": token_accuracy, "em": exact_match, "n": n_sentences}.

    `neutral_only=True` restricts scoring to tokens whose gold label is identical with and without
    PROIEL_LEMMA_OVERRIDES, and computes EM over those positions only. Use it for ANY comparison
    between models trained on different tagset conventions — see build_proiel_eval.
    """
    was_training = model.training
    model.eval()

    # [tok_correct, tok_total, em_count, n_sentences] per subset
    agg = {"nt": [0, 0, 0, 0], "classical": [0, 0, 0, 0]}
    samples: list[tuple[str, list[str], list[str]]] = []

    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        texts = [r["text"] for r in chunk]
        preds = _batch_pos_predict(model, tokenizer, texts, device)
        for r, pred in zip(chunk, preds):
            gold = r["gold"]
            keep = r.get("neutral") if neutral_only else None
            if keep is None:
                keep = [True] * len(gold)
            idx = [i for i, k in enumerate(keep) if k]
            tok_correct = sum(1 for i in idx if i < len(pred) and pred[i] == gold[i])
            a = agg[r["subset"]]
            a[0] += tok_correct
            a[1] += len(idx)
            a[2] += int(tok_correct == len(idx))   # EM over the scored positions
            a[3] += 1
            if len(samples) < 3:
                samples.append((r["subset"], gold, pred))

    if was_training:
        model.train()

    # Diagnostic: show what the model actually emits so a 0.0 EM is never ambiguous —
    # Greek prose (task not learned), wrong tag format, and <empty> generation look different.
    for sub, gold, pred in samples:
        print(f"    sample[{sub}] gold: {' '.join(gold[:10])}")
        print(f"    sample[{sub}] pred: {' '.join(pred[:10]) if pred else '<empty>'}")

    def pack(a: list[int]) -> dict:
        return {"tok": a[0] / max(1, a[1]), "em": a[2] / max(1, a[3]), "n": a[3]}

    nt, cl = agg["nt"], agg["classical"]
    overall = [nt[0] + cl[0], nt[1] + cl[1], nt[2] + cl[2], nt[3] + cl[3]]
    return {"nt": pack(nt), "classical": pack(cl), "overall": pack(overall)}


# ── Generic per-task evaluation (all non-denoise tasks are word-aligned) ────────
# pos / lemma / morphology / normalize / restore all emit one whitespace unit per input
# word, so a single token-accuracy + exact-match evaluator covers every one of them. Mirrors the
# structure of evaluate_tagging in app_koine_hexapla.py.

def _batch_predict(model, tokenizer, texts: list[str], device: str, prefix: str,
                   upper: bool) -> list[list[str]]:
    """Batched greedy decode of `<prefix><text>` for a list of inputs.

    Right-padded and greedy for the same reasons as _batch_pos_predict: T5's encoder takes an
    attention mask (left-padding corrupts its relative-position encoding), and repetition penalties
    would corrupt outputs whose units legitimately repeat.
    """
    import torch

    max_words = max(len(t.split()) for t in texts)
    max_new = min(400, max_words * 8 + 10)

    encoded = tokenizer([prefix + t for t in texts], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        out = model.generate(input_ids=encoded["input_ids"],
                             attention_mask=encoded["attention_mask"],
                             max_new_tokens=max_new, num_beams=1, do_sample=False)
    decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in out]
    return [(d.upper() if upper else d).split() for d in decoded]


def evaluate_tagging(model, tokenizer, records: list[dict], device: str, prefix: str,
                     upper: bool = False, batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Token-accuracy + exact-match for any word-aligned task, WITH a copy baseline.

    Returns `tok`, `em`, `copy`, `lift`, and `edit` — and `lift`/`edit` are the ones that mean
    something for near-copy tasks.

    Raw token accuracy is a trap whenever the correct output resembles the input. The `normalize`
    task scored a healthy-looking 0.86 for 54,000 steps while having learned nothing at all: only
    1-3 words per sentence are corrupted, so ~90% of tokens are already correct in the input and
    echoing it scores 0.90. The model sat *below* that no-op baseline and no one could see it.

      * `copy` — what you get by emitting the input verbatim.
      * `lift` — tok − copy. This is the real signal. `restore` scores +0.665 (genuine skill);
        `normalize` scored −0.011 (worse than a no-op).
      * `edit` — accuracy restricted to the positions that actually differ between input and gold,
        i.e. the only positions where the task is being asked to do anything.

    A task whose `lift` is near zero has not been learned, no matter how high `tok` is.
    """
    if not records:
        return {"tok": 0.0, "em": 0.0, "copy": 0.0, "lift": 0.0, "edit": 0.0, "n": 0}
    was_training = model.training
    model.eval()

    tok_correct = tok_total = em = copy_correct = 0
    edit_correct = edit_total = 0
    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        preds = _batch_predict(model, tokenizer, [r["text"] for r in chunk], device, prefix, upper)
        for r, pred in zip(chunk, preds):
            gold = r["gold"]
            src  = (r["text"].upper() if upper else r["text"].lower()).split()
            tok_correct  += sum(1 for g, p in zip(gold, pred) if g == p)
            copy_correct += sum(1 for g, s in zip(gold, src) if g == s)
            tok_total    += len(gold)
            em           += int(pred == gold)
            # Positions the task is actually being asked to change.
            for i, g in enumerate(gold):
                if i < len(src) and src[i] == g:
                    continue
                edit_total += 1
                if i < len(pred) and pred[i] == g:
                    edit_correct += 1

    if was_training:
        model.train()
    tok  = tok_correct / max(1, tok_total)
    copy = copy_correct / max(1, tok_total)
    return {"tok": tok, "em": em / max(1, len(records)), "copy": copy, "lift": tok - copy,
            "edit": edit_correct / max(1, edit_total), "n": len(records)}


# Gates for checkpoint selection. POS-NT is set just under published Koine-T5's 0.966 (this run's
# corrected tagset mapping shifts ~2-5% of gold labels, so exact parity is not the right bar);
# GATE_LEMMA is Koine-T5's measured lemma dev-accuracy under this harness.
GATE_POS_NT = 0.950
GATE_LEMMA  = 0.760

# Tasks whose mean score decides between gate-passing checkpoints, each paired with the metric to
# rank it on. morphology is not a copy task (input is Greek, output is tag codes) so raw accuracy is
# honest there. normalize and restore ARE near-copy tasks, so they are ranked on `lift` — accuracy
# over the copy-the-input baseline. Ranking normalize on `tok` is what made a model with NEGATIVE
# lift look like a 0.86 performer for 54,000 steps.
_SECONDARY_TASKS = {"morphology": "tok", "restore": "lift", "normalize": "lift"}


def evaluate_all(model, tokenizer, pos_eval: list[dict], task_evals: dict[str, list[dict]],
                 device: str) -> dict:
    """Score every task and return metrics plus a no-regression selection key.

    ``select_key`` = (1, mean secondary score) once POS-NT and lemma clear their gates, else
    (0, pos_tok). Any gate-passer outranks any non-passer, so the secondary tasks can only ever be
    optimised *subject to* the analysis tasks holding — the same mechanism as
    app_koine_hexapla.py's evaluate_all. Before the gates are met, POS token-accuracy still tracks
    early progress so selection is never blind.

    Selecting on POS alone (the previous behaviour) makes best/ the best *tagger*, not the best
    omni model — with 8 tasks that is the wrong objective.
    """
    pos_m = evaluate_pos_em(model, tokenizer, pos_eval, device)
    per_task = {"pos": {"tok": pos_m["overall"]["tok"], "em": pos_m["overall"]["em"],
                        "n": pos_m["overall"]["n"]}}
    for task, records in task_evals.items():
        prefix = f"{task}: "
        per_task[task] = evaluate_tagging(model, tokenizer, records, device, prefix,
                                          upper=(task == "morphology"))

    lemma_tok = per_task.get("lemma", {}).get("tok", 0.0)
    gated = pos_m["nt"]["tok"] >= GATE_POS_NT and lemma_tok >= GATE_LEMMA

    secondary = [per_task[t][metric] for t, metric in _SECONDARY_TASKS.items()
                 if per_task.get(t, {}).get("n")]
    secondary_score = sum(secondary) / len(secondary) if secondary else 0.0

    select_key = (1.0, secondary_score) if gated else (0.0, pos_m["overall"]["tok"])
    return {"pos": pos_m, "tasks": per_task, "gated": gated,
            "secondary_score": secondary_score, "select_key": select_key}


def carve_holdout(pools: dict[str, list[dict]], per_task: int = HOLDOUT_PER_TASK,
                  seed: int = 7) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split a deterministic held-out slice off each non-denoise pool for evaluation.

    Returns (train_pools, task_evals). Held-out examples are REMOVED from the training pools, so
    the secondary scores that drive checkpoint selection are not measured on trained-on data.
    `pos` is excluded — it is evaluated on PROIEL dev, which is already disjoint from PROIEL train.
    """
    import random as _random
    rng = _random.Random(seed)

    train_pools: dict[str, list[dict]] = {}
    task_evals: dict[str, list[dict]] = {}
    for task, pool in pools.items():
        if task in ("denoise", "pos", "synoptic") or len(pool) < per_task * 4:
            train_pools[task] = pool
            continue
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        held = set(idx[:per_task])
        train_pools[task] = [e for i, e in enumerate(pool) if i not in held]
        # GreTa's SentencePiece case-folds on encode, so the model was trained to emit lowercase
        # and can only ever be scored against lowercase gold. morphology is the exception: its
        # tags are case-unique, so pred is upper-cased back instead (see evaluate_all).
        prefix = f"{task}: "
        fold = (lambda s: s.upper()) if task == "morphology" else (lambda s: s.lower())
        task_evals[task] = [
            {"text": pool[i]["input_text"].removeprefix(prefix),
             "gold": fold(pool[i]["target_text"]).split()}
            for i in sorted(held)
        ]
    return train_pools, task_evals


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

def _prune_checkpoints(output_dir: Path, keep: int, fingerprint: str) -> None:
    """Keep only the `keep` most-recent step-N checkpoints OF THIS RUN (matching fingerprint).

    Scoped to the current fingerprint so a fresh run started in a volume that still holds a
    previous config's (possibly higher-numbered) step dirs never deletes its own checkpoint to
    keep a stale one. Foreign checkpoints are left untouched (they are surfaced + ignored at
    resume time), and best/ / final/ are never step-N dirs so they are always safe.
    """
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


def _training_loop(
    model,
    tokenizer,
    pools: dict[str, list[dict]],
    eval_records: list[dict],
    task_evals: dict[str, list[dict]],
    output_dir: Path,
    device: str,
    volume=None,
) -> None:
    """Main training loop: balanced 8-task sampling + gated multi-task eval + best selection."""
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

    # Linear warmup + cosine decay to zero.
    # CRITICAL: scheduler.step() fires once per OPTIMIZER update (every GRAD_ACCUM micro-steps),
    # so LambdaLR feeds this the optimizer-step count, not the micro-step count the loop uses.
    # WARMUP_STEPS / MAX_STEPS are micro-steps, so convert to optimizer steps here — otherwise
    # warmup runs GRAD_ACCUM× too long and the cosine (denominator ~MAX_STEPS while the counter
    # only reaches MAX_STEPS/GRAD_ACCUM) never anneals.
    warmup_opt = max(1, WARMUP_STEPS // GRAD_ACCUM)
    total_opt  = max(warmup_opt + 1, MAX_STEPS // GRAD_ACCUM)

    def lr_lambda(opt_step: int) -> float:
        if opt_step < warmup_opt:
            return opt_step / warmup_opt
        progress = min(1.0, (opt_step - warmup_opt) / (total_opt - warmup_opt))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    # bfloat16 needs no GradScaler (that guards float16 underflow); bf16 autocast gives the same
    # memory saving without float16's NaN risk.
    USE_BF16 = True

    # Deterministic eval subset (stable across evals → comparable token-acc for model selection)
    eval_subset = _select_eval_subset(eval_records, EVAL_MAX_PER_SUBSET) if eval_records else []
    best_key = (-1.0, -1.0)   # gated selection key from evaluate_all — see that docstring

    # Run fingerprint: a short hash of the config that defines "the same run". Auto-resume only
    # from checkpoints stamped with this exact fingerprint, so a stale or foreign checkpoint on the
    # persistent volume cannot resume into a changed config, or silently no-op the run because its
    # step number already exceeds MAX_STEPS.
    fingerprint = hashlib.sha1(json.dumps({
        "max_steps": MAX_STEPS, "lr": LR, "batch": BATCH_SIZE, "accum": GRAD_ACCUM,
        "warmup": WARMUP_STEPS, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "seq": MAX_SEQ_LEN,
        "weights": {t: TASK_WEIGHTS[t] for t in _TASK_ORDER},
        # Bump `rev` to force a fresh run — superseded checkpoints are ignored, not resumed.
        "rev": 7,   # omni v3: gloss dropped, dense normalize, lift-aware selection
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
        # compatible = same fingerprint and still short of MAX_STEPS (i.e. genuine unfinished work)
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
                # Restore optimizer + scheduler + best-score + RNG so the LR schedule continues
                # smoothly instead of re-warming from zero on resume.
                state_file = latest / "training_state.pt"
                if state_file.exists():
                    state = torch.load(str(state_file), map_location=device, weights_only=False)
                    optimizer.load_state_dict(state["optimizer"])
                    scheduler.load_state_dict(state["scheduler"])
                    step     = int(state.get("step", latest_step))
                    best_key = tuple(state.get("best_key", (-1.0, -1.0)))
                    try:
                        torch.set_rng_state(state["torch_rng"].cpu())
                        rng.bit_generator.state = state["numpy_rng"]
                        py_rng.setstate(state["python_rng"])
                    except Exception:
                        pass  # RNG restore is best-effort; weights + optimizer are what matter
                    print(f"Resuming {latest.name} (fp={fingerprint}): step={step}, "
                          f"best_key={best_key}, lr={scheduler.get_last_lr()[0]:.2e}")
                else:
                    print(f"Resuming {latest.name} weights only (no training_state; LR re-warms)")

    # Surface (but do not touch) checkpoints we are deliberately ignoring
    if step == 0:
        foreign = [d.name for d in step_dirs
                   if not (d / "run_fp.txt").exists()
                   or (d / "run_fp.txt").read_text().strip() != fingerprint]
        if foreign:
            shown = ", ".join(sorted(foreign)[:6]) + ("…" if len(foreign) > 6 else "")
            print(f"Fresh start (fp={fingerprint}); ignoring {len(foreign)} checkpoint(s) from a "
                  f"different run config: {shown}")

    # Report pool sizes + per-task epoch coverage under the weighted sampler.
    # NOTE: this table is the single best early-warning signal in the whole script. The 9-task
    # regression it was written to catch printed "pos ~0.92 epochs" here and went unread.
    active_tasks = [t for t in _TASK_ORDER if pools.get(t)]
    w_total = sum(TASK_WEIGHTS[t] for t in active_tasks) or 1.0
    print(f"Starting training from step {step} → {MAX_STEPS}")
    print("  Task pools: " + " | ".join(f"{t}={len(pools.get(t, [])):,}" for t in _TASK_ORDER))
    for t in active_tasks:
        seen = MAX_STEPS * BATCH_SIZE * TASK_WEIGHTS[t] / w_total
        pool_n = max(1, len(pools[t]))
        print(f"    {t:9s} ~{seen / pool_n:5.2f} epochs  ({seen:,.0f} examples over {pool_n:,})")
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

        # Multi-task validation + gated best-checkpoint selection. Token accuracy (not EM) drives
        # the pre-gate signal: full-sentence EM is ~0 early on and cannot discriminate checkpoints.
        if eval_subset and step % EVAL_STEPS == 0:
            m   = evaluate_all(model, tokenizer, eval_subset, task_evals, device)
            pos = m["pos"]
            print(f"  [eval step {step}] POS tok={pos['overall']['tok']:.3f} "
                  f"(EM={pos['overall']['em']:.3f})  |  NT tok={pos['nt']['tok']:.3f} "
                  f"em={pos['nt']['em']:.3f} (n={pos['nt']['n']})  |  "
                  f"Classical tok={pos['classical']['tok']:.3f} "
                  f"em={pos['classical']['em']:.3f} (n={pos['classical']['n']})")
            print("    tasks: " + "  ".join(
                (f"{t}={v['tok']:.3f}(lift{v['lift']:+.3f})"
                 if t in _SECONDARY_TASKS and _SECONDARY_TASKS[t] == "lift"
                 else f"{t}={v['tok']:.3f}")
                for t, v in m["tasks"].items() if t != "pos")
                + f"  |  gated={m['gated']} secondary={m['secondary_score']:.3f}")
            if m["select_key"] > best_key:
                best_key = m["select_key"]
                model.save_pretrained(str(best_dir))
                (best_dir / "metrics.json").write_text(json.dumps(
                    {"step": step, "select_key": list(best_key), **m}, indent=2, default=str))
                _stamp(best_dir)
                print(f"  [best] select_key={best_key} (gated={m['gated']}) → saved {best_dir}")
                if volume is not None:
                    _commit()

        if step % SAVE_STEPS == 0:
            ckpt = output_dir / f"step-{step}"
            model.save_pretrained(str(ckpt))
            _stamp(ckpt)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_key": list(best_key),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": rng.bit_generator.state,
                "python_rng": py_rng.getstate(),
            }, str(ckpt / "training_state.pt"))
            _prune_checkpoints(output_dir, CKPT_KEEP, fingerprint)
            print(f"  [checkpoint] saved → {ckpt}  (kept last {CKPT_KEEP})")
            if volume is not None:
                _commit()

    # Final save
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    _stamp(final_dir)
    print(f"\nTraining complete. Final adapters: {final_dir}")
    if best_key[0] >= 0:
        print(f"Best adapter (select_key={best_key}): {best_dir}")
    if _OVERLONG_DROPS:
        # Should be empty — filter_overlong_examples runs before training. A non-zero count means
        # a pool is emitting sequences the pre-filter never saw; surface it rather than hide it.
        print(f"[warn] examples dropped mid-training for exceeding {MAX_SEQ_LEN} tokens: "
              f"{dict(sorted(_OVERLONG_DROPS.items()))}")
    if volume is not None:
        _commit()


# ── Inference ────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    input_text: str,
    task: str = "denoise",
    device: str = "cpu",
) -> str:
    """Generate a response for any task.

    Decoding strategy is task-conditional:
      * pos / lemma / morphology → greedy (outputs legitimately repeat tokens;
                                   penalties would corrupt tag/lemma sequences).
      * all others               → Contrastive Search (anti-degeneration for
                                   free-form Greek or English output).

    Tasks:
      - denoise:           NO prefix — pass corrupted text with <extra_id_N> masks directly
      - pos:               prefix ``pos: <greek text>``
      - lemma:             prefix ``lemma: <greek text>``
      - morphology:        prefix ``morphology: <greek text>``
      - normalize:         prefix ``normalize: <crasis/itacism text>``
      - restore:           prefix ``restore: <UNCIAL TEXT>``
      - synoptic_mk_to_mt: prefix ``synoptic mark_to_matt: <mark text>``
      - synoptic_mk_to_lk: prefix ``synoptic mark_to_luke: <mark text>``
    """
    import torch

    task_prefixes = {
        # denoise trains on raw corrupted ids with NO prefix (see _collate_batch);
        # prefixing at inference mismatches training and degrades the fills.
        "denoise":           "",
        "pos":               "pos: ",
        "lemma":             "lemma: ",
        "morphology":        "morphology: ",
        "normalize":         "normalize: ",
        "restore":           "restore: ",
        "synoptic_mk_to_mt": "synoptic mark_to_matt: ",
        "synoptic_mk_to_lk": "synoptic mark_to_luke: ",
    }

    prefix = task_prefixes.get(task, "")
    full_input = prefix + input_text if not input_text.startswith(prefix) else input_text

    inputs = tokenizer(full_input, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)

    # Decoding is chosen per task, and the choice matters as much as the weights do.
    #
    # Previously everything except pos/lemma/morphology fell through to contrastive search with
    # repetition_penalty=1.25 + no_repeat_ngram_size=3 + encoder_no_repeat_ngram_size=3. For
    # normalize and restore — near-copy tasks whose correct output repeats the input — forbidding
    # repeated 3-grams makes the right answer literally unreachable. Degenerate output under that
    # config is a decoding artefact, not evidence about the trained model.
    if task in _GREEDY_TASKS:
        # Aligned, deterministic tasks: one output unit per input word, units legitimately repeat.
        gen_kwargs = dict(max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=1, do_sample=False)
    elif task.startswith("synoptic"):
        # Paraphrase with a single best answer — beam search, no repetition penalties (Greek
        # parallel text shares long spans with its source by definition).
        gen_kwargs = dict(max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=4, early_stopping=True)
    else:
        # denoise only: open-ended span infilling, where contrastive search belongs.
        gen_kwargs = dict(
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            penalty_alpha=GEN_PENALTY_ALPHA,
            top_k=GEN_TOP_K,
            repetition_penalty=GEN_REP_PENALTY,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
            early_stopping=True,
            trust_remote_code=True,
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


@app.local_entrypoint()  # type: ignore[misc]
def upload_morphgnt() -> None:
    """Upload the local MorphGNT corpus (all 27 books) to the Modal data volume.

    Safe to re-run (skips present files).
    Requires: data/raw/morphgnt/ containing the *-morphgnt.txt files.
    """
    import subprocess

    local = Path(MORPHGNT_LOCAL_DIR)
    if not local.exists():
        print(f"ERROR: {local} not found. Make sure data/raw/morphgnt/ is populated.")
        sys.exit(1)

    print(f"Uploading {local} → Modal volume '{DATA_VOLUME}:/morphgnt' (--force) ...")
    result = subprocess.run(
        ["modal", "volume", "put", "--force", DATA_VOLUME, str(local), "/morphgnt"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Upload failed:\n{result.stderr}")
        sys.exit(1)
    print(f"MorphGNT uploaded to '{DATA_VOLUME}:/morphgnt' (all 27 NT books).")


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes=_VOLUMES,
    timeout=TIMEOUT,
) if modal is not None else None
def train() -> None:
    """Full Koine-T5 Omni training run on Modal A10G GPU.

    Trains on EIGHT balanced task pools (pos / lemma / morphology / gloss / normalize /
    restore / synoptic / denoise), fed by:
      - Gospel corpus (processed parquets)
      - UD_Ancient_Greek-PROIEL (downloaded at runtime)
      - MorphGNT all 27 NT books (data/raw/morphgnt/, pre-staged on disk)
      - MACULA Greek TSV (downloaded at runtime, CC BY-SA)
      - Synthetic normalize/restore examples (generated from raw corpus)

    All external downloads are graceful-fallback: if a resource is unavailable,
    its pool is skipped and training continues on remaining pools.
    Outputs LoRA adapters to /outputs/koine_t5_omni/ (step-N + best/ + final/).
    """
    device     = "cuda"
    output_dir = Path("/outputs/koine_t5_omni")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path    = "/data/processed/tokens.parquet"
    pericopes_path = "/data/processed/pericopes.parquet"

    # Phase 1: tokenizer (register the T5 sentinels)
    print("Phase 1: Building Koine-T5 Omni tokenizer with sentinel tokens...")
    tokenizer = build_tokenizer()
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    print(f"  Sentinel <extra_id_0> -> id {tokenizer.convert_tokens_to_ids('<extra_id_0>')}")

    # Model: bfloat16 + LoRA (same architecture as v1 — 8 tasks need no arch change)
    print(f"\nPhase 2: Loading GreTa base (bfloat16, LoRA r={LORA_R}, encoder+decoder)...")
    model = load_model_with_lora(tokenizer, device=device)

    # Phase 3: resolve PROIEL (graceful fallback if unavailable)
    print("\nPhase 3: Resolving PROIEL treebank...")
    proiel_dir = resolve_proiel_dir(allow_download=True)
    if proiel_dir:
        print(f"  PROIEL available at: {proiel_dir}")
    else:
        print("  [warn] PROIEL NOT available — falling back to Synoptic+MorphGNT-only.")

    # Phase 4: build all eight task pools + eval sets
    print("\nPhase 4: Building task pools (8 tasks)...")
    pools = build_task_pools(tokens_path, pericopes_path, proiel_dir)

    # Drop anything that cannot fit MAX_SEQ_LEN *before* the epoch table is computed, so the
    # reported coverage counts only trainable examples and no target is ever cut mid-sequence.
    pools = filter_overlong_examples(pools, tokenizer)

    # Hold out a slice of each secondary pool so gated selection is not scored on trained-on data.
    pools, task_evals = carve_holdout(pools)
    if task_evals:
        print("  Held-out eval sets: "
              + ", ".join(f"{t}={len(v)}" for t, v in sorted(task_evals.items())))

    eval_records = build_proiel_eval(proiel_dir, "dev") if proiel_dir else []
    print(f"  Eval records (PROIEL dev): {len(eval_records)}")

    # Ship the morphology decode table next to the adapter so the compact tags are invertible.
    morphgnt_dir = resolve_morphgnt_dir()
    if morphgnt_dir:
        import json as _json
        table = build_morph_decode_table(morphgnt_dir)
        (output_dir / "morph_tags.json").write_text(_json.dumps(table, indent=2))
        print(f"  Morphology decode table: {len(table)} tags → {output_dir / 'morph_tags.json'}")

    # Phase 5: train
    print("\nStarting Koine-T5 Omni training loop...")
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME) if modal is not None else None
    _training_loop(model, tokenizer, pools, eval_records, task_evals, output_dir, device,
                   volume=output_vol)

    print("\nDone. Download adapters:")
    print(f"  modal volume get {OUTPUT_VOLUME} koine_t5_omni/best  models/koine_t5_omni/best")
    print(f"  modal volume get {OUTPUT_VOLUME} koine_t5_omni/final models/koine_t5_omni/final")
    print("\nLocal demo:")
    print("  python modal/app_koine_t5_omni.py demo models/koine_t5_omni/best")


def _load_adapter(tokenizer, adapter_path: str, device: str):
    """Load GreTa + a saved LoRA adapter for evaluation."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM

    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, dtype=torch.bfloat16)
    if hasattr(base.config, "tie_word_embeddings"):
        base.config.tie_word_embeddings = False
    base.config.vocab_size = len(tokenizer)
    return PeftModel.from_pretrained(base, adapter_path).to(device).eval()


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes=_VOLUMES,
    timeout=3600,
) if modal is not None else None
def run_test(adapter: str = "/outputs/koine_t5_omni/best") -> None:
    """Score a trained adapter ONCE on the PROIEL **test** split — the headline numbers.

    Kept separate from training on purpose: `best/` is selected on dev, so quoting a dev number as
    the result would be selection-contaminated. Run this after training completes.
    """
    import json

    device    = "cuda"
    tokenizer = build_tokenizer()
    model     = _load_adapter(tokenizer, adapter, device)

    proiel_dir = resolve_proiel_dir(allow_download=True)
    if not proiel_dir:
        print("PROIEL unavailable — cannot run the held-out test evaluation.")
        return

    records = build_proiel_eval(proiel_dir, "test")
    print(f"PROIEL test: {len(records):,} sentences")
    m = evaluate_pos_em(model, tokenizer, records, device)
    print(f"\n  POS  NT        tok={m['nt']['tok']:.4f}  em={m['nt']['em']:.4f}  "
          f"(n={m['nt']['n']})")
    print(f"  POS  Classical tok={m['classical']['tok']:.4f}  em={m['classical']['em']:.4f}  "
          f"(n={m['classical']['n']})")
    print(f"  POS  pooled    tok={m['overall']['tok']:.4f}  em={m['overall']['em']:.4f}  "
          f"(n={m['overall']['n']})")

    out = Path(adapter) / "test_metrics.json"
    out.write_text(json.dumps({"split": "test", "adapter": adapter, **m}, indent=2))
    print(f"\nWrote {out}")
    _commit()


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes=_VOLUMES,
    timeout=3600,
) if modal is not None else None
def reeval_v1(adapter: str = "/outputs/koine_t5_v1/best") -> None:
    """Re-score the published Koine-T5 adapter under THIS script's corrected tagset mapping.

    `PROIEL_LEMMA_OVERRIDES` changes the gold label for ~2-5% of PROIEL tokens, so omni's POS
    number is not directly comparable to the 0.966 NT that published Koine-T5 reported under the
    old mapping. This produces an apples-to-apples baseline on the same gold. It does NOT restate
    the published Koine-T5 result — that number stands as measured.
    """
    import json

    device    = "cuda"
    tokenizer = build_tokenizer()
    model     = _load_adapter(tokenizer, adapter, device)

    proiel_dir = resolve_proiel_dir(allow_download=True)
    if not proiel_dir:
        print("PROIEL unavailable — cannot re-evaluate.")
        return

    for split in ("dev", "test"):
        records = build_proiel_eval(proiel_dir, split)
        m = evaluate_pos_em(model, tokenizer, records, device)
        print(f"\n[koine-t5 v1 @ corrected mapping — PROIEL {split}, n={len(records):,}]")
        print(f"  NT        tok={m['nt']['tok']:.4f}  em={m['nt']['em']:.4f}")
        print(f"  Classical tok={m['classical']['tok']:.4f}  em={m['classical']['em']:.4f}")
        print(f"  pooled    tok={m['overall']['tok']:.4f}  em={m['overall']['em']:.4f}")
        Path(f"/outputs/koine_t5_omni/v1_baseline_{split}.json").write_text(
            json.dumps({"split": split, "adapter": adapter, **m}, indent=2))
    _commit()


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes=_VOLUMES,
    timeout=7200,
) if modal is not None else None
def compare(omni: str = "/outputs/koine_t5_omni/best",
            v1: str = "/outputs/koine_t5_v1/best") -> None:
    """Compare omni against Koine-T5 v1 on PROIEL test, scored two ways.

    Scoring both models against the override-corrected gold is NOT a fair comparison, even though
    it uses identical code and the identical split: the overrides encode the tagset convention omni
    was trained on, so v1 is marked wrong for answering in the convention it was actually taught.
    That penalty lands on 39.3% of NT and 85.5% of Classical test SENTENCES, which destroys the
    exact-match comparison specifically.

    This prints both views so the gap between them is visible:
      * FULL    — all tokens, corrected gold. Flatters omni; do not quote as a model comparison.
      * NEUTRAL — only tokens where both conventions agree (94.4% of them). This is the number
                  that reflects tagging skill rather than answer-key alignment.
    """
    import json

    device    = "cuda"
    tokenizer = build_tokenizer()
    proiel_dir = resolve_proiel_dir(allow_download=True)
    if not proiel_dir:
        print("PROIEL unavailable.")
        return
    records = build_proiel_eval(proiel_dir, "test")
    n_tok  = sum(len(r["gold"]) for r in records)
    n_keep = sum(sum(r["neutral"]) for r in records)
    print(f"PROIEL test: {len(records):,} sentences, {n_tok:,} tokens "
          f"({n_keep:,} convention-neutral = {n_keep / n_tok:.1%})\n")

    out: dict[str, dict] = {}
    for name, path in (("omni", omni), ("koine-t5-v1", v1)):
        model = _load_adapter(tokenizer, path, device)
        out[name] = {
            "full":    evaluate_pos_em(model, tokenizer, records, device),
            "neutral": evaluate_pos_em(model, tokenizer, records, device, neutral_only=True),
        }
        del model

    for mode in ("full", "neutral"):
        label = ("FULL (corrected gold — flatters omni)" if mode == "full"
                 else "NEUTRAL (convention-agnostic — the fair comparison)")
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(f"{'model':14} {'NT tok':>8} {'NT EM':>8} {'Cl tok':>8} {'Cl EM':>8} "
              f"{'pool tok':>9} {'pool EM':>8}")
        for name in out:
            m = out[name][mode]
            print(f"{name:14} {m['nt']['tok']:8.4f} {m['nt']['em']:8.4f} "
                  f"{m['classical']['tok']:8.4f} {m['classical']['em']:8.4f} "
                  f"{m['overall']['tok']:9.4f} {m['overall']['em']:8.4f}")
        a, b = out["omni"][mode], out["koine-t5-v1"][mode]
        print(f"{'Δ (pp)':14} {(a['nt']['tok']-b['nt']['tok'])*100:+8.1f} "
              f"{(a['nt']['em']-b['nt']['em'])*100:+8.1f} "
              f"{(a['classical']['tok']-b['classical']['tok'])*100:+8.1f} "
              f"{(a['classical']['em']-b['classical']['em'])*100:+8.1f} "
              f"{(a['overall']['tok']-b['overall']['tok'])*100:+9.1f} "
              f"{(a['overall']['em']-b['overall']['em'])*100:+8.1f}")

    Path("/outputs/koine_t5_omni/comparison_test.json").write_text(json.dumps(out, indent=2))
    print("\nQuote the NEUTRAL row. Report the FULL row only alongside the caveat above.")
    _commit()


# ── Local Demo / Inference ─────────────────────────────────────────────────────

def _run_demo(adapter_path: str | None = None) -> None:
    """Run a local inference demo comparing GreTa base vs Koine-T5."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Koine-T5 demo on {device}\n")

    tokenizer = build_tokenizer(local_files_only=True)

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
            "note": "Expected: v- ra n- rp ra n-",
        },
        {
            "task": "lemma",
            "name": "Matthew 6:9 — Lemmatization",
            "text": "Πάτερ ἡμῶν ὁ ἐν τοῖς οὐρανοῖς",
            "note": "Expected: πατήρ ἐγώ ὁ ἐν ὁ οὐρανός",
        },
        {
            "task": "morphology",
            "name": "Luke 1:46 — Full Morphology Tagging (compact encoding)",
            "text": "μεγαλύνει ἡ ψυχή μου τὸν κύριον",
            "note": "Expected: v-3pais ra n-nsf rp-gs ra n-asm  (decode via morph_tags.json)",
        },
        {
            "task": "normalize",
            "name": "Crasis / Spelling Normalization",
            "text": "κἀγὼ εἶπον αὐτῷ",
            "note": "Expected: καὶ ἐγὼ εἶπον αὐτῷ",
        },
        {
            "task": "restore",
            "name": "Uncial to Polytonic Restoration",
            "text": "ΕΝ ΑΡΧΗ ΗΝ Ο ΛΟΓΟΣ",
            "note": "Expected: ἐν ἀρχῇ ἦν ὁ λόγος",
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
        adapter = sys.argv[2] if len(sys.argv) > 2 else "models/koine_t5_omni/best"
        _run_demo(adapter_path=adapter)
    else:
        print(__doc__)
        print("\nTo run the demo locally:  python modal/app_koine_t5_omni.py demo [adapter_path]")
        print("To train on Modal:        modal run modal/app_koine_t5_omni.py::train")
        print("To upload corpus first:   modal run modal/app_koine_t5_omni.py::upload_corpus")
        print("To upload MorphGNT:       modal run modal/app_koine_t5_omni.py::upload_morphgnt")
        print("To pre-stage PROIEL:      modal run modal/app_koine_t5_omni.py::upload_proiel")
        print("Held-out test numbers:    modal run modal/app_koine_t5_omni.py::run_test")
        print("v1 baseline (same gold):  modal run modal/app_koine_t5_omni.py::reeval_v1")
