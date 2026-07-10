"""Configuration dataclasses for all SynoptiQ training phases.

Uses frozen dataclasses for immutability. All paths are relative
to the project root unless specified as absolute.

Rationale for frozen dataclasses over YAML-only configs:
  - Type safety: mypy catches wrong parameter types at check time
  - IDE auto-complete: works without runtime inspection
  - Immutability: no accidental mutation during training
  - Serialization: convert to dict via dataclasses.asdict() for W&B logging

Usage:
    config = DAPTConfig()                          # All defaults
    config = DAPTConfig(max_steps=50_000)          # Override one field

    # Load from YAML:
    import yaml
    from dataclasses import fields
    raw = yaml.safe_load(open("configs/training.yaml"))
    config = TrainingConfig(**{k: v for k, v in raw.items()
                                if k in {f.name for f in fields(TrainingConfig)}})
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration."""

    # Paths
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    external_dir: Path = Path("data/external")

    # Alignment
    alignment_method: Literal["needleman_wunsch", "smith_waterman"] = "needleman_wunsch"
    gap_open_penalty: float = -5.0
    gap_extend_penalty: float = -0.5
    lemma_match_score: float = 2.0
    surface_match_bonus: float = 1.0
    pos_match_bonus: float = 0.5

    # Splits (pericope-atomic — never split a pericope across splits)
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20
    random_seed: int = 42

    # Augmentation
    n_bootstrap_samples: int = 100
    moving_window_size: int = 5
    moving_window_stride: int = 2
    scribal_error_rate: float = 0.01  # For synthetic scribal noise simulation

    def __post_init__(self) -> None:
        """Validate fractions sum to 1.0."""
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            msg = f"train_frac + val_frac + test_frac must sum to 1.0, got {total}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""

    # Base model — confirmed correct HuggingFace ID as of 2026-07
    base_model: str = "bowphs/GreTa"
    model_type: Literal["t5-small", "t5-base"] = "t5-base"
    max_seq_length: int = 512
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12

    # Dropouts
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # LoRA — used for task-specific adaptation (Phase 2 multi-task, Phase 3+)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "k", "v", "o")

    # Q reconstruction (FiD)
    fid_num_beams: int = 5
    fid_temperature: float = 1.0
    fid_max_generation_length: int = 512
    fid_min_generation_length: int = 3
    fid_vocab_size: int = 32_128  # GreTa SentencePiece vocab (32K + 128 sentinels)


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameter configuration (Phases 3–5)."""

    # Optimization
    learning_rate: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduling
    warmup_steps: int = 1000
    scheduler: Literal["cosine", "linear", "constant_with_warmup"] = "cosine"
    num_cycles: float = 0.5

    # Steps
    max_steps: int = 100_000
    eval_steps: int = 500
    save_steps: int = 2000
    logging_steps: int = 50

    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch = 32

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: Literal["min", "max"] = "min"

    # Mixed precision
    use_amp: bool = True  # fp16 on GPU, no-op on CPU


@dataclass(frozen=True)
class StudyConfig:
    """Preregistration parameters for the source-criticism study (Q reconstruction
    + 2SH-vs-Farrer source identification).

    This dataclass IS the preregistration artifact. Its ``dataclasses.asdict``
    JSON is hashed (see ``synoptiq.data.study_design.config_hash``) and the hash is
    committed to ``docs/SOURCE_CRITICISM_STUDY.md`` §10 *before* the double tradition
    is ever scored. ``scripts/run_channel_test.py`` refuses to run unless the live
    config hash matches the frozen one — this is the mechanical guard against the
    post-hoc analysis drift that undid the earlier direction work.

    Nothing here selects a decision *threshold* by hand: E2/E1 claim thresholds
    are DERIVED from the null-control noise floor (gate G3) and the empirical
    power curve, both computed on known-answer data before unblinding.
    """

    # ── Cross-validation over the triple tradition ────────────────────────────
    # Folds are over *full triples* (pericopes with Mt, Mk and Lk all present).
    n_folds: int = 5
    fold_seed: int = 20260707  # date the direction cleanup closed — never re-rolled

    # ── Latent-source Monte-Carlo (E1 bottleneck branch) ──────────────────────
    # p(Lk | Mt) under 2SH marginalises over reconstructed sources G_Mt(Mt).
    # log-mean-exp over K importance samples; sensitivity swept over the tuple.
    latent_samples: tuple[int, ...] = (1, 5, 10)

    # ── Operator-transfer sensitivity (threat T6, Hägerland 2019) ─────────────
    # Interpolate R_Lk (Luke-redacts-Mark operator) with a free-composition
    # Lukan LM; verdict is only claimed if stable across this grid.
    fidelity_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    # ── Statistics ────────────────────────────────────────────────────────────
    n_bootstrap: int = 10_000
    ci_level: float = 0.95
    bootstrap_seed: int = 42
    # Power / kill-criterion K2: E1 (double tradition) is only run if the
    # empirical detection rate at the DT sample size clears this bar.
    power_target: float = 0.80
    power_n_sims: int = 2000

    # ── Kill criteria budget (K1) ─────────────────────────────────────────────
    max_gate_iterations: int = 2  # architecture re-tries before publishing a null

    # ── Text editions for the assimilation ablation (threat T8) ───────────────
    editions: tuple[str, ...] = ("sblgnt", "wh", "robinson_pierpont")

    # ── Output locations ──────────────────────────────────────────────────────
    output_dir: Path = Path("outputs/study")
    folds_path: Path = Path("outputs/study/folds.json")
    gates_dir: Path = Path("outputs/study/gates")
    unblind_sentinel: Path = Path("outputs/study/DT_UNBLINDED")


@dataclass(frozen=True)
class DAPTConfig:
    """Domain-adaptive pre-training configuration (Phase 2).

    DAPT uses LoRA (not full fine-tune) to prevent catastrophic forgetting
    on the ~1M-token Koine corpus. The backbone (GreTa) is frozen; only
    LoRA adapters on attention layers are updated.

    Training mixes 70% Koine + 30% classical Greek (replay buffer)
    to anchor the model's general Greek linguistic representations.
    """

    # Corpus components to include in DAPT
    corpus_components: tuple[str, ...] = (
        "sblgnt",  # ~138K tokens — primary NT text
        "lxx",  # ~600K tokens — LXX Septuagint (Koine control)
        "josephus",  # ~300K tokens — Josephus (Koine narrative)
        "apostolic",  # ~35K tokens  — Apostolic Fathers (post-NT Koine)
    )

    # Classical Greek replay buffer (prevents catastrophic forgetting)
    replay_components: tuple[str, ...] = ("first1k",)
    replay_fraction: float = 0.30  # 30% of each batch from replay

    # T5-style span corruption (better than BERT-style for enc-dec architectures)
    mlm_probability: float = 0.15
    mean_noise_span_length: float = 3.0

    # LoRA for DAPT
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "k", "v", "o", "wi", "wo")  # incl. MLP

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 2  # Effective batch = 32
    learning_rate: float = 1e-4  # Higher than fine-tuning
    max_steps: int = 80_000
    warmup_steps: int = 2000
    eval_steps: int = 1000
    save_steps: int = 5000

    # Go/no-go benchmark: MLM perplexity ratio ≤ this threshold
    # (KoineFormer perplexity / from-scratch baseline perplexity)
    target_mlm_perplexity_ratio: float = 1.05
