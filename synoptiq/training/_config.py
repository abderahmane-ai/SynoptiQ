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

    # Direction scorer head: 10 fixed features + swap-equivariant classifier
    direction_num_classes: int = 3  # A→B, B→A, independent
    asymmetry_num_features: int = 10
    direction_signed_features: int = 6
    direction_independence_features: int = 10

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


@dataclass(frozen=True)
class BayesianConfig:
    """Bayesian model comparison configuration (Phase 6)."""

    # MCMC (PyMC NUTS sampler)
    num_chains: int = 4
    num_warmup: int = 500
    num_samples: int = 2000
    target_accept: float = 0.90
    random_seed: int = 42

    # Bridge sampling via rpy2 → R bridgesampling
    bridge_n_iterations: int = 50_000

    # Prior sensitivity grid: Gamma(alpha, beta) priors on Beta distribution params
    prior_alpha_range: tuple[float, float, int] = (1.0, 5.0, 5)  # start, stop, n_steps
    prior_beta_range: tuple[float, float, int] = (1.0, 5.0, 5)

    # Convergence diagnostics thresholds
    r_hat_threshold: float = 1.01  # R-hat < 1.01 required for all parameters
    min_ess: int = 400  # Minimum effective sample size

    # MC Dropout for uncertainty estimation (Phase 6 input generation)
    mc_dropout_passes: int = 20  # Number of stochastic forward passes per pericope
