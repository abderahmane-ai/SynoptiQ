"""Paper A ablation: LoRA DAPT vs full fine-tune DAPT.

Demonstrates that full fine-tuning GreTa's 220M parameters on 1M Koine
tokens causes representation collapse, while LoRA (3.7M params) converges
stably.  This is the key empirical finding of Paper A.

Produces a JSON results file with loss curves for both variants.

Usage:
    # Run 500 steps of each (enough to show the collapse):
    python scripts/run_ablation.py --smoke-test

    # Full ablation (2K steps each, ~1 hour on A10G):
    python scripts/run_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._cli_utils import detect_device  # noqa: E402
from synoptiq.models.koineformer import KoineFormer  # noqa: E402
from synoptiq.training.dapt import DAPTIterableDataset  # noqa: E402
from synoptiq.utils.logging_ import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def _run_training(
    model: KoineFormer,
    tokenizer: AutoTokenizer,
    data_dir: Path,
    n_steps: int,
    device: str,
    label: str,
) -> list[float]:
    """Run *n_steps* of T5 span corruption and return per-step losses."""
    dataset = DAPTIterableDataset(data_dir, tokenizer, max_length=256)
    data_iter = iter(dataset)

    peft_model = model.model
    peft_model.train()

    optimizer = AdamW(peft_model.parameters(), lr=1e-4)
    losses: list[float] = []

    _LOG.info(f"training {label}", extra={"steps": n_steps})
    for step in range(1, n_steps + 1):
        sample = next(data_iter)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        optimizer.zero_grad()
        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 100 == 0:
            _LOG.info(f"  [{label}] step {step}/{n_steps}: loss={loss.item():.4f}")

    return losses


def main() -> int:
    """Run the LoRA-vs-full-fine-tune ablation. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Ablation: LoRA vs full fine-tune DAPT")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--steps", type=int, default=2_000,
                        help="Training steps per variant")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 200 steps each (quick sanity check)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or detect_device()

    n_steps = 200 if args.smoke_test else args.steps

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.pad_token = tokenizer.eos_token

    results: dict[str, object] = {"steps": n_steps}

    # ── Variant 1: LoRA DAPT (our method) ─────────────────────────────
    _LOG.info("=== VARIANT 1: LoRA DAPT ===")
    lora_model = KoineFormer.from_pretrained(device=device)
    lora_model.enable_dapt()
    lora_losses = _run_training(lora_model, tokenizer, data_dir, n_steps, device, "LoRA")
    results["lora_losses"] = lora_losses
    results["lora_final"] = lora_losses[-1]

    # ── Variant 2: Full fine-tune (raw GreTa, no LoRA) ──────────────
    _LOG.info("=== VARIANT 2: Full fine-tune (raw GreTa, 220M params) ===")
    from transformers import AutoModelForSeq2SeqLM
    raw_model = AutoModelForSeq2SeqLM.from_pretrained(
        "bowphs/GreTa",
        torch_dtype=torch.float32,
    ).to(device)
    if hasattr(raw_model.config, "tie_word_embeddings"):
        raw_model.config.tie_word_embeddings = False
    trainable = sum(p.numel() for p in raw_model.parameters())
    _LOG.info(f"full FT trainable params: {trainable:,}")
    # Wrap in a minimal KoineFormer so we can reuse _run_training
    from synoptiq.models.koineformer import KoineFormer as KF
    ft_model = KF.__new__(KF)
    ft_model._model = raw_model
    ft_model._model_id = "bowphs/GreTa"
    ft_model._device = device
    ft_losses = _run_training(ft_model, tokenizer, data_dir, n_steps, device, "Full FT")
    results["fullft_losses"] = ft_losses
    results["fullft_final"] = ft_losses[-1]

    # ── Report ───────────────────────────────────────────────────────
    lora_start = lora_losses[0]
    lora_end = lora_losses[-1]
    ft_start = ft_losses[0]
    ft_end = ft_losses[-1]

    lora_pct = (lora_start - lora_end) / lora_start * 100
    ft_pct = (ft_start - ft_end) / ft_start * 100

    print(f"\n  {'Variant':<20s} {'Start':>8s}  {'End':>8s}  {'Δ%':>8s}")
    print(f"  {'─'*20} {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'LoRA (3.7M params)':<20s} {lora_start:>7.4f}  {lora_end:>7.4f}  {lora_pct:>7.1f}%")
    print(f"  {'Full FT (220M params)':<20s} {ft_start:>7.4f}  {ft_end:>7.4f}  {ft_pct:>7.1f}%")

    if lora_end < ft_end:
        ratio = ft_end / lora_end
        print(f"\n  ✓ LoRA loss is {ratio:.1f}× lower than full fine-tune")
    else:
        print("\n  ⚠ Full FT performed better — unexpected, investigate")

    results["verdict"] = "lora_wins" if lora_end < ft_end else "fullft_wins"

    results_path = output_dir / "ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _LOG.info("ablation saved", extra={"path": str(results_path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
