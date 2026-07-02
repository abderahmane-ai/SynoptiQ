"""Modal deployment for KoineFormer DAPT training.

Supports detached cloud training — runs on Modal's A10G GPU, survives
laptop sleep, checkpoints saved to persistent volume.

Usage:
    # Step 1: Upload data to Modal volume (once):
    modal run modal/app_dapt.py::upload_data

    # Step 2: Start training (detached — runs in cloud even if laptop sleeps):
    modal run modal/app_dapt.py::start_training

    # Step 3: Monitor progress (from any machine, any time):
    modal app logs synoptiq-dapt

    # Step 4: Download checkpoints when done:
    modal volume get synoptiq-outputs outputs/dapt/ models/koineformer/dapt/

    # Or do everything in one shot:
    modal run modal/app_dapt.py::upload_data && modal run modal/app_dapt.py::start_training
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import modal  # type: ignore[import-untyped]
except ImportError:
    modal = None  # type: ignore[assignment]

# ── Constants ────────────────────────────────────────────────────────────────

DATA_VOLUME = "synoptiq-data"
OUTPUT_VOLUME = "synoptiq-outputs"
IMAGE_PYTHON = "3.12"
GPU_TYPE = "A10G"
TIMEOUT_SECONDS = 86_400  # 24 hours

_REQUIREMENTS = [
    "torch>=2.6.0",
    "transformers>=4.51.0",
    "peft>=0.14.0",
    "safetensors>=0.4.0",
    "sentencepiece>=0.2.0",
]

# ── Image ───────────────────────────────────────────────────────────────────


def _build_image() -> Any:  # noqa: F821
    """Build the Modal container image with SynoptiQ installed."""
    if modal is None:
        msg = "Modal not installed — run with `modal run`"
        raise RuntimeError(msg)

    image = modal.Image.debian_slim(python_version=IMAGE_PYTHON)
    for req in _REQUIREMENTS:
        image = image.pip_install(req)
    # Also install core SynoptiQ deps that aren't in the GPU list.
    image = image.pip_install("pandas", "pyarrow", "biopython", "pyyaml", "tqdm", "numpy")
    # Copy the project source into the image and install it.
    image = image.add_local_dir("synoptiq", "/app/synoptiq", copy=True)
    image = image.add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
    image = image.add_local_file("README.md", "/app/README.md", copy=True)
    image = image.run_commands("pip install /app/")
    return image


# ── App ─────────────────────────────────────────────────────────────────────

app = modal.App("synoptiq-dapt") if modal is not None else None


def _get_volumes() -> tuple[Any, Any]:  # noqa: F821
    """Get or create Modal volumes."""
    data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
    return data_vol, output_vol


# ── Step 1: Upload data (runs locally) ──────────────────────────────────────


@app.local_entrypoint()  # type: ignore[misc]
def upload_data() -> None:
    """Upload local data/raw/ to the Modal synoptiq-data volume.

    Runs on YOUR machine — copies files from the local filesystem into
    the cloud volume.  Run once before training.  Skips files already
    present.
    """
    local_raw = Path("data/raw")
    if not local_raw.exists():
        print(f"ERROR: {local_raw} not found. Run prepare_data.py first.")
        sys.exit(1)

    # Count local files
    local_files = sum(1 for _ in local_raw.rglob("*") if _.is_file())
    print(f"Uploading {local_files} files from {local_raw} ...")

    # Upload raw corpora.
    import subprocess
    result = subprocess.run(
        ["modal", "volume", "put", "--force", DATA_VOLUME, str(local_raw), "/raw"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Raw upload failed: {result.stderr}")
        sys.exit(1)
    print(f"Raw data uploaded.")

    # Also upload processed Parquet files (needed for eval on Modal).
    local_proc = Path("data/processed")
    if local_proc.exists():
        result = subprocess.run(
            ["modal", "volume", "put", DATA_VOLUME, str(local_proc), "/processed"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Processed upload failed: {result.stderr}")
        else:
            print(f"Processed data uploaded.")

    print(f"Upload complete. Volume: {DATA_VOLUME}")


# ── Step 2: Train (detached) ───────────────────────────────────────────────


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
    },
    timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def start_training() -> None:
    """Run the full DAPT training loop on Modal GPU.

    Logs progress to stdout (streamable via `modal app logs synoptiq-dapt`).
    Checkpoints saved to the synoptiq-outputs volume.
    Training survives laptop sleep — runs entirely in the cloud.
    """
    from synoptiq.models.koineformer import KoineFormer
    from synoptiq.training.dapt import DAPTConfig, DAPTTrainer
    from transformers import AutoTokenizer

    data_dir = Path("/data/raw")
    output_dir = Path("/outputs/dapt")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    print(f"Loading KoineFormer on {GPU_TYPE}...")
    model = KoineFormer.from_pretrained(device=device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Check for existing checkpoint
    ckpt_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
    )
    if ckpt_dirs:
        latest = ckpt_dirs[-1]
        step = latest.name.split("-")[1]
        print(f"Found checkpoint at step {step} — will auto-resume")

    config = DAPTConfig(
        batch_size=8,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=20_000,
        val_steps=500,
        save_steps=2_000,
        grad_accum_steps=1,
        max_length=512,
        use_amp=True,
        output_dir=output_dir,
    )

    # Get the output volume for crash-safe commits
    output_vol = modal.Volume.from_name(OUTPUT_VOLUME) if modal is not None else None

    print(f"Starting DAPT: {config.max_steps} steps, batch={config.batch_size}, "
          f"seq_len={config.max_length}, AMP={config.use_amp}")
    print(f"Data: /data/raw  |  Checkpoints: /outputs/dapt")
    print(f"Auto-resume: yes  |  Volume commits: yes")
    print(f"Monitor: modal app logs synoptiq-dapt")
    print(f"{'='*50}")

    trainer = DAPTTrainer(model, data_dir, tokenizer, config, device=device)
    history = trainer.run(
        resume=True,
        commit_volume=True,
        volume=output_vol,
    )

    final_loss = history["loss"][-1]
    best_loss = min(history["loss"])
    print(f"\n{'='*50}")
    print(f"DAPT COMPLETE")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Checkpoints: /outputs/dapt/")
    print(f"  Download: modal volume get {OUTPUT_VOLUME} outputs/dapt/ models/")
    print(f"{'='*50}")


# ── Step 3: Ablation (LoRA vs full fine-tune) ──────────────────────────────


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
    },
    timeout=3600,
) if modal is not None else None
def run_ablation(n_steps: int = 2_000) -> None:
    """Run LoRA vs full fine-tune ablation on Modal GPU.

    Trains both variants for *n_steps* and saves loss curves to the output
    volume.  Full FT typically collapses after 1-2K steps on 1M-token data.
    """
    import json

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    from synoptiq.models.koineformer import KoineFormer
    from synoptiq.training.dapt import DAPTIterableDataset

    data_dir = Path("/data/raw")
    output_dir = Path("/outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _train(model_obj, label, steps):
        ds = DAPTIterableDataset(data_dir, tokenizer, max_length=512)
        it = iter(ds)
        opt = AdamW(model_obj.parameters(), lr=1e-4)
        losses = []
        model_obj.train()
        for s in range(1, steps + 1):
            sample = next(it)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            mask = torch.ones_like(input_ids)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model_obj(input_ids=input_ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())
            if s % 200 == 0:
                print(f"  [{label}] step {s}/{steps}: loss={out.loss.item():.4f}")
        return losses

    # Variant 1: LoRA
    print("=== LoRA DAPT ===")
    lora_model = KoineFormer.from_pretrained(device=device)
    lora_model.enable_dapt()
    peft_m = lora_model.model
    peft_m.resize_token_embeddings(len(tokenizer))
    lora_losses = _train(peft_m, "LoRA", n_steps)

    # Variant 2: Full fine-tune (raw GreTa)
    print("=== Full Fine-Tune ===")
    raw = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa", torch_dtype=torch.float32).to(device)
    if hasattr(raw.config, "tie_word_embeddings"):
        raw.config.tie_word_embeddings = False
    raw.resize_token_embeddings(len(tokenizer))
    ft_losses = _train(raw, "Full FT", n_steps)

    # Report
    lora_end = lora_losses[-1]
    ft_end = ft_losses[-1]
    print(f"\n{'='*50}")
    print(f"  LoRA final loss: {lora_end:.4f}")
    print(f"  Full FT final loss: {ft_end:.4f}")
    print(f"  {'✓ LoRA wins' if lora_end < ft_end else 'Full FT lower loss'}")

    results = {
        "steps": n_steps,
        "lora_losses": lora_losses,
        "lora_final": lora_end,
        "fullft_losses": ft_losses,
        "fullft_final": ft_end,
        "verdict": "lora_wins" if lora_end < ft_end else "fullft_lower",
    }
    (output_dir / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results: /outputs/ablation/ablation_results.json")


# ── Step 4: Full FT DAPT + downstream eval ────────────────────────────────


@app.function(  # type: ignore[misc]
    gpu=GPU_TYPE,
    image=_build_image(),
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True),
    },
    timeout=TIMEOUT_SECONDS,
) if modal is not None else None
def train_and_eval_full_ft() -> None:
    """Train full fine-tune DAPT and evaluate downstream POS accuracy.

    Trains GreTa without LoRA on Koine DAPT for 20K steps, then evaluates
    POS tagging accuracy using the same linear-probe protocol used for the
    zero-shot and LoRA models.  Saves results to /outputs/full_ft/.
    """
    import json
    from collections import defaultdict

    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    from synoptiq.training.dapt import DAPTIterableDataset

    device = "cuda"
    bsz, n_steps = 8, 20_000
    output_dir = Path("/outputs/full_ft")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + tokenizer ─────────────────────────────────────────
    print("Loading raw GreTa (no LoRA) for full fine-tune...")
    model = AutoModelForSeq2SeqLM.from_pretrained("bowphs/GreTa", torch_dtype=torch.float32).to(device)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreTa")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    trainable = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}")

    # ── Train ──────────────────────────────────────────────────────────
    ds = DAPTIterableDataset(Path("/data/raw"), tokenizer, max_length=512)
    it = iter(ds)
    opt = AdamW(model.parameters(), lr=1e-4)
    sched = CosineAnnealingLR(opt, T_max=n_steps)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    losses: list[float] = []
    best_loss = float("inf")

    print(f"Training full FT DAPT: {n_steps} steps, batch={bsz}")
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        for _ in range(1):  # grad accum = 1
            sample = next(it)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            mask = torch.ones_like(input_ids)
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            scaler.scale(outputs.loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        losses.append(outputs.loss.item())
        if step % 500 == 0:
            avg = sum(losses[-100:]) / min(100, len(losses))
            print(f"  step {step}/{n_steps}: loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if step % 2_000 == 0:
            ckpt = output_dir / f"step-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))

    # Final save
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"Full FT DAPT complete. Final loss: {losses[-1]:.4f}")

    # ── Evaluate downstream POS accuracy ──────────────────────────────
    print("Evaluating downstream POS tagging...")

    # We need the SynoptiQ Corpus.  Load it from /data/processed/ or
    # reconstruct from the raw data that was uploaded.  Since we uploaded
    # data/raw/ only, we load the tokenizer-based probe directly from
    # the encoder hidden states.

    # Build a simple POS tag set from the encoder's behavior.
    # We evaluate by extracting hidden states on test verses and training
    # a linear probe on train verses — same protocol as local eval.
    from synoptiq.training.dapt import _extract_text_from_dir

    # We need the processed corpus.  Reconstruct via the Corpus API
    # using the Parquet files if they exist on the volume, otherwise
    # fall back to a head-only probe.
    import sys
    sys.path.insert(0, "/app")
    from synoptiq.data.corpus import Corpus

    processed = Path("/data/processed")
    if processed.exists():
        corpus = Corpus.from_parquet(
            processed / "tokens.parquet",
            processed / "pericopes.parquet",
            alignments_path=processed / "alignments.json",
            splits_path=processed / "splits.json",
        )
    else:
        print("No processed corpus on volume — cannot evaluate POS. "
              "Run upload_data with processed/ files too.")
        return

    # POS tag vocabulary
    pos_to_idx: dict[str, int] = {}
    for token in corpus.get_tokens(split="train"):
        p = token.get("pos", "")
        if p and p not in pos_to_idx:
            pos_to_idx[p] = len(pos_to_idx)
    n_classes = len(pos_to_idx)
    print(f"POS classes: {n_classes}")

    # Extract per-word hidden states
    encoder = model.encoder
    encoder.eval()

    def _extract(data_split, max_verses=500):
        all_h, all_l = [], []
        n_v = 0
        for book in ("Matthew", "Mark", "Luke"):
            tokens = corpus.get_tokens(book=book, split=data_split)
            verses = defaultdict(list)
            for t in tokens:
                if n_v >= max_verses:
                    break
                verses[(int(t["chapter"]), int(t["verse"]))].append(t)
            for vref, vt in verses.items():
                if n_v >= max_verses or len(vt) < 3:
                    continue
                n_v += 1
                text = " ".join(str(t["text"]) for t in vt)
                enc = tokenizer(text, max_length=128, truncation=True,
                                padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].to(device)
                am = enc["attention_mask"].to(device)
                with torch.no_grad():
                    h = encoder(input_ids=ids, attention_mask=am).last_hidden_state[0]
                stoks = tokenizer.convert_ids_to_tokens(ids[0])
                boundaries = [si for si, st in enumerate(stoks)
                              if st.startswith("▁") or si == 0 or st == "[PAD]"]
                for wi, ws in enumerate(boundaries):
                    if wi >= len(vt):
                        break
                    we = boundaries[wi + 1] if wi + 1 < len(boundaries) else len(stoks)
                    we = min(we, am.sum().item())
                    if ws >= we:
                        continue
                    wh = h[ws:we].mean(dim=0)
                    p = vt[wi].get("pos", "")
                    if p in pos_to_idx:
                        all_h.append(wh.cpu())
                        all_l.append(pos_to_idx[p])
        if not all_h:
            return torch.zeros(0, 768), torch.zeros(0, dtype=torch.long)
        return torch.stack(all_h), torch.tensor(all_l, dtype=torch.long)

    X_train, y_train = _extract("train")
    X_test, y_test = _extract("test")
    print(f"Train: {len(X_train)} tokens, Test: {len(X_test)} tokens")

    # Linear probe
    probe = torch.nn.Linear(768, n_classes).to(device)
    opt_p = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    Xt, yt = X_train.to(device), y_train.to(device)
    probe.train()
    for ep in range(10):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt_p.zero_grad()
            crit(probe(Xt[idx]), yt[idx]).backward()
            opt_p.step()
    probe.eval()
    with torch.no_grad():
        preds = probe(X_test.to(device)).argmax(dim=1)
        acc = (preds == y_test.to(device)).sum().item() / len(y_test)
    print(f"\n  Full FT DAPT POS accuracy: {acc:.2%} ({acc*100:.1f}%)")

    # Save results
    results = {
        "model": "GreTa full fine-tune DAPT (20K steps)",
        "pos_accuracy": acc,
        "n_test_tokens": len(y_test),
        "final_training_loss": losses[-1],
        "best_training_loss": min(losses),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n{'='*50}")
    print(f"Full FT POS accuracy: {acc:.2%}")
    print(f"Results saved: /outputs/full_ft/results.json")
    print(f"{'='*50}")


# ── Local entrypoint ───────────────────────────────────────────────────────


@app.local_entrypoint()  # type: ignore[misc]
def main() -> None:
    """Local entry point — prints usage instructions."""
    print("""
    SynoptiQ DAPT — Modal GPU Training
    ==================================

    Step 1 — Upload data (once, idempotent):
      modal run modal/app_dapt.py::upload_data

    Step 2 — Start training (auto-resumes from checkpoint):
      modal run modal/app_dapt.py::start_training

    Step 3 — Monitor progress (live logs):
      modal app logs synoptiq-dapt

    Step 4 — Download trained model:
      modal volume get synoptiq-outputs outputs/dapt/ models/koineformer/dapt/
    """)


if __name__ == "__main__":
    main()
