"""Forward/score/fusion tests for Redactor and FusionInDecoder on a tiny CPU T5."""

from __future__ import annotations

import torch
from transformers import T5Config, T5ForConditionalGeneration

from synoptiq.models.fid import FusionInDecoder
from synoptiq.models.redactor import Redactor


def _tiny_t5() -> T5ForConditionalGeneration:
    cfg = T5Config(
        vocab_size=64, d_model=16, d_kv=8, d_ff=32,
        num_layers=2, num_decoder_layers=2, num_heads=2,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    torch.manual_seed(0)
    return T5ForConditionalGeneration(cfg)


class _FakeTok:
    pad_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        ids = [(len(w) % 60) + 2 for w in text.split()] or [2]
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


# ── Redactor ──────────────────────────────────────────────────────────────────


def test_redactor_forward_produces_loss_and_logits() -> None:
    r = Redactor(_tiny_t5(), _FakeTok(), name="R_Lk")
    src = torch.randint(2, 64, (1, 7))
    tgt = torch.randint(2, 64, (1, 5))
    out = r.forward(src, torch.ones_like(src), tgt)
    assert out.loss.item() > 0
    assert out.logits.shape == (1, 5, 64)


def test_redactor_score_is_positive_scalar_per_sequence() -> None:
    r = Redactor(_tiny_t5(), _FakeTok())
    src = torch.randint(2, 64, (2, 6))
    tgt = torch.randint(2, 64, (2, 4))
    nll = r.score(src, torch.ones_like(src), tgt)
    assert nll.shape == (2,)
    assert torch.all(nll > 0)


def test_redactor_encode_pair_shapes_and_keys() -> None:
    r = Redactor(_tiny_t5(), _FakeTok())
    batch = r.encode_pair("α β γ", "δ ε")
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[1] == 3
    assert batch["labels"].shape[1] == 2


def test_redactor_generate_returns_tokens() -> None:
    r = Redactor(_tiny_t5(), _FakeTok())
    src = torch.randint(2, 64, (1, 6))
    gen = r.generate(src, torch.ones_like(src), max_new_tokens=4, num_beams=1)
    assert gen.ndim == 2 and gen.shape[0] == 1


# ── FusionInDecoder ─────────────────────────────────────────────────────────────


def test_fid_fuse_concatenates_encoder_states() -> None:
    fid = FusionInDecoder(_tiny_t5(), _FakeTok())
    w1 = torch.randint(2, 64, (1, 7))
    w2 = torch.randint(2, 64, (1, 5))
    enc, mask = fid.fuse_witnesses([w1, w2], [torch.ones_like(w1), torch.ones_like(w2)])
    # sequence axis is the sum of the two witness lengths; hidden dim = d_model
    assert enc.last_hidden_state.shape == (1, 12, 16)
    assert mask.shape == (1, 12)


def test_fid_forward_and_score_over_two_witnesses() -> None:
    fid = FusionInDecoder(_tiny_t5(), _FakeTok())
    w1 = torch.randint(2, 64, (1, 6))
    w2 = torch.randint(2, 64, (1, 8))
    tgt = torch.randint(2, 64, (1, 5))
    masks = [torch.ones_like(w1), torch.ones_like(w2)]
    out = fid.forward([w1, w2], masks, tgt)
    assert out.loss.item() > 0
    assert out.logits.shape == (1, 5, 64)
    nll = fid.score([w1, w2], masks, tgt)
    assert nll.shape == (1,) and nll.item() > 0


def test_fid_single_witness_also_works() -> None:
    # Source-dropout means the same weights must handle one witness too.
    fid = FusionInDecoder(_tiny_t5(), _FakeTok())
    w = torch.randint(2, 64, (1, 6))
    tgt = torch.randint(2, 64, (1, 4))
    out = fid.forward([w], [torch.ones_like(w)], tgt)
    assert out.logits.shape == (1, 4, 64)


def test_fid_rejects_mismatched_witnesses() -> None:
    fid = FusionInDecoder(_tiny_t5(), _FakeTok())
    w = torch.randint(2, 64, (1, 6))
    try:
        fid.fuse_witnesses([w, w], [torch.ones_like(w)])
    except ValueError as e:
        assert "equal length" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_fid_generate_from_fused_witnesses() -> None:
    # Exercises the encoder_outputs plumbing through model.generate.
    fid = FusionInDecoder(_tiny_t5(), _FakeTok())
    w1 = torch.randint(2, 64, (1, 6))
    w2 = torch.randint(2, 64, (1, 5))
    gen = fid.generate([w1, w2], [torch.ones_like(w1), torch.ones_like(w2)],
                       max_new_tokens=4, num_beams=1)
    assert gen.ndim == 2 and gen.shape[0] == 1


def test_fid_encode_example_builds_witness_lists() -> None:
    fid = FusionInDecoder(_tiny_t5(), _FakeTok())
    batch = fid.encode_example(["α β", "γ δ ε"], "ζ")
    assert len(batch["witness_input_ids"]) == 2
    assert len(batch["witness_masks"]) == 2
    assert batch["labels"].shape[1] == 1
