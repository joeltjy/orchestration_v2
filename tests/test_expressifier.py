"""Expressifier: chunking, tokenization, augmentation, dataset and model (tiny GPT-2, nothing downloaded)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from orchestration_v2.constants import (
    AMT_DENSE_DURATION_OFFSET,
    AMT_DENSE_NOTE_OFFSET,
    AMT_DENSE_VELOCITY_OFFSET,
    EXPRESSIFIER_CHUNK_LENGTHS,
    EXPRESSIFIER_CHUNK_PROBS,
    EXPRESSIFIER_SCORE_VELOCITY,
    EXPRESSIFIER_VOCAB_SIZE,
    LABEL_IGNORE_INDEX,
)
from orchestration_v2.preprocessing.expressifier_tokenize import (
    augment,
    augmentation_schedule,
    chunk_bounds,
    piece_rng,
    tokenize_chunk,
    tokenize_piece,
    write_manifest,
    write_shard,
)


def _piece(n=300, seed=0):
    rng = np.random.default_rng(seed)
    score_on = np.sort(rng.integers(0, 4 * n, n)) * 0.125  # quantised, non-decreasing
    perf_on = score_on * 1.1 + rng.normal(0, 0.03, n)  # expressive timing: not monotonic
    return dict(score_on=score_on, score_off=score_on + 0.25, perf_on=perf_on, perf_off=perf_on + 0.2,
                pitch=rng.integers(40, 90, n), velocity=rng.integers(30, 100, n),
                interpolated=rng.random(n) < 0.05)


def test_chunk_lengths_follow_distribution():
    rng = np.random.default_rng(0)
    bounds = chunk_bounds(2_000_000, rng)
    lengths = np.array([b - a for a, b in bounds[:-1]])
    for L, p in zip(EXPRESSIFIER_CHUNK_LENGTHS, EXPRESSIFIER_CHUNK_PROBS):
        assert abs((lengths == L).mean() - p) < 0.01
    assert bounds[0][0] == 0 and all(b == a2 for (_, b), (a2, _) in zip(bounds, bounds[1:]))


def test_tokenize_chunk_layout_and_relative_onsets():
    p = _piece(64)
    inp, out = tokenize_chunk(p["score_on"], p["score_off"], p["perf_on"], p["perf_off"], p["pitch"], p["velocity"])
    assert inp.shape == out.shape == (4 * 64,)
    qi, qo = inp.reshape(-1, 4), out.reshape(-1, 4)
    assert qi[0, 0] == 0 and (np.diff(qi[:, 0]) >= 0).all()  # score: relative, onset order
    assert qo[:, 0].min() == 0  # performance: relative to its earliest onset ...
    assert (np.diff(qo[:, 0]) < 0).any()  # ... and allowed to go backwards (score note order kept)
    assert (qi[:, 2] == qo[:, 2]).all() and (qi[:, 2] == AMT_DENSE_NOTE_OFFSET + p["pitch"]).all()
    assert (qi[:, 3] == AMT_DENSE_VELOCITY_OFFSET + EXPRESSIFIER_SCORE_VELOCITY).all()
    assert (qo[:, 3] == AMT_DENSE_VELOCITY_OFFSET + p["velocity"]).all()
    assert (qi[:, 1] >= AMT_DENSE_DURATION_OFFSET).all() and inp.max() < EXPRESSIFIER_VOCAB_SIZE


def test_tokenize_chunk_rejects_overlong_span():
    p = _piece(16)
    p["score_on"][-1] = 200.0
    assert tokenize_chunk(p["score_on"], p["score_off"], p["perf_on"], p["perf_off"], p["pitch"],
                          p["velocity"]) is None


def test_augmentation_schedule_and_transforms():
    assert augmentation_schedule(16) == ["none"] + ["pitch"] * 5 + ["tempo"] * 5 + ["velocity"] * 5
    p = _piece(50)
    q = augment(p, "pitch", piece_rng("x", 1, 0))
    shift = q["pitch"] - p["pitch"]
    assert len(set(shift)) == 1 and abs(int(shift[0])) <= 12
    t = augment(p, "tempo", piece_rng("x", 2, 0))
    f = t["perf_on"][5] / p["perf_on"][5]
    assert 0.5 <= f <= 2.0 and np.allclose(t["score_on"], p["score_on"] * f)
    v = augment(p, "velocity", piece_rng("x", 3, 0))
    assert np.array_equal(v["score_on"], p["score_on"]) and v["velocity"].min() >= 1


def test_shard_roundtrip_dataset_and_collate(tmp_path):
    torch = pytest.importorskip("torch")
    from orchestration_v2.dataloaders.expressifier_dataloader import (
        ExpressifierDataset,
        collate_expressifier_batch,
    )

    p = _piece(400)
    chunks = [(i, o, m, "piece0", k) for i, o, m, k in tokenize_piece("piece0", p, ["none", "pitch"], 0)]
    for split in ("train", "test"):
        write_shard(tmp_path / split / "s0", chunks)
    write_manifest(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["splits"]["train"][0]["n_chunks"] == len(chunks)
    ds = ExpressifierDataset(tmp_path, "train")
    assert len(ds) == len(chunks)
    item = ds[0]
    n = len(chunks[0][2])
    assert item["input_ids"].shape == (4 * n,)
    masked = ~torch.from_numpy(np.repeat(chunks[0][2], 4))
    assert (item["labels"][masked] == LABEL_IGNORE_INDEX).all()
    batch = collate_expressifier_batch([ds[0], ds[1]])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape


def _tiny_model():
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2Model

    from orchestration_v2.models.expressifier import ExpressifierModel

    torch.manual_seed(0)
    enc = GPT2Model(GPT2Config(n_embd=64, n_layer=2, n_head=4, n_positions=512, vocab_size=EXPRESSIFIER_VOCAB_SIZE))
    return ExpressifierModel(enc, decoder_layers=2, dropout=0.0).eval()


def _batch(n_notes=8):
    torch = pytest.importorskip("torch")
    p = _piece(n_notes)
    inp, out = tokenize_chunk(p["score_on"], p["score_off"], p["perf_on"], p["perf_off"], p["pitch"], p["velocity"])
    i, o = torch.tensor(inp, dtype=torch.long)[None], torch.tensor(out, dtype=torch.long)[None]
    return i, torch.ones_like(i), o


def test_model_loss_and_field_masking():
    model = _tiny_model()
    i, m, o = _batch()
    res = model(input_ids=i, attention_mask=m, output_ids=o, labels=o)
    assert res.loss is not None and bool(res.loss.isfinite())
    top = res.logits.argmax(-1)[0].reshape(-1, 4)
    assert (top[:, 1] >= AMT_DENSE_DURATION_OFFSET).all() and (top[:, 1] < AMT_DENSE_NOTE_OFFSET).all()
    assert (top[:, 3] >= AMT_DENSE_VELOCITY_OFFSET).all()


def test_decoder_is_causal_encoder_is_bidirectional():
    model = _tiny_model()
    i, m, o = _batch()
    base = model(input_ids=i, attention_mask=m, output_ids=o).logits
    o2 = o.clone()
    o2[0, 20] = AMT_DENSE_DURATION_OFFSET + 7  # change a later performance token
    changed = model(input_ids=i, attention_mask=m, output_ids=o2).logits
    assert (base[0, :21] - changed[0, :21]).abs().max() < 1e-5  # positions <= 20 cannot see token 20 ...
    assert (base[0, 21:] - changed[0, 21:]).abs().max() > 0  # ... later ones do
    i2 = i.clone()
    i2[0, -1] = AMT_DENSE_VELOCITY_OFFSET + 1  # change the LAST score token
    h1, h2 = model.encode(i, m), model.encode(i2, m)
    assert (h1[0, 0] - h2[0, 0]).abs().max() > 0  # the first score token sees it: bidirectional


def test_generate_shapes_and_pitch_copy():
    model = _tiny_model()
    i, m, _ = _batch(6)
    out = model.generate(i, m, temperature=1.0, top_p=0.9)
    assert out.shape == i.shape
    assert (out[0].reshape(-1, 4)[:, 2] == i[0].reshape(-1, 4)[:, 2]).all()


def test_pairs_frame_columns():
    from orchestration_v2.preprocessing.expressifier_alignment import AlignedNotes
    from orchestration_v2.preprocessing.expressifier_pairs import pair_notes_frame

    z = np.zeros(3)
    a = AlignedNotes(pitch=np.array([60, 62, 64]), score_onset_beats=z, score_offset_beats=z + 1,
                     score_onset_s=z, score_offset_s=z + 0.5, score_velocity=np.full(3, 80), perf_onset_s=z,
                     perf_offset_s=z + 0.4, velocity=np.full(3, 64), interpolated=np.zeros(3, bool),
                     ref_index=np.arange(3), info={})
    df = pair_notes_frame("p", a)
    assert isinstance(df, pd.DataFrame) and len(df) == 3 and {"note_idx", "perf_onset_s"} <= set(df.columns)
