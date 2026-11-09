"""Expressifying SymphonyNet piano parts: segmentation, dedupe, sliding window, mapping (tiny model, no data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orchestration_v2.constants import AMT_DENSE_MAX_ONSET
from orchestration_v2.sampling.expressify_symphonynet import (
    SegmentJob,
    assemble,
    load_mapping,
    order_and_dedupe,
    piece_jobs,
    prepare_notes,
    segment_ids,
)


def _notes(n=300, seed=0, gap_at=None):
    rng = np.random.default_rng(seed)
    on = np.sort(rng.integers(0, 4 * n, n)) * 0.125
    if gap_at is not None:
        on[gap_at:] += 30.0
    return pd.DataFrame(dict(instrument=rng.integers(0, 2, n), note_index_in_instrument=np.arange(n),
                             onset=on, offset=on + 0.25, pitch=rng.integers(40, 90, n),
                             velocity=rng.integers(30, 100, n)))


def test_segment_ids_gap_is_silence_not_onset_distance():
    on = np.array([0.0, 1.0, 9.0, 20.0, 27.9])
    off = np.array([0.5, 1.0, 12.0, 20.5, 28.0])
    # 1.0 -> 9.0: silent 8.0 s exactly -> new segment; 9.0 -> 20.0: note sounds until 12.0, silence 8.0 -> new;
    # 20.0 -> 27.9: silence 7.4 -> same
    assert segment_ids(on, off, 8.0).tolist() == [0, 0, 1, 2, 2]
    # a long held note bridges a long onset gap
    assert segment_ids(np.array([0.0, 15.0]), np.array([10.0, 16.0]), 8.0).tolist() == [0, 0]


def test_dedupe_keeps_longest_and_points_duplicates_to_it():
    df = pd.DataFrame(dict(instrument=[0, 1, 0], note_index_in_instrument=[0, 0, 1], onset=[1.0, 1.0, 2.0],
                           offset=[1.5, 3.0, 2.5], pitch=[60, 60, 62], velocity=[70, 80, 90]))
    out = order_and_dedupe(df)
    assert out.instrument.tolist() == [1, 0, 0] and out.dup_of.tolist() == [-1, 0, -1]


def test_next_window_sliding_and_guard():
    n = 300
    j = SegmentJob(key="k", segment=0, sample=0, score_on_cs=np.arange(n) * 10, score_dur_cs=np.full(n, 20),
                   pitch=np.full(n, 60), seed=0)
    spans = []
    while not j.finished:
        s, e = j.next_window()
        spans.append((s, e, j.done_notes))
        inp, pre, plen, ref = j.window_tokens(s, e)
        out = pre.copy()
        out[plen:] = inp[plen:]  # "perform" exactly as the score
        j.apply(s, e, out, ref)
    assert spans == [(0, 128, 0), (64, 192, 128), (128, 256, 192), (172, 300, 256)]
    assert np.array_equal(j.perf_on_cs, j.score_on_cs)  # re-relativised prefixes round-trip exactly
    # a sparse passage whose 128-note score span exceeds the onset range -> smaller windows
    k = SegmentJob(key="k", segment=0, sample=0, score_on_cs=np.arange(n) * 200, score_dur_cs=np.full(n, 20),
                   pitch=np.full(n, 60), seed=0)
    s, e = k.next_window()
    assert (s, e) == (0, 32) and k.score_on_cs[e - 1] - k.score_on_cs[s] <= AMT_DENSE_MAX_ONSET


def _tiny_model():
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2Model

    from orchestration_v2.constants import EXPRESSIFIER_VOCAB_SIZE
    from orchestration_v2.models.expressifier import ExpressifierModel

    torch.manual_seed(0)
    enc = GPT2Model(GPT2Config(n_embd=64, n_layer=2, n_head=4, n_positions=512, vocab_size=EXPRESSIFIER_VOCAB_SIZE))
    return ExpressifierModel(enc, decoder_layers=2, dropout=0.0).eval()


def test_generate_prefix_positions_are_kept():
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    j = SegmentJob(key="k", segment=0, sample=0, score_on_cs=np.arange(16) * 25, score_dur_cs=np.full(16, 20),
                   pitch=np.arange(16) + 50, seed=0)
    inp = torch.from_numpy(j.window_tokens(0, 16)[0])[None]
    pre = model.generate(inp, torch.ones_like(inp))
    out = model.generate(inp, torch.ones_like(inp), prefix_ids=pre, prefix_len=torch.tensor([32]))
    assert torch.equal(out[0, :32], pre[0, :32])


def test_single_window_segment_matches_plain_generate_and_batching_does_not_matter():
    torch = pytest.importorskip("torch")
    from orchestration_v2.sampling.expressify_symphonynet import run_jobs

    model = _tiny_model()
    notes = prepare_notes(order_and_dedupe(_notes(40, gap_at=20)))
    assert notes.segment.nunique() == 2
    jobs = piece_jobs("p", notes, num_samples=1, seed=0)
    run_jobs(model, jobs, batch_size=2)
    # plain generate on segment 0 with the same per-job uniforms
    ref_job = piece_jobs("p", notes, num_samples=1, seed=0)[0]
    inp = torch.from_numpy(ref_job.window_tokens(0, ref_job.n)[0])[None]
    u = torch.rand(inp.shape[1], generator=ref_job.torch_generator(), dtype=torch.float64)[None]
    plain = model.generate(inp, torch.ones_like(inp), uniforms=u)[0].numpy().reshape(-1, 4)
    assert np.array_equal(jobs[0].perf_vel, plain[:, 3] - 11_640)
    # one job at a time gives the same result as two in a batch
    solo = piece_jobs("p", notes, num_samples=1, seed=0)
    run_jobs(model, solo, batch_size=1)
    for a, b in zip(jobs, solo):
        assert np.array_equal(a.perf_on_cs, b.perf_on_cs) and np.array_equal(a.perf_vel, b.perf_vel)


def test_long_segment_and_mapping_rows(tmp_path):
    pytest.importorskip("torch")
    from orchestration_v2.sampling.expressify_symphonynet import run_jobs

    model = _tiny_model()
    raw = _notes(300, seed=1)
    raw = pd.concat([raw, raw.iloc[[5]].assign(instrument=7, offset=raw.onset.iloc[5] + 0.1)], ignore_index=True)
    notes = prepare_notes(order_and_dedupe(raw))
    jobs = piece_jobs("p", notes, num_samples=2, seed=0)
    run_jobs(model, jobs, batch_size=4)
    assert all(j.finished and j.n_windows > 1 for j in jobs if j.n > 128)
    mapping, segs = assemble("p", notes, jobs)
    assert len(mapping) == 2 * len(raw)  # every original note, every sample
    m0 = mapping[mapping["sample"] == 0]
    assert np.allclose(m0.t_start + m0.orig_onset, m0.orig_onset_abs)
    for _, g in m0.groupby("segment"):
        assert g.orig_onset.min() == 0 and g.expr_onset.min() == 0
    dup = m0[m0.dup_of >= 0]
    assert (dup.instrument == 7).sum() == 1  # the injected duplicate (random notes may add more)
    src = m0.set_index("row").loc[dup.dup_of.to_numpy()]
    assert np.array_equal(dup.expr_onset.to_numpy(), src.expr_onset.to_numpy())
    assert np.array_equal(dup.expr_velocity.to_numpy(), src.expr_velocity.to_numpy())
    assert (m0.expr_pitch == m0.pitch).all() and (m0.window_first >= 0).all()
    path = tmp_path / "p.parquet"
    mapping.to_parquet(path, index=False)
    assert len(load_mapping(path)) <= len(raw)


def test_shard_files_partition(tmp_path):
    from orchestration_v2.sampling.expressify_symphonynet import shard_files

    files = []
    for i in range(10):
        f = tmp_path / f"p{i}.mid"
        f.write_bytes(b"x" * (i + 1) * 10)
        files.append(f)
    shards = [shard_files(files, 3, k) for k in range(3)]
    assert sorted(sum(shards, [])) == sorted(files) and len({f for s in shards for f in s}) == 10
    assert shards[0][0].name == "p9"  # largest first
