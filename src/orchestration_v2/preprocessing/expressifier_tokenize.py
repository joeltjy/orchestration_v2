"""Chunk + tokenize (score, performance) pairs for the expressifier.

Input pairs (one row per score note, in score order = sorted by score onset, then pitch):
  * stage 1: ``expressifier_pairs`` output (PianoCoRe A*), no augmentation, PianoCoRe's train/test split.
    Performances whose ASAP recording is a MAESTRO validation/test file are dropped from stage-1 train so
    stage-2 evaluation stays unseen.
  * stage 2: ``aligned_pianocore_maestro`` (MAESTRO crops), materialised augmentation (default 16x, schedule
    and ranges as ``preprocessing.amt_dense``: none + pitch +-12 + tempo x0.5-2 + velocity x0.5-1.5),
    MAESTRO's split (train; validation+test -> test).

Chunking: walk each piece in score order cutting consecutive chunks whose length is drawn from
``EXPRESSIFIER_CHUNK_LENGTHS`` with ``EXPRESSIFIER_CHUNK_PROBS`` (1/2 x 128, 1/4 x 64, 1/8 x 32, 1/8 x 16).
Each chunk is onset-relativised separately on each side (score and performance onsets minus the chunk's
earliest onset on that side) and encoded as dense-AMT quads, 4 tokens per note:
  input  = score       (onset, duration, note, EXPRESSIFIER_SCORE_VELOCITY), score-onset order
  output = performance (onset, duration, note, velocity), SAME note order (performance onsets may decrease)
Interpolated notes (in the score, not played) stay in both sequences but are masked out of the loss.

Output: ``<out>/<split>/<shard>/{inputs,outputs}.npy`` (int16 tokens), ``loss_mask.npy`` (bool per note),
``offsets.npy`` (chunk boundaries in notes), ``chunks.json``; plus ``<out>/manifest.json``. Each shard is
written to a temp dir and renamed when complete, and finished shards are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from orchestration_v2.constants import (
    AMT_DENSE_DURATION_OFFSET,
    AMT_DENSE_MAX_DURATION,
    AMT_DENSE_MAX_ONSET,
    AMT_DENSE_NOTE_OFFSET,
    AMT_DENSE_ONSET_OFFSET,
    AMT_DENSE_VELOCITY_OFFSET,
    EXPRESSIFIER_ALIGNED_DIRNAME,
    EXPRESSIFIER_CHUNK_LENGTHS,
    EXPRESSIFIER_CHUNK_PROBS,
    EXPRESSIFIER_MIN_CHUNK_NOTES,
    EXPRESSIFIER_PAIRS_DIRNAME,
    EXPRESSIFIER_SCORE_VELOCITY,
    EXPRESSIFIER_STAGE1_DATA_DIRNAME,
    EXPRESSIFIER_STAGE2_DATA_DIRNAME,
)

SCRATCH_ROOT = Path("/data/scratch-oc40/joel")
CS = 100  # dense-AMT time unit: centiseconds


# ------------------------------------------------------------------ chunking / tokens
def chunk_bounds(n_notes: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Consecutive [start, end) note ranges with lengths drawn from the chunk-length distribution."""
    out, i = [], 0
    lengths, probs = np.array(EXPRESSIFIER_CHUNK_LENGTHS), np.array(EXPRESSIFIER_CHUNK_PROBS)
    while i < n_notes:
        L = int(rng.choice(lengths, p=probs))
        j = min(n_notes, i + L)
        if j - i >= EXPRESSIFIER_MIN_CHUNK_NOTES:
            out.append((i, j))
        i = j
    return out


def _quads(onset_cs: np.ndarray, dur_cs: np.ndarray, pitch: np.ndarray, vel: np.ndarray) -> np.ndarray:
    q = np.stack([AMT_DENSE_ONSET_OFFSET + onset_cs, AMT_DENSE_DURATION_OFFSET + dur_cs,
                  AMT_DENSE_NOTE_OFFSET + pitch, AMT_DENSE_VELOCITY_OFFSET + vel], axis=1)
    return q.reshape(-1).astype(np.int16)


def tokenize_chunk(score_on: np.ndarray, score_off: np.ndarray, perf_on: np.ndarray, perf_off: np.ndarray,
                   pitch: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Score/performance arrays of one chunk (score order) -> (input tokens, output tokens), each 4N.
    None if a relative onset does not fit the dense-AMT onset range."""
    s_on = np.rint(score_on * CS).astype(np.int64)
    s_rel = s_on - s_on.min()
    p_on = np.rint(perf_on * CS).astype(np.int64)
    p_rel = p_on - p_on.min()
    if s_rel.max() > AMT_DENSE_MAX_ONSET or p_rel.max() > AMT_DENSE_MAX_ONSET:
        return None
    s_dur = np.clip(np.rint((score_off - score_on) * CS).astype(np.int64), 0, AMT_DENSE_MAX_DURATION)
    p_dur = np.clip(np.rint((perf_off - perf_on) * CS).astype(np.int64), 0, AMT_DENSE_MAX_DURATION)
    pitch = np.clip(pitch.astype(np.int64), 0, 127)
    vel = np.clip(velocity.astype(np.int64), 0, 127)
    inp = _quads(s_rel, s_dur, pitch, np.full_like(pitch, EXPRESSIFIER_SCORE_VELOCITY))
    out = _quads(p_rel, p_dur, pitch, vel)
    return inp, out


# ------------------------------------------------------------------ augmentation
def augmentation_schedule(factor: int) -> list[str]:
    """Same schedule as ``preprocessing.amt_dense._augmentation_schedule``: none + pitch/tempo/velocity."""
    if factor <= 0:
        raise ValueError(f"augmentation_factor must be positive, got {factor}")
    schedule, (per_type, extra) = ["none"], divmod(factor - 1, 3)
    for kind in ("pitch", "tempo", "velocity"):
        schedule += [kind] * (per_type + (1 if extra > 0 else 0))
        extra -= 1
    return schedule


def augment(piece: dict[str, np.ndarray], kind: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """pitch: same shift on score + performance (kept inside 0..127); tempo: same factor on every score and
    performance time; velocity: factor on performance velocities."""
    p = dict(piece)
    if kind == "pitch":
        lo, hi = max(-12, -int(p["pitch"].min())), min(12, 127 - int(p["pitch"].max()))
        p["pitch"] = p["pitch"] + int(rng.integers(lo, hi + 1))
    elif kind == "tempo":
        f = float(rng.uniform(0.5, 2.0))
        for k in ("score_on", "score_off", "perf_on", "perf_off"):
            p[k] = p[k] * f
    elif kind == "velocity":
        f = float(rng.uniform(0.5, 1.5))
        p["velocity"] = np.clip(np.rint(p["velocity"] * f), 1, 127)
    elif kind != "none":
        raise ValueError(kind)
    return p


def piece_rng(key: str, copy_idx: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode()), copy_idx])


def piece_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    df = df.sort_values("note_idx")
    return dict(score_on=df.score_onset_s.to_numpy(float), score_off=df.score_offset_s.to_numpy(float),
                perf_on=df.perf_onset_s.to_numpy(float), perf_off=df.perf_offset_s.to_numpy(float),
                pitch=df.pitch.to_numpy(np.int64), velocity=df.velocity.to_numpy(np.int64),
                interpolated=df.interpolated.to_numpy(bool))


def tokenize_piece(key: str, piece: dict[str, np.ndarray], schedule: list[str], seed: int):
    """Yield (input, output, loss_mask, aug_kind) for every chunk of every augmented copy of a piece."""
    for c, kind in enumerate(schedule):
        rng = piece_rng(key, c, seed)
        p = augment(piece, kind, rng)
        for a, b in chunk_bounds(len(p["pitch"]), rng):
            tok = tokenize_chunk(p["score_on"][a:b], p["score_off"][a:b], p["perf_on"][a:b], p["perf_off"][a:b],
                                 p["pitch"][a:b], p["velocity"][a:b])
            if tok is not None:
                yield tok[0], tok[1], ~p["interpolated"][a:b], kind


# ------------------------------------------------------------------ shards
def write_shard(dest: Path, chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]]) -> int:
    """chunks: (input, output, loss_mask, piece_key, aug_kind). Atomic: temp dir, then rename."""
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = np.array([len(m) for _, _, m, _, _ in chunks], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n)])
    np.save(tmp / "inputs.npy", np.concatenate([c[0] for c in chunks]) if chunks else np.zeros(0, np.int16))
    np.save(tmp / "outputs.npy", np.concatenate([c[1] for c in chunks]) if chunks else np.zeros(0, np.int16))
    np.save(tmp / "loss_mask.npy", np.concatenate([c[2] for c in chunks]) if chunks else np.zeros(0, bool))
    np.save(tmp / "offsets.npy", offsets)
    (tmp / "chunks.json").write_text(json.dumps([[c[3], c[4]] for c in chunks]))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return len(chunks)


def write_manifest(out: Path) -> dict:
    manifest = {"splits": {}}
    for split_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        shards = []
        for s in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")):
            offsets = np.load(s / "offsets.npy", mmap_mode="r")
            shards.append({"dir": str(s.relative_to(out)), "n_chunks": int(len(offsets) - 1),
                           "n_notes": int(offsets[-1])})
        manifest["splits"][split_dir.name] = shards
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for split, shards in manifest["splits"].items():
        print(f"{split}: {len(shards)} shards, {sum(s['n_chunks'] for s in shards)} chunks, "
              f"{sum(s['n_notes'] for s in shards)} notes", flush=True)
    return manifest


# ------------------------------------------------------------------ stage inputs
def stage1_batches(pairs_dir: Path, aligned_dir: Path | None):
    """Yield (shard_name, {split: [(key, df)]}) per expressifier_pairs batch file."""
    pairs = pd.read_parquet(pairs_dir / "pairs.parquet")
    pairs = pairs[pairs.status == "ok"].set_index("id")
    holdout = set()
    if aligned_dir is not None and (aligned_dir / "pieces.parquet").exists():
        pcs = pd.read_parquet(aligned_dir / "pieces.parquet")
        holdout = set(pcs.performance_id[pcs.maestro_split != "train"])
    for f in sorted((pairs_dir / "batches").glob("*.notes.parquet")):
        notes = pd.read_parquet(f)
        groups: dict[str, list] = {}
        for pid, df in notes.groupby("pair_id", sort=True):
            if pid not in pairs.index:
                continue
            meta = pairs.loc[pid]
            split = "test" if meta.split == "test" else "train"
            if split == "train" and meta.performance_id in holdout:
                continue  # its recording is a MAESTRO validation/test file: keep stage-2 eval unseen
            groups.setdefault(split, []).append((pid, df))
        yield f.name.replace(".notes.parquet", ""), groups


def stage2_batches(aligned_dir: Path, pieces_per_shard: int):
    pieces = pd.read_parquet(aligned_dir / "pieces.parquet")
    notes = pd.read_parquet(aligned_dir / "notes.parquet")
    split_of = {int(r.piece_id): ("train" if r.maestro_split == "train" else "test") for r in pieces.itertuples()}
    by_split: dict[str, list] = {}
    for pid, df in notes.groupby("piece_id", sort=True):
        by_split.setdefault(split_of[int(pid)], []).append((f"maestro_piece_{int(pid)}", df))
    for split, items in by_split.items():
        for k in range(0, len(items), pieces_per_shard):
            yield f"{split}_{k // pieces_per_shard:04d}", {split: items[k:k + pieces_per_shard]}


def run(stage: int, out: Path, *, source: Path, aligned_dir: Path | None, augmentation_factor: int, seed: int,
        pieces_per_shard: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    schedule = augmentation_schedule(augmentation_factor)
    gen = stage1_batches(source, aligned_dir) if stage == 1 else stage2_batches(source, pieces_per_shard)
    (out / "config.json").write_text(json.dumps(dict(stage=stage, source=str(source), seed=seed,
                                                     augmentation_factor=augmentation_factor, schedule=schedule,
                                                     chunk_lengths=EXPRESSIFIER_CHUNK_LENGTHS,
                                                     chunk_probs=EXPRESSIFIER_CHUNK_PROBS)))
    for shard_name, groups in gen:
        for split, items in groups.items():
            dest = out / split / shard_name
            if dest.exists():
                continue
            chunks = [(i, o, m, key, kind) for key, df in items
                      for i, o, m, kind in tokenize_piece(key, piece_arrays(df), schedule, seed)]
            n = write_shard(dest, chunks)
            print(f"{split}/{shard_name}: {len(items)} pieces -> {n} chunks", flush=True)
    write_manifest(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Chunk + tokenize expressifier (score, performance) pairs.")
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--source", type=Path, default=None, help="stage 1: pairs dir; stage 2: aligned dataset dir")
    ap.add_argument("--aligned-dir", type=Path, default=SCRATCH_ROOT / EXPRESSIFIER_ALIGNED_DIRNAME)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--augmentation-factor", type=int, default=None, help="default: 1 (stage 1), 16 (stage 2)")
    ap.add_argument("--pieces-per-shard", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.stage == 1:
        source = args.source or SCRATCH_ROOT / EXPRESSIFIER_PAIRS_DIRNAME
        out = args.output_dir or SCRATCH_ROOT / EXPRESSIFIER_STAGE1_DATA_DIRNAME
        factor = args.augmentation_factor or 1
    else:
        source = args.source or args.aligned_dir
        out = args.output_dir or SCRATCH_ROOT / EXPRESSIFIER_STAGE2_DATA_DIRNAME
        factor = args.augmentation_factor or 16
    run(args.stage, out, source=source, aligned_dir=args.aligned_dir if args.stage == 1 else None,
        augmentation_factor=factor, seed=args.seed, pieces_per_shard=args.pieces_per_shard)


if __name__ == "__main__":
    main()
