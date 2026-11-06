"""Stage-1 expressifier data: (score, performance) note pairs for every PianoCoRe-A* performance.

For each A* row the refined score/performance alignment is mapped onto PianoCoRe's ORIGINAL performance
MIDI (``performance_midi_bytes``), so performance onsets, offsets and velocities are the recording's, not
the refined re-timed copy (see ``expressifier_alignment``).

Saves as it goes: rows are processed in batches of ``--rows-per-batch``; each finished batch is written as
``<out>/batches/shard<SS>_batch<BBBB>.notes.parquet`` + ``.pairs.json`` and skipped on rerun. One slurm array
task per PianoCoRe parquet shard (``--shard-index``). Progress is printed per batch.

    python -u -m orchestration.preprocessing.expressifier_pairs --shard-index 0 --num-workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from orchestration_v2.constants import EXPRESSIFIER_PAIRS_DIRNAME
from orchestration_v2.preprocessing.expressifier_alignment import align_refined_to_reference, notes_from_midi

SCRATCH_ROOT = Path("/data/scratch-oc40/joel")
PIANOCORE_PARQUET_DIR = SCRATCH_ROOT / "piano_core_dataset" / "parquet" / "data"
DEFAULT_OUTPUT_DIR = SCRATCH_ROOT / EXPRESSIFIER_PAIRS_DIRNAME
META_COLS = ["id", "composer", "composition", "movement", "performance_id", "performance_dataset", "split"]
BYTE_COLS = ["performance_midi_bytes", "refined_score_midi_bytes", "refined_performance_midi_bytes",
             "refined_alignment_bytes"]


def pair_notes_frame(pair_id: str, aligned) -> pd.DataFrame:
    a = aligned
    return pd.DataFrame(dict(
        pair_id=pair_id, note_idx=np.arange(len(a.pitch), dtype=np.int32), pitch=a.pitch.astype(np.int16),
        score_onset_beats=a.score_onset_beats, score_offset_beats=a.score_offset_beats,
        score_onset_s=a.score_onset_s, score_offset_s=a.score_offset_s, score_velocity=a.score_velocity,
        perf_onset_s=a.perf_onset_s, perf_offset_s=a.perf_offset_s, velocity=a.velocity.astype(np.int16),
        interpolated=a.interpolated))


def process_row(row: dict) -> tuple[pd.DataFrame | None, dict]:
    meta = {k: row[k] for k in META_COLS}
    try:
        reference = notes_from_midi(row["performance_midi_bytes"])
        aligned, why = align_refined_to_reference(row["refined_score_midi_bytes"],
                                                  row["refined_performance_midi_bytes"],
                                                  row["refined_alignment_bytes"], reference)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the shard
        aligned, why = None, f"error: {type(exc).__name__}: {exc}"
    if aligned is None:
        return None, dict(meta, status="skipped", reason=why)
    # reference = original performance: time it from its first note
    t0 = float(np.min(aligned.perf_onset_s[~aligned.interpolated])) if (~aligned.interpolated).any() else 0.0
    aligned.perf_onset_s = aligned.perf_onset_s - t0
    aligned.perf_offset_s = aligned.perf_offset_s - t0
    return pair_notes_frame(row["id"], aligned), dict(meta, status="ok", **aligned.info)


def process_batch(rows: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    frames, infos = [], []
    for row in rows:
        df, info = process_row(row)
        infos.append(info)
        if df is not None:
            frames.append(df)
    notes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return notes, infos


def run_shard(shard_path: Path, output_dir: Path, *, shard_index: int, rows_per_batch: int, num_workers: int,
              max_rows: int | None = None) -> None:
    out = output_dir / "batches"
    out.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(shard_path)
    meta = pf.read(columns=META_COLS + ["tier_a_star", "is_refined"]).to_pandas()
    keep = np.where(meta.tier_a_star.fillna(False) & meta.is_refined.fillna(False))[0]
    if max_rows is not None:
        keep = keep[:max_rows]
    batches = [keep[i:i + rows_per_batch] for i in range(0, len(keep), rows_per_batch)]
    todo = [b for b, idx in enumerate(batches) if not (out / f"shard{shard_index:02d}_batch{b:04d}.pairs.json").exists()]
    print(f"{shard_path.name}: {len(keep)} A* rows in {len(batches)} batches, {len(todo)} to do", flush=True)
    if not todo:
        return
    table = pf.read(columns=META_COLS + BYTE_COLS)
    from concurrent.futures import FIRST_COMPLETED, wait

    pending = list(todo)
    inflight: dict = {}
    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        while pending or inflight:
            while pending and len(inflight) < 2 * num_workers:  # bounded: only a few batches' bytes in flight
                b = pending.pop(0)
                rows = table.take(pa_indices(batches[b])).to_pylist()
                inflight[pool.submit(process_batch, rows)] = b
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                b = inflight.pop(fut)
                notes, infos = fut.result()
                stem = out / f"shard{shard_index:02d}_batch{b:04d}"
                if len(notes):
                    notes.to_parquet(f"{stem}.notes.parquet", index=False, compression="zstd")
                Path(f"{stem}.pairs.json").write_text(json.dumps(infos, default=float))  # last = done marker
                done += 1
                ok = sum(1 for i in infos if i["status"] == "ok")
                print(f"[{done}/{len(todo)}] {stem.name}: {ok}/{len(infos)} pairs ok, {len(notes)} notes",
                      flush=True)


def pa_indices(idx):
    import pyarrow as pa

    return pa.array(np.asarray(idx, dtype=np.int64))


def merge(output_dir: Path) -> None:
    """Collect all batch infos into pairs.parquet (notes stay in per-batch files)."""
    infos = []
    for p in sorted((output_dir / "batches").glob("*.pairs.json")):
        for x in json.loads(p.read_text()):
            x["batch"] = p.name.replace(".pairs.json", "")
            infos.append(x)
    pairs = pd.DataFrame(infos)
    pairs.to_parquet(output_dir / "pairs.parquet", index=False)
    ok = pairs[pairs.status == "ok"]
    skipped = pairs[pairs.status != "ok"]
    reasons = skipped.reason.astype(str).str.slice(0, 40).value_counts().head(5).to_dict() if len(skipped) else {}
    print(f"merged: {len(ok)} ok pairs / {len(pairs)} rows; notes {int(ok.n_notes.sum()) if len(ok) else 0}; "
          f"skipped reasons: {reasons}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet-dir", type=Path, default=PIANOCORE_PARQUET_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--shard-index", type=int, default=None, help="one PianoCoRe parquet file (sorted order)")
    ap.add_argument("--rows-per-batch", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: first N A* rows per shard")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args(argv)
    shards = sorted(glob.glob(str(args.parquet_dir / "*.parquet")))
    if not args.merge_only:
        indices = range(len(shards)) if args.shard_index is None else [args.shard_index]
        for k in indices:
            run_shard(Path(shards[k]), args.output_dir, shard_index=k, rows_per_batch=args.rows_per_batch,
                      num_workers=args.num_workers, max_rows=args.max_rows)
    if args.merge_only or args.shard_index is None:
        merge(args.output_dir)


if __name__ == "__main__":
    main()
