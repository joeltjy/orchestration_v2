"""Expressify the piano parts of SymphonyNet's classical piano concertos (see EXPRESSIFIER.md).

For every piece: merge all acoustic-piano tracks (GM program 0-1) into one part, split it into continuous
segments (a new segment starts after >= 8 s with no piano note sounding), run each segment through the
expressifier with a sliding window (128 notes; once full, keep the second half as a forced decoder prefix and
continue), and store a row-aligned mapping

    original (onset, dur, pitch, velocity)  ->  expressive (onset, dur, pitch, velocity)

with both onsets relative to their segment's start (each segment starts at 0) plus ``t_start`` and the exact
absolute original onset on every row.

    python -m orchestration_v2.sampling.expressify_symphonynet \\
        --checkpoint /data/scratch-oc40/joel/checkpoints/expressifier_stage2/best \\
        --output-dir /data/scratch-oc40/joel/expressified/symphonynet_piano_concertos/expressifier_stage2_best

One ``<piece>.parquet`` (+ ``<piece>.segments.parquet``) per piece, written atomically; pieces already written
are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from orchestration_v2.constants import (
    AMT_DENSE_DURATION_OFFSET,
    AMT_DENSE_MAX_DURATION,
    AMT_DENSE_MAX_ONSET,
    AMT_DENSE_NOTE_OFFSET,
    AMT_DENSE_ONSET_OFFSET,
    AMT_DENSE_VELOCITY_OFFSET,
    EXPRESSIFIER_MAX_NOTES,
    EXPRESSIFIER_SCORE_VELOCITY,
    EXPRESSIFIER_TOKENS_PER_NOTE,
)

T = EXPRESSIFIER_TOKENS_PER_NOTE
CS = 100  # dense-AMT time unit: centiseconds
CLASSICAL_ROOT = Path("/data/scratch-oc40/joel/symphonynet/classical")
PIECE_LIST = "symphony_net_pieces_list.txt"
PIANO_PROGRAMS = (0, 1)  # acoustic grand, bright acoustic
SEGMENT_GAP_S = 8.0
MIN_WINDOW_NOTES = 4
# "concert" in the name and a piano track, but not a piano concerto (orchestral piano / harpsichord as program 0)
NOT_PIANO_CONCERTO = re.compile(r"violin|brandeburg", re.I)


# ------------------------------------------------------------------ piece selection / piano part
def piano_concerto_files(root: Path = CLASSICAL_ROOT) -> list[Path]:
    """Classical files whose name contains 'concert', that exist, are not excluded, and have a piano track."""
    import pretty_midi

    names = [ln.strip() for ln in (root / PIECE_LIST).read_text().splitlines() if re.search("concert", ln, re.I)]
    out = []
    for name in names:
        path = root / name
        if not path.is_file() or NOT_PIANO_CONCERTO.search(name):
            continue
        pm = pretty_midi.PrettyMIDI(str(path))
        if any(not i.is_drum and i.program in PIANO_PROGRAMS and i.notes for i in pm.instruments):
            out.append(path)
    return out


def piano_part(midi_path: Path) -> pd.DataFrame:
    """All notes of all acoustic-piano instruments, in score order (onset, pitch), one row per original note.

    Exact duplicates (same onset and pitch, e.g. the same note on two piano tracks) cannot be played twice: the
    longest is kept and the others get ``dup_of`` = the kept row (row = position in this frame)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    rows = []
    for k, inst in enumerate(pm.instruments):
        if inst.is_drum or inst.program not in PIANO_PROGRAMS:
            continue
        for j, n in enumerate(inst.notes):
            rows.append((k, j, float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    df = pd.DataFrame(rows, columns=["instrument", "note_index_in_instrument", "onset", "offset", "pitch",
                                     "velocity"])
    return order_and_dedupe(df)


def order_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(_dur=df.offset - df.onset)
    df = df.sort_values(["onset", "pitch", "_dur", "instrument", "note_index_in_instrument"],
                        ascending=[True, True, False, True, True], kind="mergesort").reset_index(drop=True)
    key = list(zip(np.round(df.onset.to_numpy(), 6), df.pitch.to_numpy()))
    first: dict[tuple, int] = {}
    dup_of = np.full(len(df), -1, dtype=np.int64)
    for r, k in enumerate(key):
        if k in first:
            dup_of[r] = first[k]
        else:
            first[k] = r
    return df.drop(columns="_dur").assign(dup_of=dup_of)


def segment_ids(onset: np.ndarray, offset: np.ndarray, gap: float = SEGMENT_GAP_S) -> np.ndarray:
    """Segment id per note (score order): a new segment starts when no note has sounded for >= ``gap`` s."""
    seg = np.zeros(len(onset), dtype=np.int64)
    sounding_until, s = -np.inf, 0
    for i, (on, off) in enumerate(zip(onset, offset)):
        if i > 0 and on - sounding_until >= gap:
            s += 1
        seg[i] = s
        sounding_until = max(sounding_until, off)
    return seg


# ------------------------------------------------------------------ sliding-window generation
def _quads(on: np.ndarray, du: np.ndarray, pitch: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return np.stack([AMT_DENSE_ONSET_OFFSET + on, AMT_DENSE_DURATION_OFFSET + du, AMT_DENSE_NOTE_OFFSET + pitch,
                     AMT_DENSE_VELOCITY_OFFSET + vel], axis=1).reshape(-1).astype(np.int64)


@dataclass
class SegmentJob:
    """One (piece, segment, sample): score notes in score order, performance filled window by window."""
    key: str
    segment: int
    sample: int
    score_on_cs: np.ndarray  # absolute score onsets, centiseconds (rint, as in training)
    score_dur_cs: np.ndarray  # clipped to the model's duration range
    pitch: np.ndarray
    seed: int
    window: int = EXPRESSIFIER_MAX_NOTES
    perf_on_cs: np.ndarray = field(init=False)  # absolute (segment-local) expressive onsets
    perf_dur_cs: np.ndarray = field(init=False)
    perf_vel: np.ndarray = field(init=False)
    window_first: np.ndarray = field(init=False)
    done_notes: int = 0
    n_windows: int = 0
    n_onset_clipped: int = 0

    def __post_init__(self):
        n = len(self.pitch)
        self.perf_on_cs = np.zeros(n, np.int64)
        self.perf_dur_cs = np.zeros(n, np.int64)
        self.perf_vel = np.zeros(n, np.int64)
        self.window_first = np.full(n, -1, np.int64)
        self._rng = None

    @property
    def n(self) -> int:
        return len(self.pitch)

    @property
    def finished(self) -> bool:
        return self.done_notes >= self.n

    def torch_generator(self):
        import torch

        if self._rng is None:
            self._rng = torch.Generator().manual_seed(self.seed)
        return self._rng

    def next_window(self) -> tuple[int, int]:
        """[start, end) of the next window: up to window/2 new notes after a forced prefix, as much prefix as
        fits in ``window`` notes. Halved while the score span or the prefix's performance span would exceed the
        model's onset range."""
        g, w = self.done_notes, self.window
        while True:
            end = min(self.n, w if g == 0 else g + w // 2)
            start = 0 if g == 0 else max(0, end - w)
            s_span = self.score_on_cs[start:end].max() - self.score_on_cs[start:end].min()
            p_span = (self.perf_on_cs[start:g].max() - self.perf_on_cs[start:g].min()) if g > start else 0
            if (s_span <= AMT_DENSE_MAX_ONSET and p_span <= AMT_DENSE_MAX_ONSET) or w <= MIN_WINDOW_NOTES:
                return start, end
            w //= 2

    def window_tokens(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        """(score tokens, prefix tokens padded to the window, prefix length in tokens, ref) for a window."""
        g = self.done_notes
        s_rel = self.score_on_cs[start:end] - self.score_on_cs[start:end].min()
        self.n_onset_clipped += int((s_rel > AMT_DENSE_MAX_ONSET).sum())
        inp = _quads(np.minimum(s_rel, AMT_DENSE_MAX_ONSET), self.score_dur_cs[start:end], self.pitch[start:end],
                     np.full(end - start, EXPRESSIFIER_SCORE_VELOCITY))
        prefix = np.zeros_like(inp)
        ref = 0
        if g > start:
            p = self.perf_on_cs[start:g]
            ref = int(p.min())
            self.n_onset_clipped += int((p - ref > AMT_DENSE_MAX_ONSET).sum())
            prefix[:T * (g - start)] = _quads(np.minimum(p - ref, AMT_DENSE_MAX_ONSET), self.perf_dur_cs[start:g],
                                              self.pitch[start:g], self.perf_vel[start:g])
        return inp, prefix, T * (g - start), ref

    def apply(self, start: int, end: int, out: np.ndarray, ref: int) -> None:
        g = self.done_notes
        q = out[:T * (end - start)].reshape(-1, T)[g - start:]
        self.perf_on_cs[g:end] = ref + (q[:, 0] - AMT_DENSE_ONSET_OFFSET)
        self.perf_dur_cs[g:end] = q[:, 1] - AMT_DENSE_DURATION_OFFSET
        self.perf_vel[g:end] = q[:, 3] - AMT_DENSE_VELOCITY_OFFSET
        self.window_first[g:end] = self.n_windows
        self.n_windows += 1
        self.done_notes = end


def run_window_batch(model, jobs: list[SegmentJob], *, temperature: float, top_p: float | None,
                     device) -> None:
    """One window step for each job (batched, right-padded)."""
    import torch

    specs = []
    for j in jobs:
        start, end = j.next_window()
        inp, prefix, plen, ref = j.window_tokens(start, end)
        # per-job uniforms: the job's own generator, so samples do not depend on batch composition
        u = torch.rand(len(inp), generator=j.torch_generator(), dtype=torch.float64)
        specs.append((j, start, end, inp, prefix, plen, ref, u))
    L = max(len(s[3]) for s in specs)
    B = len(specs)
    ids = torch.zeros(B, L, dtype=torch.long)
    pre = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.long)
    uni = torch.full((B, L), 0.5, dtype=torch.float64)
    plens = torch.zeros(B, dtype=torch.long)
    for b, (_, _, _, inp, prefix, plen, _, u) in enumerate(specs):
        n = len(inp)
        ids[b, :n] = torch.from_numpy(inp)
        pre[b, :n] = torch.from_numpy(prefix)
        mask[b, :n] = 1
        uni[b, :n] = u
        plens[b] = plen
    out = model.generate(ids.to(device), mask.to(device), temperature=temperature, top_p=top_p,
                         prefix_ids=pre, prefix_len=plens, uniforms=uni.to(device)).cpu().numpy()
    for b, (j, start, end, _, _, _, ref, _) in enumerate(specs):
        j.apply(start, end, out[b], ref)


def run_jobs(model, jobs: Iterable[SegmentJob], *, batch_size: int, temperature: float = 1.0,
             top_p: float | None = 0.95, device="cpu",
             on_done: Callable[[SegmentJob], None] | None = None, log: Callable[[str], None] | None = None) -> None:
    """Keep up to ``batch_size`` segments in flight; each round advances every in-flight segment one window."""
    it: Iterator[SegmentJob] = iter(jobs)
    active: list[SegmentJob] = []
    exhausted, rounds, t0 = False, 0, time.time()
    while True:
        while not exhausted and len(active) < batch_size:
            nxt = next(it, None)
            if nxt is None:
                exhausted = True
            elif nxt.n == 0:
                if on_done:
                    on_done(nxt)
            else:
                active.append(nxt)
        if not active:
            return
        run_window_batch(model, active, temperature=temperature, top_p=top_p, device=device)
        rounds += 1
        still = []
        for j in active:
            if j.finished:
                if on_done:
                    on_done(j)
            else:
                still.append(j)
        active = still
        if log and rounds % 10 == 0:
            log(f"round {rounds}: {len(active)} segments in flight, {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ per-piece assembly
def piece_jobs(key: str, notes: pd.DataFrame, *, num_samples: int, seed: int,
               window: int = EXPRESSIFIER_MAX_NOTES) -> list[SegmentJob]:
    kept = notes[notes.dup_of < 0]
    jobs = []
    for s in range(num_samples):
        for seg, g in kept.groupby("segment", sort=True):
            on_cs = np.rint(g.onset.to_numpy() * CS).astype(np.int64)
            dur_cs = np.clip(np.rint((g.offset - g.onset).to_numpy() * CS).astype(np.int64), 0,
                             AMT_DENSE_MAX_DURATION)
            jobs.append(SegmentJob(key=key, segment=int(seg), sample=s, score_on_cs=on_cs, score_dur_cs=dur_cs,
                                   pitch=g.pitch.to_numpy(np.int64), window=window,
                                   seed=zlib.crc32(f"{seed}/{key}/{seg}/{s}".encode())))
    return jobs


def prepare_notes(notes: pd.DataFrame, gap: float = SEGMENT_GAP_S) -> pd.DataFrame:
    """Segment ids (from the kept notes; duplicates inherit their kept note's) and t_start per segment."""
    notes = notes.copy()
    kept = notes.dup_of < 0
    seg = np.full(len(notes), -1, np.int64)
    seg[kept.to_numpy()] = segment_ids(notes.onset[kept].to_numpy(), notes.offset[kept].to_numpy(), gap)
    dup = ~kept.to_numpy()
    seg[dup] = seg[notes.dup_of.to_numpy()[dup]]
    notes["segment"] = seg
    notes["t_start"] = notes.groupby("segment").onset.transform("min")
    return notes


def assemble(key: str, notes: pd.DataFrame, jobs: list[SegmentJob]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-aligned mapping (one row per original note per sample) and the segment table."""
    frames, segs = [], []
    kept_rows = np.flatnonzero(notes.dup_of.to_numpy() < 0)
    for s in sorted({j.sample for j in jobs}):
        expr_on = np.zeros(len(notes))
        expr_dur = np.zeros(len(notes))
        expr_vel = np.zeros(len(notes), np.int64)
        wfirst = np.full(len(notes), -1, np.int64)
        for j in (j for j in jobs if j.sample == s):
            rows = kept_rows[notes.segment.to_numpy()[kept_rows] == j.segment]
            expr_on[rows] = (j.perf_on_cs - j.perf_on_cs.min()) / CS
            expr_dur[rows] = j.perf_dur_cs / CS
            expr_vel[rows] = j.perf_vel
            wfirst[rows] = j.window_first
            if s == 0:
                g = notes.iloc[rows]
                segs.append(dict(segment=j.segment, t_start=float(g.onset.min()), t_end=float(g.onset.max()),
                                 n_notes=int((notes.segment == j.segment).sum()), n_kept_notes=j.n,
                                 n_windows=j.n_windows, short=j.n < MIN_WINDOW_NOTES,
                                 n_onset_clipped=j.n_onset_clipped))
        dup = np.flatnonzero(notes.dup_of.to_numpy() >= 0)
        src = notes.dup_of.to_numpy()[dup]
        expr_on[dup], expr_dur[dup], expr_vel[dup], wfirst[dup] = expr_on[src], expr_dur[src], expr_vel[src], wfirst[src]
        frames.append(pd.DataFrame(dict(
            piece=key, segment=notes.segment.to_numpy(), sample=s, row=np.arange(len(notes)),
            instrument=notes.instrument.to_numpy(), note_index_in_instrument=notes.note_index_in_instrument.to_numpy(),
            t_start=notes.t_start.to_numpy(), orig_onset=(notes.onset - notes.t_start).to_numpy(),
            orig_onset_abs=notes.onset.to_numpy(), orig_dur=(notes.offset - notes.onset).to_numpy(),
            pitch=notes.pitch.to_numpy(), orig_velocity=notes.velocity.to_numpy(),
            expr_onset=expr_on, expr_dur=expr_dur, expr_pitch=notes.pitch.to_numpy(), expr_velocity=expr_vel,
            dup_of=notes.dup_of.to_numpy(), window_first=wfirst)))
    mapping = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    seg_df = pd.DataFrame(segs, columns=["segment", "t_start", "t_end", "n_notes", "n_kept_notes", "n_windows",
                                         "short", "n_onset_clipped"])
    return mapping, seg_df.sort_values("segment").reset_index(drop=True)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_mapping(path: str | Path, sample: int = 0) -> dict[tuple[int, tuple], tuple]:
    """{(segment, (orig_onset, orig_dur, pitch, orig_velocity)): (expr_onset, expr_dur, pitch, expr_velocity)}.
    Convenience only: original tuples are not guaranteed unique, the parquet rows are the real mapping."""
    df = pd.read_parquet(path)
    df = df[df["sample"] == sample]
    return {(int(r.segment), (r.orig_onset, r.orig_dur, int(r.pitch), int(r.orig_velocity))):
            (r.expr_onset, r.expr_dur, int(r.expr_pitch), int(r.expr_velocity)) for r in df.itertuples()}


# ------------------------------------------------------------------ CLI
def main(argv=None) -> None:
    import torch

    from orchestration_v2.models.expressifier import load_expressifier_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=CLASSICAL_ROOT)
    ap.add_argument("--limit", type=int, default=None, help="first N pieces only (dry run)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=float, default=SEGMENT_GAP_S)
    ap.add_argument("--window", type=int, default=EXPRESSIFIER_MAX_NOTES)
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    files = piano_concerto_files(args.root)
    if args.limit:
        files = files[:args.limit]
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / "manifest.json").write_text(json.dumps(dict(settings=settings, files=[f.name for f in files]), indent=1))
    todo = [f for f in files if not (out / f"{f.stem}.parquet").exists()]
    log(f"{len(files)} piano concertos, {len(files) - len(todo)} already done, {len(todo)} to do")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_expressifier_checkpoint(args.checkpoint, device=device)
    pending: dict[str, dict] = {}

    def jobs():
        for f in todo:
            notes = prepare_notes(piano_part(f), args.gap)
            js = piece_jobs(f.stem, notes, num_samples=args.num_samples, seed=args.seed, window=args.window)
            pending[f.stem] = dict(notes=notes, jobs=js, left=len(js))
            if not js:  # no piano notes at all: nothing to generate
                write_piece(f.stem)
            log(f"queued {f.stem}: {len(notes)} piano notes, {notes.segment.nunique()} segments")
            yield from js

    def write_piece(key: str) -> None:
        p = pending.pop(key)
        mapping, segs = assemble(key, p["notes"], p["jobs"])
        _atomic_parquet(segs, out / f"{key}.segments.parquet")
        _atomic_parquet(mapping, out / f"{key}.parquet")  # written last: its presence = piece done
        log(f"wrote {key}: {len(mapping)} rows, {len(segs)} segments, "
            f"{int(segs.n_windows.sum()) if len(segs) else 0} windows, "
            f"{int(segs.n_onset_clipped.sum()) if len(segs) else 0} clipped onsets")

    def on_done(j: SegmentJob) -> None:
        pending[j.key]["left"] -= 1
        if pending[j.key]["left"] == 0:
            write_piece(j.key)

    run_jobs(model, jobs(), batch_size=args.batch_size, temperature=args.temperature, top_p=args.top_p,
             device=device, on_done=on_done, log=log)
    log("done")


if __name__ == "__main__":
    main()
