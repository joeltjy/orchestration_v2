"""Score <-> performance note alignment for the expressifier.

PianoCoRe stores, per performance, a *refined* score and a *refined* performance with a one-to-one note
alignment: refined score note ``i`` <-> refined performance note ``perf_idx[i]`` (both sorted by
(tick time, pitch)); ``interpolated[i]`` marks score notes the pianist did not play (PianoCoRe filled them in).

The refined performance is re-timed (a global sync shift plus piecewise ``ShiftOnset`` shifts written as MIDI
markers), so its onsets/offsets are not the recording's. ``align_refined_to_reference`` maps every played
refined note onto the matching note of a *reference* recording (PianoCoRe's original performance MIDI, or the
original MAESTRO MIDI) and takes onset, offset and velocity from there:

* the refined performance is split into segments at its ShiftOnset markers;
* per segment, the shift to the reference is the candidate offset under which the most notes find a
  same-pitch reference note within 20 ms;
* every played note is matched one-to-one to the same-pitch reference note whose onset agrees within
  ``match_tol`` after that shift (notes near a boundary also try the other segments' shifts);
* interpolated notes keep PianoCoRe's timing, moved by the same local shift.
"""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import symusic

MATCH_TOL = 0.010   # s
VOTE_TOL = 0.020    # s
MIN_MATCHED = 0.98  # share of played notes that must find a reference note

Note = tuple[float, float, int, int]  # onset s, offset s, pitch, velocity


def notes_from_midi(midi: bytes | str) -> list[Note]:
    """All notes of a MIDI file (bytes or path), sorted by (onset, offset, pitch, velocity), in seconds."""
    if isinstance(midi, (bytes, bytearray)):
        score = symusic.Score.from_midi(bytes(midi), ttype="second")
    else:
        score = symusic.Score.from_file(str(midi), ttype="second")
    return sorted((n.time, n.end, n.pitch, n.velocity) for tr in score.tracks for n in tr.notes)


def sorted_by_tick(midi: bytes) -> list[dict]:
    """Notes sorted by (tick time, pitch) with beats and seconds (PianoCoRe's alignment order)."""
    st = symusic.Score.from_midi(bytes(midi), ttype="tick")
    ss = symusic.Score.from_midi(bytes(midi), ttype="second")
    nt = [(n.time, n.end, n.pitch, n.velocity) for tr in st.tracks for n in tr.notes]
    ns = [(n.time, n.end) for tr in ss.tracks for n in tr.notes]  # same note order as nt
    tpq = st.ticks_per_quarter
    order = sorted(range(len(nt)), key=lambda i: (nt[i][0], nt[i][2]))
    return [dict(on_b=nt[i][0] / tpq, off_b=nt[i][1] / tpq, on_s=ns[i][0], off_s=ns[i][1],
                 pitch=nt[i][2], vel=nt[i][3]) for i in order]


def shift_markers(midi: bytes) -> list[float]:
    """Times of PianoCoRe's ``ShiftOnset_<shift>_<time>`` markers in a refined performance."""
    sc = symusic.Score.from_midi(bytes(midi), ttype="second")
    return sorted(float(m.text.split("_")[2]) for m in sc.markers if m.text.startswith("ShiftOnset_"))


def match_notes(ref_on: np.ndarray, ref_pitch: np.ndarray, ref: list[Note], shift, tol: float,
                used: set[int] | None = None) -> np.ndarray:
    """One-to-one: note i -> closest unused reference note of the same pitch within ``tol`` of
    ``ref_on[i] + shift(i)``. Returns reference indices (-1 = no match)."""
    by_p: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for j, (t, _e, p, _v) in enumerate(ref):
        by_p[p].append((t, j))
    tims = {p: np.array([t for t, _ in v]) for p, v in by_p.items()}
    used = set() if used is None else used
    out = np.full(len(ref_on), -1, dtype=np.int64)
    for i in np.argsort(ref_on):
        p = int(ref_pitch[i])
        if p not in tims:
            continue
        target = ref_on[i] + shift(i)
        ts = tims[p]
        lo, hi = np.searchsorted(ts, target - tol), np.searchsorted(ts, target + tol, side="right")
        best, bj = tol, -1
        for q in range(lo, hi):
            j = by_p[p][q][1]
            d = abs(ts[q] - target)
            if d <= best and j not in used:
                best, bj = d, j
        if bj >= 0:
            used.add(bj)
            out[i] = bj
    return out


@dataclass
class AlignedNotes:
    """One row per refined score note, in score order (sorted by score tick, pitch)."""

    pitch: np.ndarray
    score_onset_beats: np.ndarray
    score_offset_beats: np.ndarray
    score_onset_s: np.ndarray
    score_offset_s: np.ndarray
    score_velocity: np.ndarray
    perf_onset_s: np.ndarray
    perf_offset_s: np.ndarray
    velocity: np.ndarray
    interpolated: np.ndarray
    ref_index: np.ndarray  # index into the reference notes (-1 if interpolated)
    info: dict


def align_refined_to_reference(refined_score_midi: bytes, refined_performance_midi: bytes,
                               refined_alignment: bytes, reference: list[Note], *,
                               match_tol: float = MATCH_TOL,
                               min_matched: float = MIN_MATCHED) -> tuple[AlignedNotes | None, str | None]:
    """Align PianoCoRe's refined score to a reference recording. Returns (aligned, None) or (None, reason)."""
    S = sorted_by_tick(refined_score_midi)
    P = sorted_by_tick(refined_performance_midi)
    z = np.load(io.BytesIO(refined_alignment))
    pidx, interp = z["perf_idx"].astype(np.int64), z["interpolated"].astype(bool)
    n = len(S)
    if len(P) != n or len(pidx) != n:
        return None, "refined score/performance lengths differ"
    perf = [P[pidx[i]] for i in range(n)]
    if any(perf[i]["pitch"] != S[i]["pitch"] for i in range(n)):
        return None, "pitch mismatch between refined score and performance"
    on = np.array([x["on_s"] for x in perf])
    pitch = np.array([x["pitch"] for x in perf])
    played = ~interp
    tims: dict[int, np.ndarray] = {}
    for p in np.unique([r[2] for r in reference]) if reference else []:
        tims[int(p)] = np.array([r[0] for r in reference if r[2] == p])

    def hits(idx: np.ndarray, d: float) -> int:
        c = 0
        for i in idx:
            ts = tims.get(int(pitch[i]))
            if ts is None:
                continue
            k = np.searchsorted(ts, on[i] + d)
            if (k < len(ts) and abs(ts[k] - on[i] - d) <= VOTE_TOL) or (k > 0 and abs(ts[k - 1] - on[i] - d) <= VOTE_TOL):
                c += 1
        return c

    def vote(idx: np.ndarray) -> float | None:
        idx = idx[:120]
        cands = {round(t - on[i], 3) for i in idx[:6] for t in tims.get(int(pitch[i]), ())}
        return max(cands, key=lambda d: hits(idx, d)) if cands else None

    bnds = np.array(shift_markers(refined_performance_midi))
    seg = np.searchsorted(bnds, on, side="right") if len(bnds) else np.zeros(n, dtype=int)
    shifts: dict[int, float] = {}
    for k in np.unique(seg):
        idx = np.where(played & (seg == k))[0]
        if len(idx):
            d = vote(idx)
            if d is not None:
                shifts[int(k)] = d
    if not shifts:
        return None, "no shift found"
    ks = sorted(shifts)
    loc = np.array([shifts.get(int(seg[i]), shifts[min(ks, key=lambda q: abs(q - seg[i]))]) for i in range(n)])
    idx_played = np.where(played)[0]
    used: set[int] = set()
    mj = np.full(n, -1, dtype=np.int64)
    mj[idx_played] = match_notes(on[idx_played], pitch[idx_played], reference,
                                 lambda i: loc[idx_played][i], match_tol, used)
    for alt in sorted(set(shifts.values())):  # notes near a segment boundary
        miss = np.where(played & (mj < 0))[0]
        if not len(miss):
            break
        m3 = match_notes(on[miss], pitch[miss], reference, lambda i: alt, match_tol, used)
        for i, j in zip(miss, m3):
            if j >= 0:
                mj[i], loc[i] = j, alt
    share = float((mj[played] >= 0).mean()) if played.any() else 1.0
    if share < min_matched:
        return None, f"only {share:.1%} of played notes matched the reference"
    ok = mj >= 0
    err = np.abs(np.array([reference[j][0] for j in mj[ok]]) - (on[ok] + loc[ok])) if ok.any() else np.zeros(0)
    aligned = AlignedNotes(
        pitch=pitch.astype(np.int16),
        score_onset_beats=np.array([x["on_b"] for x in S]), score_offset_beats=np.array([x["off_b"] for x in S]),
        score_onset_s=np.array([x["on_s"] for x in S]), score_offset_s=np.array([x["off_s"] for x in S]),
        score_velocity=np.array([x["vel"] for x in S], dtype=np.int16),
        perf_onset_s=np.array([reference[mj[i]][0] if mj[i] >= 0 else perf[i]["on_s"] + loc[i] for i in range(n)]),
        perf_offset_s=np.array([reference[mj[i]][1] if mj[i] >= 0 else perf[i]["off_s"] + loc[i] for i in range(n)]),
        velocity=np.array([reference[mj[i]][3] if mj[i] >= 0 else perf[i]["vel"] for i in range(n)], dtype=np.int16),
        interpolated=interp, ref_index=mj,
        info=dict(n_notes=n, n_interpolated=int(interp.sum()), n_played=int(played.sum()),
                  n_matched=int(ok.sum()), unmatched_played=int((played & ~ok).sum()),
                  shift_median_s=float(np.median(loc)), shift_range_s=float(loc.max() - loc.min()),
                  max_onset_err_ms=float(1000 * err.max()) if len(err) else 0.0),
    )
    return aligned, None


__all__ = ["AlignedNotes", "MATCH_TOL", "align_refined_to_reference", "match_notes", "notes_from_midi",
           "shift_markers", "sorted_by_tick"]
