"""Outputs of the v1 reduction: reduced MIDI, side-by-side MIDI, and data for the piano-roll viewer.

* ``write_reduced_midi``: the notes present after a step (default 7), with merged BPE notes unpacked through the
  merge classes (as a decoder would), on the original tempo map and ticks per beat. One track per original track
  (split further if a track holds several tax-20 instruments).
* ``write_side_by_side_midi``: the original file's tracks, each followed by its reduced track, for MuseScore.
* ``write_viewer_data``: one ``data/p<id>.js`` per piece plus ``data/index.js``, read by ``viewer.html``.
* ``write_vocabulary``: ``data/vocab.js``, the BPE merge classes (steps 6-7) and repeated patterns (step 5) for the
  viewer's Vocabulary tab.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import mido

from orchestration_v2.reduction_v1 import (
    MEASURE_STEPS,
    NO_STEP,
    TAX20_GM,
    TAX20_NAMES,
    UNPITCHED,
    Corpus,
    Piece,
    class_track_count,
    decode_class,
    decode_note,
    measure,
    present_at,
)

DRUM_CHANNEL = 9
META_KEEP = ("set_tempo", "time_signature", "key_signature")
VIEWER_TEMPLATE = Path(__file__).with_name("viewer_v1.html")


def _leaves(corpus: Corpus, piece: Piece, step: int) -> list[tuple[int, int, int, int, int, int]]:
    """(onset_ticks, dur_ticks, pitch, instrument, track, velocity) of every note present after ``step``."""
    out = []
    for n in piece.table.notes.values():
        if not present_at(n, step):
            continue
        if n.merge_class >= 0:
            out += [(on, dur, p, inst, tr, n.vel) for on, dur, p, inst, tr in decode_note(corpus, n)]
        else:
            out.append((n.onset_ticks, n.dur_ticks, n.pitch, n.instrument, n.track_numbers[0], n.vel))
    return out


def _conductor(original: mido.MidiFile) -> mido.MidiTrack:
    events = []
    for track in original.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta and msg.type in META_KEEP:
                events.append((tick, msg))
    events.sort(key=lambda e: e[0])
    return _to_track(events, "conductor")


def _to_track(events: list[tuple[int, mido.Message]], name: str) -> mido.MidiTrack:
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    last = 0
    for tick, msg in events:
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _note_track(notes, name: str, channel: int, program: int) -> mido.MidiTrack:
    events: list[tuple[int, int, mido.Message]] = [(0, 0, mido.Message("program_change", channel=channel,
                                                                       program=program))]  # fmt: skip
    for on, dur, pitch, _, _, vel in notes:
        events.append((on, 2, mido.Message("note_on", channel=channel, note=pitch, velocity=max(1, min(127, vel)))))
        events.append((on + max(dur, 1), 1, mido.Message("note_off", channel=channel, note=pitch, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))  # note-offs before note-ons on the same tick
    return _to_track([(t, m) for t, _, m in events], name)


def _reduced_tracks(corpus: Corpus, piece: Piece, step: int) -> dict[int, list[mido.MidiTrack]]:
    """Original track number -> its reduced track(s), one per tax-20 instrument it contains."""
    by = defaultdict(list)
    for leaf in _leaves(corpus, piece, step):
        by[(leaf[4], leaf[3])].append(leaf)
    out: dict[int, list[mido.MidiTrack]] = defaultdict(list)
    n_instruments = Counter(track for track, _ in by)
    channel = 0
    for (track, inst), notes in sorted(by.items()):
        if inst == UNPITCHED:
            ch = DRUM_CHANNEL
        else:
            ch = channel
            channel = (channel + 1) % 16
            if channel == DRUM_CHANNEL:
                channel += 1
        name = piece.track_names[track] if track < len(piece.track_names) else f"track {track}"
        label = f"red: {name}" if n_instruments[track] == 1 else f"red: {name} ({TAX20_NAMES[inst]})"
        out[track].append(_note_track(sorted(notes), label, ch, TAX20_GM[inst]))
    return out


def write_reduced_midi(corpus: Corpus, piece: Piece, path: Path, step: int = 7) -> None:
    original = mido.MidiFile(str(piece.path))
    out = mido.MidiFile(type=1, ticks_per_beat=piece.ticks_per_beat)
    out.tracks.append(_conductor(original))
    for _, tracks in sorted(_reduced_tracks(corpus, piece, step).items()):
        out.tracks.extend(tracks)
    out.save(str(path))


def write_side_by_side_midi(corpus: Corpus, piece: Piece, path: Path, step: int = 7) -> None:
    """Original tracks (verbatim, renamed "orig: ...") each followed by their reduced track(s)."""
    original = mido.MidiFile(str(piece.path))
    out = mido.MidiFile(type=1, ticks_per_beat=original.ticks_per_beat)
    reduced = _reduced_tracks(corpus, piece, step)
    note_track = 0
    for track in original.tracks:
        has_notes = any(m.type == "note_on" and m.velocity > 0 for m in track)
        copy = mido.MidiTrack(m.copy() for m in track if m.type != "track_name")
        name = piece.track_names[note_track] if has_notes and note_track < len(piece.track_names) else "conductor"
        copy.insert(0, mido.MetaMessage("track_name", name=f"orig: {name}" if has_notes else name, time=0))
        out.tracks.append(copy)
        if has_notes:
            out.tracks.extend(reduced.get(note_track, []))
            note_track += 1
    out.save(str(path))


ROW = 10  # values per note in a viewer data file (see write_viewer_data)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def write_viewer_data(corpus: Corpus, piece: Piece, piece_id: int, data_dir: Path) -> dict:
    """One JS file with every note the piece ever had (and when it was present). Returns its index entry.

    Each note is ROW numbers: onset ms, duration ms, pitch, instrument, track, created step, replaced step (99 =
    never), how many notes it replaced, merged flag, merge class id (-1 for plain notes). The returned entry's
    "usage" ({class id: [notes of that class made in this piece, first onset ms]}) feeds write_vocabulary.
    """
    made_from = {}
    for r in piece.table.replacements:
        for i in r.new_ids:
            made_from[i] = len(r.old_ids)
    rows, leaves = [], []
    usage: dict[int, list[int]] = {}
    for n in sorted(piece.table.notes.values(), key=lambda n: (n.onset, n.pitch)):
        k = len(rows) // ROW
        rows += [_ms(n.onset), max(1, _ms(n.dur)), n.pitch, n.instrument, n.track_numbers[0], n.created_step,
                 n.replaced_step if n.replaced_step != NO_STEP else 99, made_from.get(n.note_id, 0),
                 1 if n.merge_class >= 0 else 0, n.merge_class]  # fmt: skip
        if n.merge_class >= 0:
            u = usage.setdefault(n.merge_class, [0, _ms(n.onset)])
            u[0] += 1
            for on, dur, pitch, inst, track in decode_note(corpus, n):
                start = piece.seconds_at(on)
                leaves += [k, _ms(start), max(1, _ms(piece.seconds_at(on + dur) - start)), pitch, inst, track]
    ignored = []
    for onset, dur, pitch, track, gm in piece.ignored:
        ignored += [_ms(onset), max(1, _ms(dur)), pitch, track, gm]
    row = measure(piece)
    payload = {
        "id": piece_id, "name": piece.name, "tracks": piece.track_names, "instruments": list(TAX20_NAMES),
        "steps": list(MEASURE_STEPS), "notes": rows, "leaves": leaves, "ignored": ignored,
        "duration_ms": max((rows[i] + rows[i + 1] for i in range(0, len(rows), ROW)), default=0),
        "stats": {k: v for k, v in row.items() if k.startswith("step")},
    }  # fmt: skip
    (data_dir / f"p{piece_id}.js").write_text("window.__loadPiece(" + json.dumps(payload, separators=(",", ":")) + ");")
    return {"id": piece_id, "name": piece.name, "n_tracks": piece.n_tracks, "duration_s": payload["duration_ms"] / 1000,
            **{f"q{s}": row[f"step{s}_note_quads"] for s in MEASURE_STEPS},
            "mean0": row["step0_mean"], "mean7": row["step7_mean"], "usage": usage}  # fmt: skip


def write_viewer(index: list[dict], out_dir: Path) -> None:
    slim = [{k: v for k, v in e.items() if k != "usage"} for e in index]
    (out_dir / "data" / "index.js").write_text("window.__index = " + json.dumps(slim, separators=(",", ":")) + ";")
    (out_dir / "viewer.html").write_text(VIEWER_TEMPLATE.read_text())


def _class_name(cls: int) -> str:
    return TAX20_NAMES[cls] if cls < 100 else f"#{cls}"


def write_vocabulary(corpus: Corpus, index: list[dict], data_dir: Path, top_pieces: int = 30,
                     max_patterns: int = 5000) -> None:  # fmt: skip
    """data/vocab.js: every merge class with its definition, BPE statistics, decoded shape and where it is used."""
    log = {e["merge_class"]: e for e in corpus.bpe_log}
    tokens = defaultdict(int)
    for cls, _pitch in corpus.merges.token_of:
        tokens[cls] += 1
    used_in: dict[int, list[list[int]]] = defaultdict(list)
    for entry in index:
        for cls, (count, first_ms) in entry.get("usage", {}).items():
            used_in[int(cls)].append([entry["id"], count, first_ms])
    classes = []
    for cls, (kind, a, b, gap, sub12) in sorted(corpus.class_defs.items()):
        e = log.get(cls, {})
        width = 1200  # draw the shape over a 1-beat span at 1200 ticks, starting at relative pitch 0
        shape = [[on, dur, pitch, inst, track] for on, dur, pitch, inst, track in
                 decode_class(corpus, cls, 0, width, 0, tuple(range(class_track_count(corpus, cls))))]  # fmt: skip
        where = sorted(used_in.get(cls, []), key=lambda u: -u[1])
        desc = f"{_class_name(a)} + {_class_name(b)}, {gap:+d} semitones"
        desc += f", second note at {sub12}/12" if kind == "w" else " (simultaneous, different tracks)"
        classes.append({
            "id": cls, "step": 6 if kind == "w" else 7, "kind": kind, "a": a, "b": b, "gap": gap,
            "sub12": sub12 if kind == "w" else None, "desc": desc, "iteration": e.get("iteration"),
            "instances": e.get("instances", 0), "merged": e.get("merged", 0), "tokens": tokens.get(cls, 0),
            "notes": len(shape), "shape": shape, "pieces": len(where), "used_in": where[:top_pieces],
        })  # fmt: skip
    patterns = []
    for e in sorted(corpus.patterns.entries.values(), key=lambda e: -e.count)[:max_patterns]:
        patterns.append({"order": list(e.pitch_order), "gaps": list(e.rounded_gaps), "count": e.count,
                         "example": [[float(on), pitch] for on, pitch in e.example]})  # fmt: skip
    payload = {"instruments": list(TAX20_NAMES), "classes": classes, "patterns": patterns,
               "n_patterns": len(corpus.patterns.entries)}  # fmt: skip
    (data_dir / "vocab.js").write_text("window.__vocab = " + json.dumps(payload, separators=(",", ":")) + ";")
