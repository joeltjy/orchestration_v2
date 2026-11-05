"""Tokenize a SymphonyNet orchestral MIDI into one AMT note sequence per track.

Each track becomes a list of ``Note`` objects whose four fields are AMT
vocabulary token ids (with velocity):

* ``onset    = TIME_OFFSET + onset_bins``          (10 ms bins)
* ``dur      = DUR_OFFSET + duration_bins``        (clipped to ``MAX_DUR - 1``)
* ``pitch    = NOTE_OFFSET + 128 * instr + pitch`` (AMT note token; instr 128 = drums)
* ``velocity = VELOCITY_OFFSET + velocity``

``midi_to_raw_notes`` gives the same notes before tokenization, as ``RawNote`` objects that keep both seconds and
MIDI ticks (plus the file's ticks per beat), so callers that need beat positions can compute them exactly.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import mido

# AMT vocabulary (matches the vendored ``amt`` package in the orchestration repo).
TIME_RESOLUTION = 100  # bins per second
MAX_TIME = TIME_RESOLUTION * 100
MAX_DUR = TIME_RESOLUTION * 10
MAX_PITCH = 128
MAX_INSTR = 129
MAX_NOTE = MAX_PITCH * MAX_INSTR
MAX_VELOCITY = 127
DRUM_INSTR = 128

TIME_OFFSET = 0
DUR_OFFSET = TIME_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
CONTROL_OFFSET = NOTE_OFFSET + MAX_NOTE + 1
SPECIAL_OFFSET = CONTROL_OFFSET + MAX_TIME + MAX_DUR + MAX_NOTE
VELOCITY_OFFSET = SPECIAL_OFFSET + 3

DEFAULT_DURATION = TIME_RESOLUTION // 4  # for notes that are never closed



@dataclass(frozen=True, slots=True)
class Note:
    """One AMT note event; every field is a token id in the AMT vocabulary."""

    onset: int
    dur: int
    pitch: int
    velocity: int

    def tokens(self) -> list[int]:
        """Flatten to AMT's ``(time, dur, note, velocity)`` token order."""
        return [self.onset, self.dur, self.pitch, self.velocity]


@dataclass(frozen=True, slots=True)
class RawNote:
    """One note before tokenization, timed both in seconds and in MIDI ticks.

    ``dur`` / ``dur_ticks`` are None for a note that is never closed by a note-off.
    """

    onset: float  # seconds
    dur: float | None  # seconds
    onset_ticks: int
    dur_ticks: int | None
    pitch: int
    instrument: int  # GM program 0-127 as read by ~/orchestration (drum channel -> 116, timpani tracks -> 47)
    velocity: int

    def onset_beats(self, ticks_per_beat: int) -> Fraction:
        """Onset in MIDI beats, exactly."""
        return Fraction(self.onset_ticks, ticks_per_beat)

    def dur_beats(self, ticks_per_beat: int) -> Fraction | None:
        """Duration in MIDI beats, exactly (None if the note is never closed)."""
        return None if self.dur_ticks is None else Fraction(self.dur_ticks, ticks_per_beat)


def _timed_messages(midi: mido.MidiFile) -> list[tuple[float, int, int, Any]]:
    """Return ``(seconds, tick, track_index, message)`` for every message, in time order.

    Tempo changes may live on any track (usually track 0), so we merge all
    tracks to build one tempo map while remembering which track each message
    came from.
    """
    tagged: list[tuple[int, int, Any]] = []  # mido sets message attributes dynamically
    for track_idx, track in enumerate(midi.tracks):
        tick = 0
        for msg in track:
            tick += msg.time
            tagged.append((tick, track_idx, msg))
    tagged.sort(key=lambda x: x[0])  # stable: keeps per-track order at equal ticks

    tempo = 500_000  # default 120 bpm
    seconds = 0.0
    last_tick = 0
    out = []
    for tick, track_idx, msg in tagged:
        seconds += mido.tick2second(tick - last_tick, midi.ticks_per_beat, tempo)
        last_tick = tick
        if msg.type == "set_tempo":
            tempo = msg.tempo
        out.append((seconds, tick, track_idx, msg))
    return out


def _orchestration_reductions():
    """~/orchestration's ``orchestration.reduction.reductions`` (path overridable with ORCHESTRATION_SRC)."""
    src = str(Path(os.environ.get("ORCHESTRATION_SRC", Path.home() / "orchestration" / "src")))
    if src not in sys.path:
        sys.path.insert(0, src)
    from orchestration.reduction import reductions  # noqa: PLC0415  # ty: ignore[unresolved-import]

    return reductions


def midi_to_raw_notes(midi_path: str | Path) -> tuple[int, list[list[RawNote]]]:
    """Read ``midi_path`` into one ``RawNote`` list per track that contains notes.

    The notes are exactly those of the first step of ~/orchestration's reduction:
    ``orchestration.reduction.reductions.create_note_representation(midi_path)``, i.e.
    ``utils.midi.extract_orchestra_notes`` followed by ``apply_program_overrides``. That means:

    * program state is kept per track (one line per MIDI track), never one program per channel across the file
      (the old merge-tracks reader inherited from AMT, which gave every track on a shared channel the program of
      whichever track set it last; fixed in ~/orchestration commit fd8b37f);
    * the staff name overrides ``program_change`` for winds, brass, timpani and voices when
      ``instrument_names.resolve_track_program`` accepts it (piccolo staves -> GM 72);
    * drum-channel notes: timpani-named tracks keep pitches 36-57 as timpani (GM 47) and drop the rest; all other
      drum-channel notes become GM 116 (unpitched percussion);
    * per-file program overrides (``PROGRAM_OVERRIDE_BY_MIDI_MARKER``) are applied;
    * notes never closed by a note-off, and zero-length notes, are dropped.

    Returns ``(ticks_per_beat, tracks)``: tracks are in MIDI track order (every track with a note-on, so the list
    lines up with the track names; a track whose notes were all dropped gives an empty list), and each track's notes
    are sorted by onset, then pitch. ``instrument`` is the GM program after all of the above.
    """
    midi = mido.MidiFile(str(midi_path))
    if midi.type == 2:
        raise ValueError("Type-2 (asynchronous) MIDI files are not supported")
    note_tracks = [k for k, t in enumerate(midi.tracks) if any(m.type == "note_on" and m.velocity > 0 for m in t)]
    by_line: dict[int, list[RawNote]] = defaultdict(list)
    for n in _orchestration_reductions().create_note_representation(Path(midi_path)):
        by_line[n.line].append(RawNote(n.onset_s, n.offset_s - n.onset_s, n.start_tick, n.end_tick - n.start_tick,
                                       n.pitch, n.program, n.velocity))  # fmt: skip
    tracks = [sorted(by_line.get(k, []), key=lambda r: (r.onset, r.pitch)) for k in note_tracks]
    return midi.ticks_per_beat, tracks


def midi_to_track_events(midi_path: str | Path) -> list[list[Note]]:
    """Tokenize ``midi_path`` into one AMT note sequence per non-empty track.

    Returns a list (one entry per MIDI track that contains notes, in track
    order) of ``Note`` lists sorted by onset.
    """
    _, tracks = midi_to_raw_notes(midi_path)
    sequences = []
    for raw_notes in tracks:
        events = []
        for n in raw_notes:
            onset = round(TIME_RESOLUTION * n.onset)
            dur = DEFAULT_DURATION if n.dur is None else round(TIME_RESOLUTION * n.dur)
            events.append(
                Note(
                    onset=TIME_OFFSET + onset,
                    dur=DUR_OFFSET + min(dur, MAX_DUR - 1),
                    pitch=NOTE_OFFSET + MAX_PITCH * n.instrument + n.pitch,
                    velocity=VELOCITY_OFFSET + min(n.velocity, MAX_VELOCITY),
                )
            )
        sequences.append(events)
    return sequences


def chunk_track_events(sequences: list[list[Note]], chunk_seconds: float = 10.0) -> list[list[list[Note]]]:
    """Split per-track event sequences into fixed-length time chunks.

    Each note goes to the chunk containing its onset (keeping its full
    duration), and onsets are re-expressed relative to the chunk start so they
    stay inside AMT's time vocabulary. Chunks with no notes are kept as empty
    lists so the timeline has no gaps.

    Returns ``chunks[chunk_idx][track_idx]`` -> list of ``Note``.
    """
    chunk_bins = round(TIME_RESOLUTION * chunk_seconds)
    if chunk_bins >= MAX_TIME:
        raise ValueError(f"chunk_seconds must be < {MAX_TIME / TIME_RESOLUTION} s")

    last_onset = max((note.onset - TIME_OFFSET for seq in sequences for note in seq), default=-1)
    n_chunks = last_onset // chunk_bins + 1
    chunks: list[list[list[Note]]] = [[[] for _ in sequences] for _ in range(n_chunks)]
    for track_idx, seq in enumerate(sequences):
        for note in seq:
            chunk_idx, rel_onset = divmod(note.onset - TIME_OFFSET, chunk_bins)
            chunks[chunk_idx][track_idx].append(replace(note, onset=TIME_OFFSET + rel_onset))
    return chunks


if __name__ == "__main__":
    import sys

    for i, seq in enumerate(midi_to_track_events(sys.argv[1])):
        instr = (seq[0].pitch - NOTE_OFFSET) // MAX_PITCH
        print(f"track {i}: instr={instr} notes={len(seq)} first={seq[0]}")
