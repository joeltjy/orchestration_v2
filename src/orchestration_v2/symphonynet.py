"""Tokenize a SymphonyNet orchestral MIDI into one AMT note sequence per track.

Each track becomes a list of ``Note`` objects whose four fields are AMT
vocabulary token ids (with velocity):

* ``onset    = TIME_OFFSET + onset_bins``          (10 ms bins)
* ``dur      = DUR_OFFSET + duration_bins``        (clipped to ``MAX_DUR - 1``)
* ``pitch    = NOTE_OFFSET + 128 * instr + pitch`` (AMT note token; instr 128 = drums)
* ``velocity = VELOCITY_OFFSET + velocity``
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
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


def _timed_messages(midi: mido.MidiFile) -> list[tuple[float, int, Any]]:
    """Return ``(seconds, track_index, message)`` for every message, in time order.

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
        out.append((seconds, track_idx, msg))
    return out


def midi_to_track_events(midi_path: str | Path) -> list[list[Note]]:
    """Tokenize ``midi_path`` into one AMT note sequence per non-empty track.

    Returns a list (one entry per MIDI track that contains notes, in track
    order) of ``Note`` lists sorted by onset.
    """
    midi = mido.MidiFile(str(midi_path))
    if midi.type == 2:
        raise ValueError("Type-2 (asynchronous) MIDI files are not supported")

    programs: dict[int, int] = defaultdict(int)  # channel -> program
    # track -> list of [onset_s, duration_s | None, pitch, instr, velocity]
    notes: dict[int, list[list]] = defaultdict(list)
    open_notes: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for t, track_idx, msg in _timed_messages(midi):
        if msg.type == "program_change":
            programs[msg.channel] = msg.program
        elif msg.type in ("note_on", "note_off"):
            instr = DRUM_INSTR if msg.channel == 9 else programs[msg.channel]
            key = (track_idx, msg.channel, msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                open_notes[key].append(len(notes[track_idx]))
                notes[track_idx].append([t, None, msg.note, instr, msg.velocity])
            elif open_notes[key]:
                note = notes[track_idx][open_notes[key].pop(0)]  # FIFO, like AMT
                note[1] = t - note[0]

    sequences = []
    for track_idx in sorted(notes):
        events = []
        for onset_s, dur_s, pitch, instr, velocity in sorted(notes[track_idx], key=lambda n: (n[0], n[2])):
            onset = round(TIME_RESOLUTION * onset_s)
            dur = DEFAULT_DURATION if dur_s is None else round(TIME_RESOLUTION * dur_s)
            events.append(
                Note(
                    onset=TIME_OFFSET + onset,
                    dur=DUR_OFFSET + min(dur, MAX_DUR - 1),
                    pitch=NOTE_OFFSET + MAX_PITCH * instr + pitch,
                    velocity=VELOCITY_OFFSET + min(velocity, MAX_VELOCITY),
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
