"""Tests for `orchestration_v2.symphonynet`."""

from fractions import Fraction

import mido

from orchestration_v2.symphonynet import midi_to_raw_notes, midi_to_track_events


def _write_midi(path):
    """Two tracks at 480 ticks per beat: a tempo change after beat 1, a violin line, and an unclosed drum hit."""
    m = mido.MidiFile(type=1, ticks_per_beat=480)
    tempo = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),  # 120 bpm: a beat is 0.5 s
            mido.MetaMessage("set_tempo", tempo=1_000_000, time=480),  # 60 bpm from beat 1: a beat is 1 s
        ]
    )
    violin = mido.MidiTrack(
        [
            mido.Message("program_change", channel=0, program=40, time=0),
            mido.Message("note_on", channel=0, note=60, velocity=80, time=240),  # beat 1/2
            mido.Message("note_off", channel=0, note=60, velocity=0, time=480),  # held 1 beat
            mido.Message("note_on", channel=0, note=64, velocity=70, time=0),  # beat 3/2
        ]
    )
    violin.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=240))  # held 1/2 beat
    drums = mido.MidiTrack(
        [
            mido.Message("note_on", channel=9, note=38, velocity=100, time=960),  # beat 2, 1/4 beat
            mido.Message("note_off", channel=9, note=38, velocity=0, time=120),
            mido.Message("note_on", channel=9, note=36, velocity=100, time=0),  # never closed
        ]
    )
    m.tracks.extend([tempo, violin, drums])
    m.save(str(path))


def test_raw_notes_keep_ticks_and_seconds(tmp_path):
    path = tmp_path / "t.mid"
    _write_midi(path)
    tpb, tracks = midi_to_raw_notes(path)
    assert tpb == 480
    assert len(tracks) == 2  # the tempo-only track has no notes

    first, second = tracks[0]
    assert (first.onset_ticks, first.dur_ticks, first.pitch, first.instrument) == (240, 480, 60, 40)
    assert first.onset_beats(tpb) == Fraction(1, 2) and first.dur_beats(tpb) == 1
    # seconds follow the tempo map: 0.25 s to the onset; the beat held spans 0.25 s at 120 bpm + 0.5 s at 60 bpm
    assert first.dur is not None and second.dur is not None
    assert abs(first.onset - 0.25) < 1e-9 and abs(first.dur - 0.75) < 1e-9
    assert second.onset_beats(tpb) == Fraction(3, 2) and second.dur_beats(tpb) == Fraction(1, 2)
    # beat 1 is reached at 0.5 s; the next half beat is at 60 bpm, so 0.5 s more
    assert abs(second.onset - 1.0) < 1e-9 and abs(second.dur - 0.5) < 1e-9

    (drum,) = tracks[1]  # the unclosed hit is dropped, as in ~/orchestration's reader
    assert drum.instrument == 116 and drum.onset_beats(tpb) == 2 and abs(drum.onset - 1.5) < 1e-9
    assert drum.dur_ticks == 120


def test_tracks_sharing_a_channel_keep_their_own_programs(tmp_path):
    """Regression: the per-channel reader gave both tracks the program set last on channel 0."""
    m = mido.MidiFile(type=1, ticks_per_beat=480)
    for program, pitch in ((40, 76), (42, 48)):  # violin, then cello, both on channel 0
        m.tracks.append(mido.MidiTrack([
            mido.Message("program_change", channel=0, program=program, time=0),
            mido.Message("note_on", channel=0, note=pitch, velocity=80, time=480),
            mido.Message("note_off", channel=0, note=pitch, velocity=0, time=480),
        ]))  # fmt: skip
    path = tmp_path / "shared.mid"
    m.save(str(path))
    _, tracks = midi_to_raw_notes(path)
    assert [[n.instrument for n in t] for t in tracks] == [[40], [42]]


def test_track_events_match_raw_notes(tmp_path):
    path = tmp_path / "t.mid"
    _write_midi(path)
    _, tracks = midi_to_raw_notes(path)
    events = midi_to_track_events(path)
    assert [len(t) for t in events] == [len(t) for t in tracks]
