"""Tests for the step rules in `orchestration_v2.reduction_v1` on small hand-made voices."""

from orchestration_v2.reduction_v1 import (
    Corpus,
    Note,
    PatternTable,
    ReductionTable,
    decode_note,
    remove_repeated_notes,
    remove_repeated_patterns,
)

TPB = 480


def _voice(spec, table, instrument=0):
    """spec: (onset_beats, dur_beats, pitch) triples -> present notes in one voice (track 0, voice 0)."""
    notes = []
    for on, dur, pitch in spec:
        on_t, dur_t = round(on * TPB), round(dur * TPB)
        n = Note(table.new_id(), on / 2, dur / 2, on_t, dur_t, TPB, pitch, instrument, 128 * instrument + pitch, 80,
                 (0,), (0,), 0)  # fmt: skip
        notes.append(table.add(n))
    return notes


def test_repeated_notes_collapse_evenly_spaced_chain():
    table = ReductionTable()
    voice = _voice([(i / 2, 0.4, 60) for i in range(6)], table)  # six 8th-note C4s
    out = remove_repeated_notes(voice, table)
    assert len(out) == 1
    (m,) = out
    assert (m.onset_ticks, m.offset_ticks, m.pitch, m.created_step) == (0, round(2.9 * TPB), 60, 4)
    assert all(not n.is_present_in_reduction and n.replaced_step == 4 for n in voice)


def test_repeated_notes_need_four_and_equal_gaps():
    table = ReductionTable()
    three = _voice([(0, 0.4, 60), (0.5, 0.4, 60), (1, 0.4, 60)], table)
    assert len(remove_repeated_notes(three, table)) == 3
    table = ReductionTable()
    uneven = _voice([(0, 0.4, 60), (0.5, 0.4, 60), (1.5, 0.4, 60), (2, 0.4, 60)], table)  # gaps 1/2, 1, 1/2
    assert len(remove_repeated_notes(uneven, table)) == 4


def test_repeated_pattern_becomes_one_chord_per_repetition():
    table, patterns = ReductionTable(), PatternTable()
    figure = [60, 67, 64, 67]  # C G E G, 16ths
    voice = _voice([(i / 4, 0.25, figure[i % 4]) for i in range(12)], table)
    out = remove_repeated_patterns(voice, table, patterns)
    chords = [n for n in out if n.created_step == 5]
    assert len(chords) == 9 and len(out) == 9  # 3 repetitions x 3 distinct pitches
    assert sorted({n.onset_ticks for n in chords}) == [0, TPB, 2 * TPB]
    assert {n.pitch for n in chords} == {60, 64, 67}
    (entry,) = patterns.entries.values()
    assert entry.pitch_order == (1, 3, 2, 3) and entry.count == 3


def test_strictly_increasing_figures_are_not_patterns():
    table, patterns = ReductionTable(), PatternTable()
    voice = _voice([(i / 4, 0.25, 60 + 2 * (i % 3)) for i in range(9)], table)  # C D E, C D E, C D E
    assert len(remove_repeated_patterns(voice, table, patterns)) == 9
    assert not patterns.entries


def test_decode_within_merge():
    corpus = Corpus([])
    cid = corpus.class_id(("w", 0, 0, 7, 6))  # violin, then violin a 5th up, starting halfway through
    m = Note(0, 0.0, 1.0, 0, 2 * TPB, TPB, 60, 0, 9999, 80, (3,), (0,), 0, merge_class=cid)
    assert decode_note(corpus, m) == [(0, TPB, 60, 0, 3), (TPB, TPB, 67, 0, 3)]


def test_within_instances_never_reuse_a_note():
    from orchestration_v2.reduction_v1 import _within_instances  # noqa: PLC0415

    table = ReductionTable()
    # overlapping legato notes: each note is followed by the next two (same class), so a naive check would reuse
    voice = _voice([(i / 2, 1.0, 60) for i in range(8)], table)
    inst = _within_instances(voice)
    used = [x for _, i, j in inst for x in (i, j)]
    assert len(used) == len(set(used))
