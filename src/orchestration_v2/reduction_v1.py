"""Reduce an orchestral corpus to fewer note quads, step by step, without losing track of what was replaced.

Overview
--------
The pipeline (``tokenize_corpus``) runs these steps in order:

0. Read every piece with ``symphonynet.midi_to_raw_notes`` into ``Note`` objects and build the ``ReductionTable``.
1. Sort each track's notes by onset, breaking ties by pitch.
2. Separate each track into voices, the same way as ``polyphony_subsegments.py`` in ``~/orchestration``.
3. Remove same-pitch duplicates and mark runs (onset snapping is skipped: pieces are on the beat grid).
4. Collapse repeated notes within a voice (``remove_repeated_notes``).
5. Collapse repeated patterns within a voice (``remove_repeated_patterns``).
6. BPE-merge notes that follow each other within a voice (``bpe_within_instruments``).
7. BPE-merge simultaneous notes across tracks (``bpe_between_instruments``).

Notes and tracks
----------------
Each track in a piece gets a unique integer id ``0 .. N-1``. Tracks must stay separate throughout: never collapse
several tracks (or lines) into one channel. A previous version made this mistake and scrambled the parts.

A note carries ``(note_id, onset, dur, onset_beats, dur_beats, note_token, vel, track_numbers, voice_ids,
is_present_in_reduction)`` (see the ``Note`` dataclass below; it also keeps ``primary_voice`` and ``run_id``).

* ``note_id`` is unique across the piece and never reused, so the ReductionTable can refer to notes by id.
* ``note_token`` says what the note stands for. It starts as just (instrument, pitch), encoded as
  ``128 * instrument + pitch`` where ``instrument`` is a tax-20 index ``0 .. 19`` (see step 0), so there are at most
  20 x 128 = 2,560 base tokens. After pattern removal and BPE it can stand for several notes.
* ``track_numbers`` is a set holding a single track for every note except cross-track merges from
  ``bpe_between_instruments``, which keep the set of track numbers they came from.
* ``voice_ids`` are the voices the note belongs to within its track, assigned in step 2. Voice ids are unique
  within a piece. Usually there is one, but a fill-in copy (see step 2) puts a note in several voices until the end
  of step 5.

Time is stored twice on every note:

* ``onset`` / ``dur`` in seconds (from the tempo map), used for the per-step measurement and the final AMT tokens;
* ``onset_beats`` / ``dur_beats`` in MIDI beats (``ticks / ticks_per_beat``, kept as exact fractions), used by
  every rule in steps 2-7 unless a rule says seconds. Every piece read in is on the beat grid (see step 0), so a
  note 3/8 of a beat after beat 27 has ``onset_beats = 27 + 3/8`` exactly, whatever the tempo map does.

A few tracks are not exactly on the grid (small jitter of about +-2 ticks, or 5-15% of notes slightly off), so
any rule that compares beat positions for equality uses a tolerance ``BEAT_TOL = 1/48 beat`` (about 10 ms at
120 bpm; it covers +-2 ticks at 120 ticks per beat): "equal" below means "within BEAT_TOL".

Both stay raw through every step. They are converted to AMT tokens only at the very end, so long merged notes are
not clipped by AMT's 10 s duration cap (``MAX_DUR``). At that point each tax-20 instrument is written back as its
own GM program (e.g. index 0 -> GM 40 violin) to form AMT note tokens.

Terminology
-----------
* **Note quad**: one ``(onset, dur, note_token, vel)`` in the AMT tokenization.
* **Merged note quad**: a note quad whose ``note_token`` stands for more than one original note (from repeated
  patterns or BPE).
* **Consecutive-onset notes**: notes that are adjacent in a voice's sorted list (sorted in step 1, split into
  voices in step 2).

ReductionTable: nothing is deleted
----------------------------------
Whenever a step replaces notes, it does not delete the old ones. It sets ``is_present_in_reduction = False`` on
the old notes, creates the new notes with ``is_present_in_reduction = True``, and records both in the
``ReductionTable``, together with which old note ids each new note replaced. The table therefore holds every note
that ever existed and how each one came about, and the current reduction is the set of notes whose flag is
``True``.

Measuring each step
-------------------
Measure how many note quads remain after step 0 (the baseline) and after steps 4, 5, 6 and 7, i.e. the steps
that replace notes (steps 1-3 change no note count worth measuring). This stays in seconds (``onset``), because it
measures token density in real time for a model's context window. For every piece, at each of those 5 points:

1. Split the current reduction (present notes only) into 10-second chunks, assigning each note quad to the chunk
   its onset falls in.
2. For each chunk, compute note quads per second: the number of note quads in the chunk divided by 10.
3. Record the min, max, mean and std of those per-chunk values for the piece, plus the start and end seconds of
   the min chunk and of the max chunk (as in ``notes_per_chunk.csv``). Mean and std describe the whole piece, so
   they have no example chunk.

Write everything to ``note_quads_per_chunk_v1.csv``. With ``n = 5`` measuring points, each piece gets ``4n = 20``
statistics and ``2n = 10`` example chunks.
"""

from __future__ import annotations

import bisect
import csv
import heapq
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import mido

from orchestration_v2.symphonynet import DRUM_INSTR, midi_to_raw_notes

# --- Constants ----------------------------------------------------------------------------------------------------

BEAT_TOL = Fraction(1, 48)  # beats; "equal" beat positions differ by at most this
SILENCE_SPLIT_S = 3.0  # step 2a
RUN_MAX_BEATS = Fraction(1, 4)  # step 3 runs
FOLLOW_TOL_BEATS = Fraction(1, 8)  # step 6 "follows"
BETWEEN_DUR_TOL_BEATS = Fraction(1, 16)  # step 7 durations
UNCLOSED_DUR_BEATS = Fraction(1, 2)  # step 0
CHUNK_SECONDS = 10.0
MEASURE_STEPS = (0, 4, 5, 6, 7)
NO_STEP = 99  # replaced_step for notes still present at the end

N_INSTRUMENTS = 20
BASE_TOKENS = N_INSTRUMENTS * 128
FIRST_MERGED_CLASS = 100  # class ids below this are base classes (= tax-20 index)
UNPITCHED = 17

# tax-20 index -> its own GM program, in LINE_STREAM_INSTRUMENTS_20 order (~/orchestration/src/common_instruments.py)
TAX20_GM = (40, 41, 42, 43, 44, 45, 48, 73, 68, 71, 70, 56, 60, 57, 58, 0, 47, 116, 25, 52)
TAX20_NAMES = (
    "violin", "viola", "cello", "contrabass", "tremolo strings", "pizzicato strings", "string ensemble", "flute",
    "oboe", "clarinet", "bassoon", "trumpet", "french horn", "trombone", "tuba", "piano", "timpani", "unpitched",
    "guitar", "choir",
)  # fmt: skip
# GM programs that remap_line_stream_program (tax-20) sends to each class, minus the ones step 0 ignores: harp 46,
# piccolo 72, and every program that only reaches unpitched (116) as a fallback.
_TAX20_MEMBERS = {
    0: (40,), 1: (41,), 2: (42,), 3: (43,), 4: (44,), 5: (45,), 6: (48, 49, 50, 51, 92), 7: (73, 74, 77, 79),
    8: (68, 69), 9: (65, 66, 71), 10: (70,), 11: (56, 59), 12: (60,), 13: (57, 61), 14: (58,),
    15: (0, 6, 9, 12, 13, 16, 19, 20), 16: (47,), 17: (116,), 18: (*range(24, 40), 85, 106), 19: (52, 53, 54),
}  # fmt: skip
GM_TO_TAX20 = {gm: idx for idx, gms in _TAX20_MEMBERS.items() for gm in gms}

PATTERN_MAX_DISTINCT_PITCHES = 3  # step 5: a repeated pattern may use at most this many distinct pitches

# read-only: the voice splitter (step 2); override with the ORCHESTRATION_SRC environment variable
ORCHESTRATION_SRC = Path(os.environ.get("ORCHESTRATION_SRC", Path.home() / "orchestration" / "src"))


def tax20_index(program: int) -> int | None:
    """tax-20 index for a GM program (or DRUM_INSTR); None if step 0 ignores it."""
    if program == DRUM_INSTR:
        return UNPITCHED
    return GM_TO_TAX20.get(program)


# --- Data structures ----------------------------------------------------------------------------------------------

# A merge class id. Ids below FIRST_MERGED_CLASS are base classes (an original note's tax-20 instrument). A merged
# class id refers to Corpus.class_defs[id] = (kind, class of first child, class of second child, pitch_gap, sub12),
# where kind is "w" (within, step 6) or "b" (between, step 7) and sub12 is the subdivision in twelfths (or -1).
MergeClass = int


@dataclass(slots=True, eq=False)
class Note:
    """One note (or merged note quad). Beat positions are kept exactly as ticks; see onset_beats / dur_beats."""

    note_id: int
    onset: float  # seconds
    dur: float  # seconds
    onset_ticks: int
    dur_ticks: int
    ticks_per_beat: int
    pitch: int  # starting pitch for merged notes
    instrument: int  # tax-20 index (first note's, for merged notes)
    note_token: int
    vel: int
    track_numbers: tuple[int, ...]  # one track, except cross-track merges from step 7 (ordered like their notes)
    voice_ids: tuple[int, ...] = ()  # several only for fill-in copies, until the end of step 5
    primary_voice: int = -1  # the voice from step 2d (not a fill-in)
    run_id: int = -1  # set in step 3 for notes in a run
    is_present_in_reduction: bool = True
    merge_class: int = -1  # -1 for plain notes, else a merged class id
    created_step: int = 0
    replaced_step: int = NO_STEP

    @property
    def onset_beats(self) -> Fraction:
        return Fraction(self.onset_ticks, self.ticks_per_beat)

    @property
    def dur_beats(self) -> Fraction:
        return Fraction(self.dur_ticks, self.ticks_per_beat)

    @property
    def offset_ticks(self) -> int:
        return self.onset_ticks + self.dur_ticks

    @property
    def offset(self) -> float:
        return self.onset + self.dur

    @property
    def cls(self) -> int:
        return self.merge_class if self.merge_class >= 0 else self.instrument


@dataclass
class Replacement:
    """One replacement: which notes were replaced by which, and in which step."""

    step: int
    old_ids: tuple[int, ...]
    new_ids: tuple[int, ...]


@dataclass
class ReductionTable:
    """Every note that ever existed in a piece, and how each replaced note came about."""

    notes: dict[int, Note] = field(default_factory=dict)  # by note_id
    replacements: list[Replacement] = field(default_factory=list)
    next_note_id: int = 0

    def add(self, note: Note) -> Note:
        self.notes[note.note_id] = note
        return note

    def new_id(self) -> int:
        self.next_note_id += 1
        return self.next_note_id - 1

    def replace(self, step: int, old: Iterable[Note], new: Iterable[Note]) -> None:
        old, new = list(old), list(new)
        for n in old:
            n.is_present_in_reduction = False
            n.replaced_step = step
        for n in new:
            n.created_step = step
            self.add(n)
        self.replacements.append(Replacement(step, tuple(n.note_id for n in old), tuple(n.note_id for n in new)))


@dataclass
class PatternEntry:
    pitch_order: tuple[int, ...]  # dense ranks
    rounded_gaps: tuple[int, ...]  # relative onset gaps, in twelfths (rounded to 1/12)
    example: list[tuple[Fraction, int]]  # (onset in beats relative to the first note, pitch)
    count: int = 0  # repetitions across the corpus


@dataclass
class PatternTable:
    entries: dict[tuple[tuple[int, ...], tuple[int, ...]], PatternEntry] = field(default_factory=dict)

    def merge(self, other: PatternTable) -> None:
        for key, e in other.entries.items():
            mine = self.entries.get(key)
            if mine is None:
                self.entries[key] = e
            else:
                mine.count += e.count


@dataclass
class MergeEntry:
    token: int
    merge_class: MergeClass
    children: tuple[MergeClass, MergeClass]
    instrument: int | None  # tax-20 index for "within" merges; None for "between"
    start_pitch: int
    pitch_gap: int
    subdivision: Fraction | None  # "within" merges only


@dataclass
class BPEMergeTable:
    entries: dict[int, MergeEntry] = field(default_factory=dict)  # by new note_token
    next_token: int = BASE_TOKENS  # after the 2,560 base tokens
    token_of: dict[tuple[int, int], int] = field(default_factory=dict)  # (class, start pitch) -> token


@dataclass
class Piece:
    name: str
    path: Path
    ticks_per_beat: int
    n_tracks: int
    track_names: list[str]
    table: ReductionTable
    tempo: list[tuple[int, float, int]]  # (tick, seconds at tick, microseconds per beat from there on)
    voices: dict[int, list[Note]] = field(default_factory=dict)  # voice id -> its notes (sorted)
    voice_kind: dict[int, str] = field(default_factory=dict)  # voice id -> "group" / "overflow" / "leftover"
    ignored: list[tuple[float, float, int, int, int]] = field(
        default_factory=list
    )  # (onset s, dur s, pitch, track, GM)
    patterns: PatternTable = field(default_factory=PatternTable)  # this piece's share, merged into the corpus's
    log: list[str] = field(default_factory=list)

    def seconds_at(self, tick: float) -> float:
        i = bisect.bisect_right(self.tempo, (tick, math.inf, 0)) - 1
        t0, s0, us = self.tempo[max(i, 0)]
        return s0 + (tick - t0) * us / 1e6 / self.ticks_per_beat


@dataclass
class Corpus:
    pieces: list[Piece]
    patterns: PatternTable = field(default_factory=PatternTable)
    merges: BPEMergeTable = field(default_factory=BPEMergeTable)
    class_defs: dict[int, tuple[str, int, int, int, int]] = field(default_factory=dict)
    class_ids: dict[tuple[str, int, int, int, int], int] = field(default_factory=dict)
    bpe_log: list[dict] = field(default_factory=list)

    def class_id(self, key: tuple[str, int, int, int, int]) -> int:
        cid = self.class_ids.get(key)
        if cid is None:
            cid = FIRST_MERGED_CLASS + len(self.class_ids)
            self.class_ids[key] = cid
            self.class_defs[cid] = key
        return cid


# --- Pipeline ----------------------------------------------------------------------------------------------


def tokenize_corpus(corpus_dir: Path, outliers_csv: Path, measurements_csv: Path, **options) -> Corpus:
    # Run the full reduction pipeline over every piece in ``corpus_dir``.
    #
    # Step 0 - read notes. Read each piece with symphonynet.midi_to_raw_notes, which returns ticks per beat and
    #   one RawNote list per track, each note timed in seconds and in ticks (onset_beats / dur_beats come from
    #   RawNote.onset_beats(tpb) / dur_beats(tpb)). Build the ReductionTable's Note objects from them.
    #   midi_to_raw_notes returns exactly the notes of the first step of ~/orchestration's reduction
    #   (reductions.create_note_representation: per-track programs, staff-name and per-file program overrides,
    #   drum channel -> 116 / timpani 47). Notes never closed by a note-off are dropped there, so the 1/2-beat
    #   fallback below no longer fires; it is kept only in case the reader changes.
    #   Pieces read in: ~/Desktop/symphonynet_classical, minus the 38 timing outliers listed in
    #   to_analyze/to_analyze.csv (13 played-in files with no beat grid, 25 at 10 or 48 ticks per beat). Every
    #   remaining piece has its notes on the beat grid in ticks (constant per-track offsets were corrected on
    #   2026-11-03/04; see README.md), so beat positions are exact.
    #   Map every note's instrument onto the 20-instrument taxonomy ("tax-20") from ~/orchestration
    #   (LINE_STREAM_INSTRUMENTS_20 and remap_line_stream_program in src/common_instruments.py). The instrument
    #   is stored as the tax-20 index 0..19, in this order:
    #     0 violin (GM 40)       1 viola (41)          2 cello (42)          3 contrabass (43)
    #     4 tremolo str (44)     5 pizzicato str (45)  6 string ens. (48)    7 flute (73)
    #     8 oboe (68)            9 clarinet (71)      10 bassoon (70)       11 trumpet (56)
    #    12 French horn (60)    13 trombone (57)      14 tuba (58)          15 piano (0)
    #    16 timpani (47)        17 unpitched (116)    18 guitar (25)        19 choir (52)
    #   GM programs that tax-20 does not name are folded into one of these where the remap gives a class (e.g.
    #   English horn 69 -> oboe). The drum channel (AMT instrument 128) maps to unpitched.
    #   Ignored for now (their notes are not read in):
    #   * harp (46) and piccolo (72), which are not tax-20 classes, even though the remap would fold them into
    #     string ensemble and flute;
    #   * leftover programs: any GM program other than 116 that the remap only sends to unpitched (116) as a
    #     fallback, e.g. vibraphone 11 or tubular bells 14.
    #
    # Step 1 - sort. Within each track, sort notes by onset, breaking ties by pitch. "Consecutive-onset
    #   notes" from here on means notes that are adjacent in this sorted list (or, once voices exist,
    #   in a voice's sorted list).
    #
    # Step 2 - separate voices. Split each track into voices, following polyphony_subsegments.py in
    #   ~/orchestration (see reduction.md there, "Splitting into voices"):
    #   a. Split the track wherever it has more than 3 s of silence (seconds, on purpose: a 3 s break feels like
    #      a break at any tempo). Voice ids never cross these splits.
    #   b. Sweep each piece left to right, counting how many notes sound at once (k). Notes that only touch at an
    #      endpoint do not overlap. Before counting, temporarily clip legato bleed into the next attack (up to
    #      1/6 beat, pulled back 1/32 beat), then restore the original times afterwards.
    #   c. Group stretches by polyphony, for l = 1 .. max_k: grow a group from unlocked spans with
    #      1 <= k <= l, gluing a later span across a gap of at most 2 beats if everything in the gap
    #      is unlocked, and, if the gap has a higher k than the current span, the gap is no longer than the
    #      current span. Lock the group only if it lasts at least max(1 s, 1 beat) (kept mixed: the 1 s floor
    #      stops very fast passages locking tiny groups); shorter spans are leftover.
    #   d. Label each group's voices top-down: voice 1 is the highest pitch. A note gets a layer only if it is
    #      the highest remaining pitch for its whole duration.
    #   e. Fill-in copies: if a note's voice p is above the bottom voice and nothing lower sounds at its onset,
    #      it is also placed on voices p+1 .. l until the next attack on that voice (or until it ends). Copies
    #      shorter than 1/16 beat are dropped.
    #   f. If two groups with the same voice count have only a gap between them, and the gap is shorter than
    #      half their combined duration, absorb the gap into them and relabel top-down.
    #   g. Notes that cannot be assigned to a group become their own leftover voice.
    #   Leftover voices (from g, and from overcrowded stretches below) can contain chords. Steps 4-6 still run on
    #   them like any other voice; results there are rougher, which is accepted (they are rare, mostly in dense
    #   tutti).
    #   Differences from ~/orchestration:
    #   * There, a stretch with k > max_k anywhere (max_k = 4; 1 for contrabass) is left unsplit, and an earlier
    #     step drops overcrowded stretches entirely. Here nothing is deleted: such a stretch is labelled top-down
    #     as usual, the highest max_k lines become voices 1 .. max_k, and every note below them goes into one
    #     leftover voice for that stretch (which can contain chords).
    #   * Fill-in copies make one note belong to several voices. Copies are kept through pattern detection
    #     (steps 4 and 5). First replacement wins: once any copy of a note has been replaced (by a repeated-note
    #     chain or a pattern chord), the note is no longer available in any other voice, so later matches skip it
    #     there. A note is therefore replaced at most once. After step 5, each note still present keeps only its
    #     primary voice (the one it was labelled with in d, not a fill-in) and the copies are dropped.
    #
    # Step 3 - onsets and runs. Onset snapping is skipped: every piece read in is on the beat grid, so notes
    #   that belong together already start on exactly the same tick, and snapping would only risk merging a
    #   grace note into its main note. (The earlier rule, kept for reference in case off-grid pieces are added
    #   back: within a track, move onsets within +-50 ms of the earliest one onto it, not chained, keeping each
    #   note's offset.)
    #   * If two same-pitch notes in the same track start within BEAT_TOL of each other, keep only the longer one
    #     (if equally long, the first in the file). This is not recorded in the ReductionTable.
    #   Runs (used by step 5, which excludes them from pattern detection): a run is k >= 5 consecutive-onset
    #   notes in the same voice (from step 2) where
    #     * every note has a different onset,
    #     * every gap between consecutive onsets is at most 1/4 beat (16ths or faster), and
    #     * every note lasts at most 1/4 beat.
    #   This is in beats ("written fast"), so a 16th-note scale counts as a run at any tempo. Each note
    #   remembers which run it belongs to, for step 5.
    #
    # Step 4 - within each voice, collapse repeated notes with remove_repeated_notes().
    # Step 5 - within each voice, collapse repeated patterns with remove_repeated_patterns().
    # Step 6 - merge notes within each voice with bpe_within_instruments().
    # Step 7 - merge notes across instruments with bpe_between_instruments().
    #
    # Returns the Corpus (every piece's ReductionTable, plus the shared PatternTable and BPEMergeTable) and writes
    # the per-step measurements to measurements_csv (note_quads_per_chunk_v1.csv). ``options`` (workers,
    # token_budget, min_count, progress, limit) are passed through to the implementation.
    return _impl_tokenize_corpus(corpus_dir, outliers_csv, measurements_csv, **options)


def remove_repeated_notes(voice: list[Note], table: ReductionTable) -> list[Note]:
    # Takes one voice's present notes (sorted, as in step 1) and returns its present notes afterwards. Replaced
    # notes are flagged not present in ``table`` and the new notes are added to it, with a Replacement record.
    #
    # Replace chains of evenly spaced same-pitch notes in one voice with a single long note.
    #
    # Finding chains: for each pair (note i, note j) in the sorted voice with i < j < i + 5, i.e. j is one of
    # the next four notes after i:
    #   * if i and j have different pitches, skip the pair;
    #   * otherwise start a "repeated note chain" n_1 = i, n_2 = j, and keep extending it with later notes of
    #     the same pitch, using the same window: the next chain note must be one of the 4 notes after the chain's
    #     current last note. The chain ends at the first step with no same-pitch note in that window (or when
    #     the gaps stop being equal).
    #
    # A repeated note chain n_1, ..., n_k is a sequence of same-pitch notes whose onset gaps in beats
    #   (onset_beats(n_2) - onset_beats(n_1), ..., onset_beats(n_k) - onset_beats(n_(k-1)))
    # are all equal within BEAT_TOL (largest gap - smallest gap <= 1/48 beat). (In seconds they need not be:
    # repeated 8ths in a ritardando are 1/2 beat apart each, but their gaps in seconds grow.)
    #
    # Consecutive chain notes must be adjacent in time: no note of the same voice may have an onset strictly
    # between onset(n_i) and onset(n_(i+1)). Notes starting at the same time as a chain note (e.g. the rest of a
    # chord) do not break the chain. The 4-note window above still applies on top of this.
    #
    # A chain must contain at least 4 notes (k >= 4). Runs (step 3) are not exempt: fast repeated notes such as
    # tremolos or repeated 16ths are collapsed here too (intended).
    #
    # Replacement: every note in the chain is replaced by one note starting at onset(n_1) and ending at
    # onset(n_k) + duration(n_k) (the same in beats and in seconds). Its velocity is the max velocity in the
    # chain (as in BPE), e.g. 80, 60, 70, 60 -> 80.
    #
    # Target complexity: O(n), or O(n log n) if the voice has to be re-sorted.
    return _impl_remove_repeated_notes(voice, table)


def remove_repeated_patterns(voice: list[Note], table: ReductionTable, patterns: PatternTable) -> list[Note]:
    """Replace a short melodic figure repeated three or more times in a row with sustained chords.

    Takes one voice's present notes and returns its present notes afterwards. Replacements are recorded in
    ``table``; every repeated pattern found is added to (or counted in) the corpus-wide ``patterns``.

    Pattern of a note sequence
        For consecutive notes ``n_1, ..., n_k`` in one voice, the pattern is:

        * the **pitch order**: the dense rank of each note's pitch within the group (lowest distinct pitch = 1;
          notes with the same pitch get the same rank), and
        * the **relative onset gaps**: each onset gap divided by the pattern's duration, both measured in beats
          (``onset_beats``). In beats, an even figure stays even through a ritardando, and a figure played twice as
          slowly (8ths instead of 16ths) still has the same relative gaps.

        A pattern of ``k`` notes has ``k`` onset gaps: the ``k - 1`` gaps between its own notes, plus a final gap
        from ``n_k``'s onset to the onset of the next note in the voice (for any repetition but the last, this is
        the first note of the next repetition). If there is no next note, the final gap runs to the end of ``n_k``
        instead. The duration is the sum of all ``k`` gaps.

        Example: C4, G4, E4, G4 as 16ths (1/4 beat apart), with the next repetition starting 1/4 beat after the
        last G4. The pitch order is (1, 3, 2, 3): both G4s get rank 3, so C4 G4 E4 A4 (1, 3, 2, 4) is a different
        pattern. The gaps are
        (1/4, 1/4, 1/4, 1/4) beats, the duration is 1 beat, and the relative gaps are (1/4, 1/4, 1/4, 1/4).

    What counts as a repeated pattern
        A pattern of length ``3 <= k <= 10`` that occurs at least three times back to back. That is, a run of
        consecutive notes ``n_1, ..., n_3k`` where the patterns of ``(n_1..n_k)``, ``(n_(k+1)..n_2k)`` and
        ``(n_(2k+1)..n_3k)`` are the same.

        Exclusions:

        * Notes that belong to a run (as defined in step 3 of ``tokenize_corpus``) are excluded from pattern
          detection. A pattern cannot span across excluded notes: they act as boundaries.
        * Patterns with more than ``PATTERN_MAX_DISTINCT_PITCHES`` (3) distinct pitches are not counted: C G E G
          (pitch order (1, 3, 2, 3), 3 distinct pitches) counts, C G E A (1, 3, 2, 4) does not.
        * Patterns whose pitch order is strictly increasing, ``(1, 2, ..., k)``, or strictly decreasing,
          ``(k, ..., 2, 1)``, are not counted as repeated patterns. With at most 3 distinct pitches this only
          excludes (1, 2, 3) and (3, 2, 1).

        Overlapping matches: when two matches cover some of the same notes, the one with the shorter ``k`` wins
        (the smallest repeating unit). E.g. A B C repeated 6 times also matches as A B C A B C repeated 3 times;
        the k = 3 match is used, giving 6 chords. If both have the same ``k``, the one that starts earlier wins.
        The losing match is dropped as a whole.

    When two patterns are "the same"
        Each relative onset gap is rounded to the nearest multiple of 1/12 (fixed bins), e.g. 0.24 -> 3/12,
        0.30 -> 4/12. A pattern's key is (pitch order, rounded gaps), and two patterns are the same exactly when their
        keys are equal. Using a fixed key, rather than "within 1/12 of each other", makes "the same" an equivalence:
        every occurrence belongs to exactly one pattern, so PatternTable counts are well defined.

        Only the pitch order is compared, not the actual pitches, so transposed repeats and sequences count as the
        same pattern: C-E-D, D-F-E and E-G-F all have pitch order (1, 3, 2).

    Replacement
        Each repetition is replaced by its own chord: one long note per distinct pitch in that repetition. The chord
        starts at the repetition's first onset and lasts until the next repetition's first onset, so successive
        chords join with no gaps. The final repetition's chord lasts until its last note ends. Each chord note's
        velocity is the max velocity of that pitch's notes in the repetition (as in BPE).

        Example: C4, G4, E4, G4 as 16ths, repeated three times, becomes three C4-E4-G4 chords of 1 beat each
        (the last one ending when its final G4 ends). The sequence C-E-D, D-F-E, E-G-F becomes three successive
        chords C-D-E, D-E-F, E-F-G.

    PatternTable
        Every repeated pattern found is recorded in a separate PatternTable, with:

        * the pattern itself,
        * its pitch order and relative onset gaps,
        * an example occurrence, as a list of ``(onset, pitch)`` with onsets relative to the first note, and
        * how many times it repeats across the corpus.

    Target complexity: O(n), or O(n log n) if the voice has to be re-sorted.
    """
    return _impl_remove_repeated_patterns(voice, table, patterns)


def bpe_within_instruments(
    corpus: Corpus, token_budget: int = 10_000, min_count: int = 50, progress: Callable[[str], None] = print
) -> None:
    """Byte-pair-style merging of notes that follow one another in the same voice.

    Works on the whole corpus at once, because "the most common merged note class" is counted over all pieces.
    Updates each piece's ReductionTable and the corpus-wide BPEMergeTable in place.

    Starting point
        Each note's note field encodes tax-20 instrument and pitch as ``128 * instrument + pitch`` (see step 0 of
        ``tokenize_corpus``). This step adds new tokens on top of those.

    Times
        Every rule below compares times in beats: ``onset_beats``, ``dur_beats`` and
        ``offset_beats = onset_beats + dur_beats``. The seconds fields (``onset``, ``dur``) are only carried along
        so merged notes can be measured and written as AMT tokens.

    "Follows"
        For notes ``a`` and ``b`` in the same voice, ``b`` follows ``a`` when ``a.onset_beats < b.onset_beats``
        and ``|a.offset_beats - b.onset_beats| < 1/8 beat``, i.e. ``b`` starts within 1/8 beat of ``a`` ending
        (in beats, so the rule means the same notation at any tempo).

    Merged note
        Every pair where ``b`` follows ``a`` defines a merged note ``m``. Here ``a`` and ``b`` are either original
        notes or merged notes from earlier iterations. ``m`` has:

        * ``children = (class of a, class of b)``. The class of an original note is just its tax-20 instrument;
          the class of a merged note is defined below. Classes, not tokens, are used so that the pitch stays out
          of the class (a token also fixes the starting pitch).
        * ``onset_beats = a.onset_beats`` and ``dur_beats = b.offset_beats - a.onset_beats``
        * ``onset = a.onset`` and ``dur = (b.onset + b.dur) - a.onset`` (the same span in seconds)
        * ``track_numbers = a.track_numbers`` and ``voice_ids = a.voice_ids``
        * ``vel = max(a.vel, b.vel)``
        * ``pitch = a.pitch``
        * ``pitch_gap = b.pitch - a.pitch`` (signed, not absolute)
        * ``subdivision = (b.onset_beats - a.onset_beats) / m.dur_beats``. For example, if ``b`` starts 1/2 beat
          after ``a`` and ``m.dur_beats`` is 3/2, the subdivision is 1/3. Because ``m.dur_beats`` ends at ``b``'s
          offset, how long ``b`` is held is part of the subdivision, so articulation is part of the class
          (intended): two quarter notes give 6/12 legato but 8/12 staccato, i.e. different classes.

    Merged note class
        Two merged notes are in the same class if they have the same ``children``, ``pitch_gap`` and
        ``subdivision``, where the subdivision is first rounded to the nearest multiple of 1/12 (fixed bins, e.g.
        0.50 and 0.52 both -> 6/12). The rounded value is part of the class key, so every merged note belongs to
        exactly one class and "the most common class" is well defined.
        They can differ in onset, track, duration, velocity and pitch.

        Because each class names its two children, one pitch gap and one subdivision per merge are enough to
        rebuild every original note's relative onset and pitch, by unpacking the children recursively. This is
        approximate in one way: a child's own duration is not stored, so ``a`` is taken to end where ``b`` starts,
        which "follows" only guarantees to within 1/8 beat. Inner notes' durations and velocities are lost.

    One BPE iteration
        1. Find the most common merged note class. Instances are counted greedily, left to right in onset order:
           a note that is already used by an earlier instance cannot be part of another, so C -> D -> E counts
           (C, D) once and leaves E free.
        2. Replace every instance of it (the same non-overlapping, left-to-right set) with a single merged note
           quad ``(onset, dur, note_token, vel)``.
           ``note_token`` is a new token meaning "this class, on this instrument, starting at this pitch", so one
           class can create at most 20 x 128 new tokens. In practice far fewer, because each instrument only
           covers part of the pitch range.
        3. Record each new token in the BPEMergeTable: its two children, instrument, starting pitch, pitch gap and
           subdivision.

    Later iterations may merge merged notes again, so one merged note quad can end up standing for many original
    notes.

    Stopping rule: repeat iterations until ``token_budget`` new tokens have been created by this step, or until
    the most common class has fewer than ``min_count`` instances in the corpus, whichever comes first. Log every
    iteration (class, instance count, tokens created) so the budget and floor can be tuned after the first run.

    Target complexity: O(n) per iteration.
    """
    _impl_bpe(corpus, 6, token_budget, min_count, progress)


def bpe_between_instruments(
    corpus: Corpus, token_budget: int = 10_000, min_count: int = 50, progress: Callable[[str], None] = print
) -> None:
    """Byte-pair-style merging of simultaneous notes across different tracks.

    Works on the whole corpus at once, like bpe_within_instruments, and updates the same tables in place.

    Candidates
        Notes in different tracks whose ``onset_beats`` are equal within BEAT_TOL (1/48 beat) and whose durations
        are within 1/16 beat of each other. Doubled parts in these files usually start on identical ticks.

    Merged note
        Each merge combines two pieces ``x`` and ``y``: original notes, or merged notes from earlier iterations.
        All the notes inside are ordered by tax-20 instrument index, with ties (e.g. Violin I and Violin II, both
        violin) broken by track number; ``x`` is the piece whose first note comes first in that order. The merged
        note ``m`` has:

        * ``children = (class of x, class of y)``. The class of an original note is just its tax-20 instrument.
        * ``pitch_gap = y.pitch - x.pitch`` (signed)
        * ``onset_beats`` / ``onset``: the earliest onset among its notes (in beats / in seconds)
        * ``track_numbers``: the set of tracks its notes came from
        * ``instruments``: its notes' tax-20 instruments, in the order above. This starts with two and grows as
          merged notes are merged again.
        * ``dur_beats`` / ``dur``: the longest duration among its notes (in beats / in seconds)
        * ``vel``: the highest velocity among its notes
        * ``pitch``: the pitch of the first note in the order
        * ``pitch_gaps``: the list of signed pitch gaps between consecutive notes in the order (one gap for two
          notes, two gaps for three, and so on)

    Merged note class
        Two merged notes are in the same class if they have the same ``children`` and ``pitch_gap``. Unpacking the
        children recursively gives the full ``instruments`` and ``pitch_gaps`` lists, so two notes in the same class
        always have the same lists.

    One BPE iteration
        1. Find the most common merged note class. Instances are counted greedily, left to right in onset order,
           and at the same onset in instrument order (tax-20 index, then track number): a note already used
           cannot be part of another instance. With three doubling tracks, the first two pair up and the third
           stays free; it can join that merged note in a later iteration.
        2. Replace every instance of it (the same non-overlapping set) with a single merged note quad
           ``(onset, dur, note_token, vel)``.
           ``note_token`` is a new token meaning "this class, starting at this pitch", so one class can create
           up to 128 new tokens (one per starting pitch).
        3. Record each new token in the BPEMergeTable: its two children, starting pitch and pitch gap.

    Later iterations may merge merged notes again.

    Stopping rule: repeat iterations until ``token_budget`` new tokens have been created by this step, or until
    the most common class has fewer than ``min_count`` instances in the corpus, whichever comes first. Log every
    iteration (class, instance count, tokens created) so the budget and floor can be tuned after the first run.

    Target complexity: O(n) per iteration.
    """
    _impl_bpe(corpus, 7, token_budget, min_count, progress)


# --- Implementation -----------------------------------------------------------------------------------------------


def _tempo_map(midi: mido.MidiFile) -> list[tuple[int, float, int]]:
    """(tick, seconds at that tick, microseconds per beat from there) at every tempo change, starting at tick 0."""
    events = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                events.append((tick, msg.tempo))
    events.sort(key=lambda e: e[0])
    out, seconds, last_tick, tempo = [(0, 0.0, 500_000)], 0.0, 0, 500_000
    for tick, new_tempo in events:
        seconds += (tick - last_tick) * tempo / 1e6 / midi.ticks_per_beat
        last_tick, tempo = tick, new_tempo
        if out[-1][0] == tick:
            out[-1] = (tick, seconds, tempo)
        else:
            out.append((tick, seconds, tempo))
    return out


def _track_names(midi: mido.MidiFile) -> list[str]:
    """Names of the tracks that contain notes, in the same order as midi_to_raw_notes."""
    names = []
    for k, track in enumerate(midi.tracks):
        if any(m.type == "note_on" and m.velocity > 0 for m in track):
            name = next((m.name for m in track if m.type == "track_name" and m.name.strip()), f"track {k}")
            names.append(name.strip())
    return names


def read_piece(path: Path) -> Piece:
    """Step 0 (and step 1): read a MIDI into Note objects, mapped onto tax-20, sorted per track."""
    tpb, tracks = midi_to_raw_notes(path)
    midi = mido.MidiFile(str(path))
    piece = Piece(path.name, path, tpb, len(tracks), _track_names(midi), ReductionTable(), _tempo_map(midi))
    if len(piece.track_names) != len(tracks):  # should not happen; keep names aligned anyway
        piece.track_names = [f"track {k}" for k in range(len(tracks))]
    for t, raws in enumerate(tracks):
        track_key = (t,)  # one shared tuple per track (memory)
        for r in raws:
            inst = tax20_index(r.instrument)
            if inst is None:
                piece.ignored.append((r.onset, r.dur if r.dur is not None else 0.0, r.pitch, t, r.instrument))
                continue
            if r.dur_ticks is None or r.dur is None:  # never closed: 1/2 beat
                dur_ticks = int(UNCLOSED_DUR_BEATS * tpb)
                dur = piece.seconds_at(r.onset_ticks + dur_ticks) - r.onset
            else:
                dur_ticks, dur = r.dur_ticks, float(r.dur)
            piece.table.add(
                Note(piece.table.new_id(), r.onset, dur, r.onset_ticks, dur_ticks, tpb, r.pitch, inst,
                     128 * inst + r.pitch, r.velocity, track_key)
            )  # fmt: skip
    return piece


def _present(notes: Iterable[Note]) -> list[Note]:
    return sorted((n for n in notes if n.is_present_in_reduction), key=lambda n: (n.onset_ticks, n.pitch, n.note_id))


def _notes_by_track(piece: Piece) -> dict[int, list[Note]]:
    by = defaultdict(list)
    for n in piece.table.notes.values():
        if n.is_present_in_reduction:
            by[n.track_numbers[0]].append(n)
    return {t: _present(ns) for t, ns in sorted(by.items())}


def separate_voices(piece: Piece) -> None:
    """Step 2: split every track into voices (polyphony_subsegments from ~/orchestration, plus the spec's rules)."""
    if str(ORCHESTRATION_SRC) not in sys.path:
        sys.path.insert(0, str(ORCHESTRATION_SRC))
    from orchestration.reduction import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        polyphony_subsegments,
    )

    from orchestration_v2.fast_splitter import install  # noqa: PLC0415

    install(polyphony_subsegments)  # O(n log n) versions of its quadratic helpers (same results)
    TimedNote, split_timed_segment = polyphony_subsegments.TimedNote, polyphony_subsegments.split_timed_segment

    voice_of: dict[tuple, int] = {}

    def vid(key: tuple, kind: str) -> int:
        if key not in voice_of:
            voice_of[key] = len(voice_of)
            piece.voice_kind[voice_of[key]] = kind
        return voice_of[key]

    for t, notes in _notes_by_track(piece).items():
        segments, current, reach = [], [], -math.inf
        for n in notes:  # a: split at silences longer than 3 s
            if current and n.onset - reach > SILENCE_SPLIT_S:
                segments.append(current)
                current = []
            current.append(n)
            reach = max(reach, n.offset)
        if current:
            segments.append(current)

        for si, seg in enumerate(segments):
            voices: dict[int, set[int]] = defaultdict(set)  # note index -> voice ids
            primary: dict[int, int] = {}
            timed = [
                TimedNote(index=i, onset=n.onset, offset=n.offset, pitch=n.pitch, program=TAX20_GM[n.instrument],
                          note_id=n.note_id, velocity=n.vel, start_tick=n.onset_ticks, end_tick=n.offset_ticks,
                          ticks_per_beat=n.ticks_per_beat)
                for i, n in enumerate(seg)
            ]  # fmt: skip
            try:
                split = split_timed_segment(timed)
                overflow, max_k = split.overflow, split.max_k
            except Exception as e:  # treat a splitter failure like an overflow stretch, and log it
                piece.log.append(f"track {t} segment {si}: splitter failed ({e!r}); used top-down layering")
                split, overflow, max_k = None, True, 4
            if overflow or split is None:
                _overflow_layers(seg, max_k, lambda layer, t=t, si=si: vid(
                    (t, si, "ovf", layer) if layer else (t, si, "left"), "overflow" if layer else "leftover"),
                    voices, primary)  # fmt: skip
            else:
                for gi, group in enumerate(split.groups):
                    if group.empty:
                        continue
                    for tn in group.notes:
                        src = tn.source_index
                        for label in group.labels.get(tn.index, ()):
                            v = vid((t, si, gi, label), "group")
                            voices[src].add(v)
                            if tn.fill_of is None:
                                primary.setdefault(src, v)
            for i, n in enumerate(seg):  # g: everything else -> this segment's leftover voice
                if not voices[i]:
                    v = vid((t, si, "left"), "leftover")
                    voices[i].add(v)
                    primary[i] = v
                n.voice_ids = tuple(sorted(voices[i]))
                n.primary_voice = primary.get(i, n.voice_ids[0])
    _rebuild_voices(piece, use_copies=True)


def _overflow_layers(seg: list[Note], max_k: int, layer_vid: Callable[[int], int], voices, primary) -> None:
    """Overflow stretch: voice = 1 + number of distinct higher pitches sounding at the onset; past max_k: leftover.

    O(n log n): notes are visited in onset order with a heap of sounding notes (by offset) and a sorted list of
    the distinct sounding pitches (at most 128), so each count is a bisect.
    """
    order = sorted(range(len(seg)), key=lambda i: seg[i].onset_ticks)
    sounding: list[tuple[int, int]] = []  # heap of (offset_ticks, pitch)
    count: Counter = Counter()  # pitch -> sounding notes
    distinct: list[int] = []  # sorted distinct sounding pitches
    leftover = layer_vid(0)

    def add(pitch: int) -> None:
        if count[pitch] == 0:
            bisect.insort(distinct, pitch)
        count[pitch] += 1

    g = 0
    while g < len(order):
        onset = seg[order[g]].onset_ticks
        h = g
        while h < len(order) and seg[order[h]].onset_ticks == onset:
            h += 1
        while sounding and sounding[0][0] <= onset:
            _, pitch = heapq.heappop(sounding)
            count[pitch] -= 1
            if count[pitch] == 0:
                distinct.pop(bisect.bisect_left(distinct, pitch))
        for i in order[g:h]:  # notes starting together all sound at this onset
            add(seg[i].pitch)
            heapq.heappush(sounding, (seg[i].offset_ticks, seg[i].pitch))
        for i in order[g:h]:
            layer = len(distinct) - bisect.bisect_right(distinct, seg[i].pitch) + 1
            v = layer_vid(layer) if layer <= max_k else leftover
            voices[i].add(v)
            primary[i] = v
        g = h


def _rebuild_voices(piece: Piece, use_copies: bool) -> None:
    by = defaultdict(list)
    for n in piece.table.notes.values():
        if n.is_present_in_reduction:
            for v in n.voice_ids if use_copies else (n.primary_voice,):
                by[v].append(n)
    piece.voices = {v: _present(ns) for v, ns in sorted(by.items())}


def mark_duplicates_and_runs(piece: Piece) -> None:
    """Step 3: drop same-pitch duplicates within a track (keep the longer), then mark runs per voice."""
    tol = float(BEAT_TOL * piece.ticks_per_beat)  # float: compared only with integer ticks (exact)
    for notes in _notes_by_track(piece).values():
        by_pitch = defaultdict(list)
        for n in notes:
            by_pitch[n.pitch].append(n)
        for same in by_pitch.values():
            keep = same[0]
            for n in same[1:]:
                if n.onset_ticks - keep.onset_ticks <= tol:
                    loser = n if n.dur_ticks <= keep.dur_ticks else keep  # equal length: the first in the file stays
                    keep = keep if loser is n else n
                    loser.is_present_in_reduction = False  # not recorded in the ReductionTable (spec)
                    loser.replaced_step = 3
                else:
                    keep = n
    _rebuild_voices(piece, use_copies=True)

    run_id = 0
    limit = float(RUN_MAX_BEATS * piece.ticks_per_beat) + tol
    for notes in piece.voices.values():
        start = 0
        for i in range(1, len(notes) + 1):
            ok = i < len(notes) and (
                notes[i].onset_ticks > notes[i - 1].onset_ticks
                and notes[i].onset_ticks - notes[i - 1].onset_ticks <= limit
                and notes[i].dur_ticks <= limit
                and notes[i - 1].dur_ticks <= limit
            )
            if not ok:
                if i - start >= 5:
                    for n in notes[start:i]:
                        if n.run_id < 0:
                            n.run_id = run_id
                    run_id += 1
                start = i


def _new_note(table: ReductionTable, like: Note, onset_ticks: int, end_ticks: int, onset: float, end: float,
              pitch: int, instrument: int, vel: int, voice: int) -> Note:  # fmt: skip
    return Note(table.new_id(), onset, end - onset, onset_ticks, end_ticks - onset_ticks, like.ticks_per_beat, pitch,
                instrument, 128 * instrument + pitch, vel, like.track_numbers, (voice,), voice)  # fmt: skip


def _voice_of(notes: list[Note]) -> int:
    """The voice a list from Piece.voices belongs to (the one all its notes share)."""
    common = set(notes[0].voice_ids)
    for n in notes[1:]:
        common &= set(n.voice_ids)
    return min(common) if common else notes[0].primary_voice


def _impl_remove_repeated_notes(voice: list[Note], table: ReductionTable) -> list[Note]:
    notes = _present(voice)
    if len(notes) < 4:
        return notes
    v = _voice_of(notes)
    tol = float(BEAT_TOL * notes[0].ticks_per_beat)
    # next_onset[i]: the first onset in the voice after notes[i]'s onset. Consecutive chain notes must be
    # adjacent in time: no note of the voice may start strictly between them (notes sharing an onset are fine).
    next_onset = [0] * len(notes)
    following = math.inf
    for k in range(len(notes) - 1, -1, -1):
        next_onset[k] = following
        if k > 0 and notes[k - 1].onset_ticks < notes[k].onset_ticks:
            following = notes[k].onset_ticks
    used = [False] * len(notes)
    new_notes = []
    for i in range(len(notes)):
        if used[i]:
            continue
        for j in range(i + 1, min(i + 5, len(notes))):
            if used[j] or notes[j].pitch != notes[i].pitch or notes[j].onset_ticks != next_onset[i]:
                continue
            g = notes[j].onset_ticks - notes[i].onset_ticks
            if g <= 0:
                continue
            chain, gmin, gmax, last = [i, j], g, g, j
            while True:
                nxt = next((m for m in range(last + 1, min(last + 5, len(notes)))
                            if not used[m] and notes[m].pitch == notes[i].pitch), None)  # fmt: skip
                if nxt is None:
                    break
                g = notes[nxt].onset_ticks - notes[last].onset_ticks
                if notes[nxt].onset_ticks != next_onset[last] or g <= 0 or max(gmax, g) - min(gmin, g) > tol:
                    break
                chain.append(nxt)
                gmin, gmax, last = min(gmin, g), max(gmax, g), nxt
            if len(chain) >= 4:
                old = [notes[c] for c in chain]
                first, final = old[0], old[-1]
                m = _new_note(table, first, first.onset_ticks, final.offset_ticks, first.onset, final.offset,
                              first.pitch, first.instrument, max(n.vel for n in old), v)  # fmt: skip
                table.replace(4, old, [m])
                new_notes.append(m)
                for c in chain:
                    used[c] = True
                break
    return _present([n for k, n in enumerate(notes) if not used[k]] + new_notes)


def _dense_ranks(pitches: list[int]) -> tuple[int, ...]:
    order = {p: r for r, p in enumerate(sorted(set(pitches)), start=1)}
    return tuple(order[p] for p in pitches)


def _impl_remove_repeated_patterns(voice: list[Note], table: ReductionTable, patterns: PatternTable) -> list[Note]:
    notes = _present(voice)
    n_notes = len(notes)
    if n_notes < 9:
        return notes
    v = _voice_of(notes)
    used = [False] * n_notes
    new_notes: list[Note] = []

    def key(s: int, k: int):
        window = notes[s : s + k]
        if any(n.run_id >= 0 for n in window):
            return None
        gaps = [window[i + 1].onset_ticks - window[i].onset_ticks for i in range(k - 1)]
        last = window[-1]
        gaps.append(notes[s + k].onset_ticks - last.onset_ticks if s + k < n_notes else last.dur_ticks)
        total = sum(gaps)
        if total <= 0:
            return None
        ranks = _dense_ranks([n.pitch for n in window])
        if max(ranks) > PATTERN_MAX_DISTINCT_PITCHES:
            return None
        if ranks == tuple(range(1, k + 1)) or ranks == tuple(range(k, 0, -1)):
            return None
        return ranks, tuple(round(12 * g / total) for g in gaps)

    for k in range(3, 11):
        cache: dict[int, object] = {}

        def kk(s: int, cache: dict[int, object] = cache, k: int = k):
            if s not in cache:
                cache[s] = key(s, k)
            return cache[s]

        s = 0
        while s + 3 * k <= n_notes:
            k0 = kk(s)
            if k0 is None or any(used[s : s + 3 * k]) or kk(s + k) != k0 or kk(s + 2 * k) != k0:
                s += 1
                continue
            reps = 3
            while s + (reps + 1) * k <= n_notes and not any(used[s + reps * k : s + (reps + 1) * k]) \
                    and kk(s + reps * k) == k0:  # fmt: skip
                reps += 1
            new_notes += _replace_pattern(notes, s, k, reps, table, v)
            for i in range(s, s + reps * k):
                used[i] = True
            entry = patterns.entries.get(k0)  # type: ignore[arg-type]
            if entry is None:
                base = notes[s]
                example = [(Fraction(n.onset_ticks - base.onset_ticks, n.ticks_per_beat), n.pitch)
                           for n in notes[s : s + k]]  # fmt: skip
                entry = patterns.entries[k0] = PatternEntry(k0[0], k0[1], example)  # type: ignore[index]
            entry.count += reps
            s += reps * k
    return _present([n for i, n in enumerate(notes) if not used[i]] + new_notes)


def _replace_pattern(notes: list[Note], s: int, k: int, reps: int, table: ReductionTable, v: int) -> list[Note]:
    made = []
    for r in range(reps):
        rep = notes[s + r * k : s + (r + 1) * k]
        if r + 1 < reps:
            nxt = notes[s + (r + 1) * k]
            end_ticks, end = nxt.onset_ticks, nxt.onset
        else:
            end_ticks, end = rep[-1].offset_ticks, rep[-1].offset
        chord = []
        for pitch in sorted({n.pitch for n in rep}):
            same = [n for n in rep if n.pitch == pitch]
            chord.append(_new_note(table, rep[0], rep[0].onset_ticks, end_ticks, rep[0].onset, end, pitch,
                                   same[0].instrument, max(n.vel for n in same), v))  # fmt: skip
        table.replace(5, rep, chord)
        made += chord
    return made


def process_piece(path: Path) -> Piece:
    """Steps 0-5 for one piece (independent of other pieces)."""
    piece = read_piece(path)  # steps 0 and 1
    separate_voices(piece)  # step 2
    mark_duplicates_and_runs(piece)  # step 3
    for v in list(piece.voices):  # step 4
        piece.voices[v] = remove_repeated_notes(piece.voices[v], piece.table)
    _rebuild_voices(piece, use_copies=True)
    for v in list(piece.voices):  # step 5
        piece.voices[v] = remove_repeated_patterns(piece.voices[v], piece.table, piece.patterns)
    shared: dict[int, tuple[int, ...]] = {}
    for n in piece.table.notes.values():  # copies end here: keep only the primary voice
        if n.is_present_in_reduction:
            n.voice_ids = shared.setdefault(n.primary_voice, (n.primary_voice,))
    _rebuild_voices(piece, use_copies=False)
    return piece


# --- BPE (steps 6 and 7) -----------------------------------------------------------------------------------------


def _within_instances(notes: list[Note], only: tuple | None = None) -> list[tuple[tuple, int, int]]:
    """Greedy, left-to-right, non-overlapping (per class) instances of "b follows a" in one voice.

    A note already used by an earlier instance of a class cannot be part of a later instance of that class.
    """
    out: list[tuple[tuple, int, int]] = []
    used: dict[tuple, set[int]] = defaultdict(set)
    n_notes = len(notes)
    if not notes:
        return out
    tol = float(FOLLOW_TOL_BEATS * notes[0].ticks_per_beat)  # float: compared only with integer ticks (exact)
    for i, a in enumerate(notes):
        a_off = a.offset_ticks
        for j in range(i + 1, min(i + 9, n_notes)):
            b = notes[j]
            if b.onset_ticks >= a_off + tol:
                break
            if b.onset_ticks <= a.onset_ticks or abs(a_off - b.onset_ticks) >= tol:
                continue
            span = b.offset_ticks - a.onset_ticks
            if span <= 0:
                continue
            key = (a.cls, b.cls, b.pitch - a.pitch, round(12 * (b.onset_ticks - a.onset_ticks) / span))
            if only is not None and key != only:
                continue
            if i in used[key] or j in used[key]:
                continue
            used[key].update((i, j))
            out.append((key, i, j))
    return out


def _between_clusters(notes: list[Note]) -> list[list[Note]]:
    """Group present notes into onset clusters (within BEAT_TOL of the cluster's first onset), in merge order."""
    if not notes:
        return []
    tol = float(BEAT_TOL * notes[0].ticks_per_beat)
    clusters, cur = [], [notes[0]]
    for n in notes[1:]:
        if n.onset_ticks - cur[0].onset_ticks <= tol:
            cur.append(n)
        else:
            clusters.append(cur)
            cur = [n]
    clusters.append(cur)
    return [sorted(c, key=lambda n: (n.instrument, n.track_numbers[0], n.pitch, n.note_id)) for c in clusters if
            len(c) > 1]  # fmt: skip


def _between_instances(cluster: list[Note], only: tuple | None = None) -> list[tuple[tuple, int, int]]:
    out, used = [], defaultdict(set)
    tol = float(BETWEEN_DUR_TOL_BEATS * cluster[0].ticks_per_beat)
    for i, x in enumerate(cluster):
        for j in range(i + 1, min(i + 13, len(cluster))):
            y = cluster[j]
            if set(x.track_numbers) & set(y.track_numbers) or abs(x.dur_ticks - y.dur_ticks) > tol:
                continue
            key = (x.cls, y.cls, y.pitch - x.pitch)
            if only is not None and key != only:
                continue
            if i in used[key] or j in used[key]:
                continue
            used[key].update((i, j))
            out.append((key, i, j))
    return out


class _Units:
    """Per-unit (voice or onset cluster) class counts with a lazily updated global max-heap."""

    def __init__(self, units: list[list[Note]], count: Callable[[list[Note]], Counter]):
        self.units, self.count = units, count
        self.unit_counts = [count(u) for u in units]
        self.total: Counter = Counter()
        for c in self.unit_counts:
            self.total.update(c)
        self.heap = [(-c, k) for k, c in self.total.items()]
        heapq.heapify(self.heap)
        self.where: dict[tuple, set[int]] = defaultdict(set)
        for ui, c in enumerate(self.unit_counts):
            for k in c:
                self.where[k].add(ui)

    def best(self) -> tuple[tuple, int] | None:
        while self.heap:
            c, k = self.heap[0]
            if self.total.get(k, 0) == -c and c < 0:
                return k, -c
            heapq.heappop(self.heap)
        return None

    def refresh(self, ui: int) -> None:
        old = self.unit_counts[ui]
        new = self.count(self.units[ui])
        for k, c in old.items():
            self.total[k] -= c
            if self.total[k] <= 0:
                del self.total[k]
            self.where[k].discard(ui)
        for k, c in new.items():
            self.total[k] += c
            self.where[k].add(ui)
        for k in set(old) | set(new):
            if self.total.get(k):
                heapq.heappush(self.heap, (-self.total[k], k))
        self.unit_counts[ui] = new


def _bpe_units(pieces: list[Piece], step: int) -> tuple[list[list[Note]], list[int]]:
    """Step 6: one unit per voice. Step 7: one unit per onset cluster. Returns (units, owning piece index)."""
    units, owner = [], []
    for pi, piece in enumerate(pieces):
        if step == 6:
            groups = [notes for notes in piece.voices.values() if len(notes) > 1]
        else:
            groups = _between_clusters(_present(n for n in piece.table.notes.values()))
        units += groups
        owner += [pi] * len(groups)
    return units, owner


def _bpe_count(step: int) -> Callable[[list[Note]], Counter]:
    if step == 6:
        return lambda u: Counter(k for k, _, _ in _within_instances(u))
    return lambda u: Counter(k for k, _, _ in _between_instances(u))


def _merge_in_unit(u: list[Note], step: int, key: tuple, cid: int, table: ReductionTable,
                   token: Callable[[int, int], int]) -> tuple[list[Note], int, dict[int, int]]:  # fmt: skip
    """Merge every (greedy, non-overlapping) instance of ``key`` in one unit.

    ``token(pitch, instrument)`` gives the merged note's token. Returns the unit's new note list, the number of
    merges, and {starting pitch: instrument} of the merged notes made.
    """
    inst = _within_instances(u, key) if step == 6 else _between_instances(u, key)
    gone, put, pitches = set(), {}, {}
    for _, i, j in inst:
        a, b = u[i], u[j]
        if step == 6:
            onset_ticks, onset, end_ticks, end = a.onset_ticks, a.onset, b.offset_ticks, b.offset
            tracks = a.track_numbers
        else:
            onset_ticks, onset = min(a.onset_ticks, b.onset_ticks), min(a.onset, b.onset)
            end_ticks, end = max(a.offset_ticks, b.offset_ticks), max(a.offset, b.offset)
            tracks = a.track_numbers + b.track_numbers
        m = Note(table.new_id(), onset, end - onset, onset_ticks, end_ticks - onset_ticks, a.ticks_per_beat, a.pitch,
                 a.instrument, token(a.pitch, a.instrument), max(a.vel, b.vel), tracks, a.voice_ids, a.primary_voice,
                 merge_class=cid)  # fmt: skip
        table.replace(step, [a, b], [m])
        put[i] = m
        gone.add(j)
        pitches.setdefault(a.pitch, a.instrument)
    new_u = [put.get(i, n) for i, n in enumerate(u) if i not in gone]
    if step == 7:
        new_u.sort(key=lambda n: (n.instrument, n.track_numbers[0], n.pitch, n.note_id))
    return new_u, len(inst), pitches


def _class_key(step: int, key: tuple) -> tuple[str, int, int, int, int]:
    return ("w", key[0], key[1], key[2], key[3]) if step == 6 else ("b", key[0], key[1], key[2], -1)


def _new_tokens(corpus: Corpus, step: int, key: tuple, cid: int, pitches: dict[int, int]) -> int:
    """Give every (class, starting pitch) seen this iteration a token, in pitch order. Returns how many were new."""
    made = 0
    for pitch in sorted(pitches):
        if (cid, pitch) in corpus.merges.token_of:
            continue
        tok = corpus.merges.next_token
        corpus.merges.next_token += 1
        corpus.merges.token_of[(cid, pitch)] = tok
        corpus.merges.entries[tok] = MergeEntry(tok, cid, (key[0], key[1]), pitches[pitch] if step == 6 else None,
                                                pitch, key[2], Fraction(key[3], 12) if step == 6 else None)  # fmt: skip
        made += 1
    return made


def _log_iteration(corpus, step, iteration, cid, key, n_inst, merged, made_now, tokens_made, progress) -> None:
    corpus.bpe_log.append(dict(step=step, iteration=iteration, merge_class=cid, children=f"{key[0]}+{key[1]}",
                               pitch_gap=key[2], subdivision_12ths=key[3] if step == 6 else "", instances=n_inst,
                               merged=merged, tokens_created=made_now, tokens_total=tokens_made))  # fmt: skip
    if iteration % 10 == 0:
        progress(f"step {step}: iteration {iteration}, {tokens_made} tokens, last class had {n_inst} instances")


def _impl_bpe(corpus: Corpus, step: int, token_budget: int, min_count: int, progress: Callable[[str], None]) -> None:
    """One BPE step in this process: one class per iteration, merged everywhere, counts updated incrementally."""
    pieces = corpus.pieces
    units, owner = _bpe_units(pieces, step)
    state = _Units(units, _bpe_count(step))
    tokens_made = iteration = 0
    while tokens_made < token_budget:
        found = state.best()
        if found is None or found[1] < min_count:
            break
        key, n_inst = found
        cid = corpus.class_id(_class_key(step, key))
        merged, seen = 0, {}
        pending: list[Note] = []
        for ui in list(state.where.get(key, ())):
            units[ui], n, pitches = _merge_in_unit(units[ui], step, key, cid, pieces[owner[ui]].table,
                                                   lambda _p, _i: -1)  # fmt: skip
            pending += [m for m in units[ui] if m.note_token == -1]
            for pitch, inst in pitches.items():
                seen.setdefault(pitch, inst)
            merged += n
            state.refresh(ui)
        if merged == 0:  # counting and merging disagree (should not happen): drop the class so we cannot loop
            state.total.pop(key, None)
            progress(f"step {step}: class {key} counted {n_inst} but merged 0; skipped")
            continue
        made_now = _new_tokens(corpus, step, key, cid, seen)
        for m in pending:
            m.note_token = corpus.merges.token_of[(cid, m.pitch)]
        tokens_made += made_now
        iteration += 1
        _log_iteration(corpus, step, iteration, cid, key, n_inst, merged, made_now, tokens_made, progress)
    if step == 6:  # keep Piece.voices in sync for anything that reads it later
        for piece in pieces:
            _rebuild_voices(piece, use_copies=False)
    progress(f"step {step}: done after {iteration} iterations, {tokens_made} new tokens")


def _bpe_worker(conn, pieces: list[Piece], step: int) -> None:
    """A worker process for parallel BPE: owns some pieces, counts and merges inside them on request."""
    units, owner = _bpe_units(pieces, step)
    state = _Units(units, _bpe_count(step))
    conn.send(dict(state.total))
    pending: list[Note] = []
    while True:
        msg = conn.recv()
        if msg[0] == "merge":
            _, key, cid = msg
            before: Counter = Counter()
            after: Counter = Counter()
            merged, seen = 0, {}
            for ui in list(state.where.get(key, ())):
                before.update(state.unit_counts[ui])
                units[ui], n, pitches = _merge_in_unit(units[ui], step, key, cid, pieces[owner[ui]].table,
                                                       lambda _p, _i: -1)  # fmt: skip
                pending += [m for m in units[ui] if m.note_token == -1 and m.merge_class == cid]
                for pitch, inst in pitches.items():
                    seen.setdefault(pitch, inst)
                merged += n
                state.refresh(ui)
                after.update(state.unit_counts[ui])
            delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
            conn.send((delta, merged, seen))
        elif msg[0] == "tokens":
            token_of = msg[1]
            for m in pending:
                m.note_token = token_of[(m.merge_class, m.pitch)]
            pending = []
        else:  # "finish"
            if step == 6:
                for piece in pieces:
                    _rebuild_voices(piece, use_copies=False)
            conn.send(pieces)
            conn.close()
            return


def _impl_bpe_parallel(corpus: Corpus, step: int, token_budget: int, min_count: int,
                       progress: Callable[[str], None], workers: int) -> None:  # fmt: skip
    """Same result as _impl_bpe, with the pieces split across worker processes (exact: same classes, same merges).

    Each iteration the main process picks the most common class from the summed counts, every worker merges it in
    its own pieces and reports how its counts changed, and the main process hands out the new tokens.
    """
    import multiprocessing as mp  # noqa: PLC0415

    pieces = corpus.pieces
    note_counts = [len(p.table.notes) for p in pieces]
    sizes = sorted(range(len(pieces)), key=lambda i: -note_counts[i])
    shards: list[list[int]] = [[] for _ in range(workers)]
    load = [0] * workers
    for i in sizes:  # balance shards by note count
        w = load.index(min(load))
        shards[w].append(i)
        load[w] += note_counts[i]
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for shard in shards:
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_bpe_worker, args=(child, [pieces[i] for i in shard], step))
        proc.start()
        child.close()
        conns.append(parent)
        procs.append(proc)
    n_pieces = len(pieces)
    corpus.pieces = []  # the workers own the pieces until the step ends; drop this process's references
    pieces.clear()
    del pieces

    total: Counter = Counter()
    for conn in conns:
        total.update(conn.recv())
    heap = [(-c, k) for k, c in total.items()]
    heapq.heapify(heap)
    tokens_made = iteration = 0
    while tokens_made < token_budget:
        while heap and (total.get(heap[0][1], 0) != -heap[0][0] or heap[0][0] >= 0):
            heapq.heappop(heap)
        if not heap or -heap[0][0] < min_count:
            break
        n_inst, key = -heap[0][0], heap[0][1]
        cid = corpus.class_id(_class_key(step, key))
        for conn in conns:
            conn.send(("merge", key, cid))
        merged, seen = 0, {}
        for conn in conns:
            delta, n, pitches = conn.recv()
            merged += n
            for pitch, inst in pitches.items():
                seen.setdefault(pitch, inst)
            for k, d in delta.items():
                if d:
                    total[k] += d
                    if total[k] <= 0:
                        del total[k]
                    else:
                        heapq.heappush(heap, (-total[k], k))
        if merged == 0:
            total.pop(key, None)
            progress(f"step {step}: class {key} counted {n_inst} but merged 0; skipped")
            continue
        made_now = _new_tokens(corpus, step, key, cid, seen)
        token_of = {(cid, p): corpus.merges.token_of[(cid, p)] for p in seen}
        for conn in conns:
            conn.send(("tokens", token_of))
        tokens_made += made_now
        iteration += 1
        _log_iteration(corpus, step, iteration, cid, key, n_inst, merged, made_now, tokens_made, progress)

    back: list[Piece | None] = [None] * n_pieces
    for conn, shard in zip(conns, shards, strict=True):
        conn.send(("finish",))
        for i, piece in zip(shard, conn.recv(), strict=True):
            back[i] = piece
    for proc in procs:
        proc.join()
    corpus.pieces = [p for p in back if p is not None]
    progress(f"step {step}: done after {iteration} iterations, {tokens_made} new tokens ({workers} workers)")


def class_track_count(corpus: Corpus, cls: int) -> int:
    """How many tracks a note of this class spans (1 unless it contains a cross-track merge)."""
    if cls < FIRST_MERGED_CLASS:
        return 1
    kind, a, b, _, _ = corpus.class_defs[cls]
    return class_track_count(corpus, a) if kind == "w" else class_track_count(corpus, a) + class_track_count(corpus, b)


def decode_class(corpus: Corpus, cls: int, on: int, dur: int, pitch: int,
                 tracks: tuple[int, ...]) -> list[tuple[int, int, int, int, int]]:  # fmt: skip
    """Unpack a class at a given onset, duration and starting pitch into (onset, dur, pitch, instrument, track)."""
    if cls < FIRST_MERGED_CLASS:
        return [(on, dur, pitch, cls, tracks[0])]
    kind, a, b, gap, sub12 = corpus.class_defs[cls]
    if kind == "w":
        split = on + round(dur * sub12 / 12)
        return decode_class(corpus, a, on, split - on, pitch, tracks) + decode_class(
            corpus, b, split, on + dur - split, pitch + gap, tracks
        )
    na = class_track_count(corpus, a)
    return decode_class(corpus, a, on, dur, pitch, tracks[:na]) + decode_class(
        corpus, b, on, dur, pitch + gap, tracks[na:]
    )


def decode_note(corpus: Corpus, note: Note) -> list[tuple[int, int, int, int, int]]:
    """Unpack a (merged) note into (onset_ticks, dur_ticks, pitch, instrument, track) leaves, as a decoder would."""
    return decode_class(corpus, note.cls, note.onset_ticks, note.dur_ticks, note.pitch, note.track_numbers)


# --- Measurement -------------------------------------------------------------------------------------------------


def present_at(note: Note, step: int) -> bool:
    return note.created_step <= step < note.replaced_step


def measure(piece: Piece) -> dict:
    """Note quads per second in 10 s chunks, at each measuring step (one CSV row)."""
    base = [n for n in piece.table.notes.values() if n.created_step == 0]
    n_chunks = int(max((n.onset for n in base), default=0.0) // CHUNK_SECONDS) + 1
    row: dict = {"piece": piece.name, "n_tracks": piece.n_tracks, "n_chunks": n_chunks}
    for step in MEASURE_STEPS:
        counts = [0] * n_chunks
        for n in piece.table.notes.values():
            if present_at(n, step):
                counts[min(int(n.onset // CHUNK_SECONDS), n_chunks - 1)] += 1
        rates = [c / CHUNK_SECONDS for c in counts]
        lo, hi = rates.index(min(rates)), rates.index(max(rates))
        row.update({
            f"step{step}_note_quads": sum(counts),
            f"step{step}_min": rates[lo], f"step{step}_min_start_s": lo * CHUNK_SECONDS,
            f"step{step}_min_end_s": (lo + 1) * CHUNK_SECONDS,
            f"step{step}_max": rates[hi], f"step{step}_max_start_s": hi * CHUNK_SECONDS,
            f"step{step}_max_end_s": (hi + 1) * CHUNK_SECONDS,
            f"step{step}_mean": round(statistics.fmean(rates), 3),
            f"step{step}_std": round(statistics.pstdev(rates), 3),
        })  # fmt: skip
    return row


def write_measurements(corpus: Corpus, path: Path) -> None:
    rows = [measure(p) for p in corpus.pieces]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def corpus_paths(corpus_dir: Path, outliers_csv: Path) -> list[Path]:
    skip = {r["piece"] for r in csv.DictReader(outliers_csv.open())}
    return [p for p in sorted(corpus_dir.glob("*.mid")) if p.name not in skip]


def _impl_tokenize_corpus(corpus_dir: Path, outliers_csv: Path, measurements_csv: Path, workers: int = 6,
                          token_budget: int = 10_000, min_count: int = 50,
                          progress: Callable[[str], None] = print, limit: int | None = None,
                          bpe_workers: int = 1, only: set[str] | None = None) -> Corpus:  # fmt: skip
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    paths = [p for p in corpus_paths(corpus_dir, outliers_csv) if only is None or p.name in only][:limit]
    pieces: list[Piece] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, piece in enumerate(pool.map(process_piece, paths, chunksize=2), start=1):
            pieces.append(piece)
            if k % 25 == 0 or k == len(paths):
                progress(f"steps 0-5: {k}/{len(paths)} pieces")
    corpus = Corpus(pieces)
    for p in pieces:
        corpus.patterns.merge(p.patterns)
    del pieces  # corpus.pieces is the only reference (parallel BPE hands the pieces to its workers)
    for step in (6, 7):
        if bpe_workers > 1:
            _impl_bpe_parallel(corpus, step, token_budget, min_count, progress, bpe_workers)
        else:
            _impl_bpe(corpus, step, token_budget, min_count, progress)
    write_measurements(corpus, measurements_csv)
    return corpus
