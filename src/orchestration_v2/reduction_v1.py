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

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

# --- Data structures (fields only; no logic yet) -------------------------------------------------------------

# A merge class, as used by both BPE steps. An original note's class is its tax-20 instrument (an int). A merged
# note's class is (kind, class of first child, class of second child, pitch_gap, subdivision), where kind is
# "within" or "between" and subdivision is the rounded Fraction for "within" merges and None for "between".
MergeClass = int | tuple


@dataclass
class Note:
    note_id: int
    onset: float  # seconds
    dur: float  # seconds
    onset_beats: Fraction
    dur_beats: Fraction
    note_token: int
    vel: int
    track_numbers: frozenset[int]  # one track, except cross-track merges from step 7
    voice_ids: frozenset[int]  # several only for fill-in copies, until the end of step 5
    primary_voice: int | None = None  # the voice from step 2d (not a fill-in)
    run_id: int | None = None  # set in step 3 for notes in a run
    is_present_in_reduction: bool = True


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


@dataclass
class PatternEntry:
    pitch_order: tuple[int, ...]  # dense ranks
    rounded_gaps: tuple[Fraction, ...]  # relative onset gaps, rounded to 1/12
    example: list[tuple[Fraction, int]]  # (onset in beats relative to the first note, pitch)
    count: int = 0  # repetitions across the corpus


@dataclass
class PatternTable:
    entries: dict[tuple[tuple[int, ...], tuple[Fraction, ...]], PatternEntry] = field(default_factory=dict)


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
    next_token: int = 20 * 128  # after the 2,560 base tokens


@dataclass
class Piece:
    name: str
    ticks_per_beat: int
    n_tracks: int
    table: ReductionTable


@dataclass
class Corpus:
    pieces: list[Piece]
    patterns: PatternTable = field(default_factory=PatternTable)
    merges: BPEMergeTable = field(default_factory=BPEMergeTable)


# --- Pipeline ----------------------------------------------------------------------------------------------


def tokenize_corpus(corpus_dir: Path, outliers_csv: Path, measurements_csv: Path) -> Corpus:
    # Run the full reduction pipeline over every piece in ``corpus_dir``.
    #
    # Step 0 - read notes. Read each piece with symphonynet.midi_to_raw_notes, which returns ticks per beat and
    #   one RawNote list per track, each note timed in seconds and in ticks (onset_beats / dur_beats come from
    #   RawNote.onset_beats(tpb) / dur_beats(tpb)). Build the ReductionTable's Note objects from them. A note
    #   never closed by a note-off gets AMT's default 250 ms (and the equivalent in beats at its tempo).
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
    # the per-step measurements to measurements_csv (note_quads_per_chunk_v1.csv).

    raise NotImplementedError


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
    #     the same pitch.
    #
    # A repeated note chain n_1, ..., n_k is a sequence of same-pitch notes whose onset gaps in beats
    #   (onset_beats(n_2) - onset_beats(n_1), ..., onset_beats(n_k) - onset_beats(n_(k-1)))
    # are all equal within BEAT_TOL (largest gap - smallest gap <= 1/48 beat). (In seconds they need not be:
    # repeated 8ths in a ritardando are 1/2 beat apart each, but their gaps in seconds grow.)
    #
    # A chain must contain at least 4 notes (k >= 4). Runs (step 3) are not exempt: fast repeated notes such as
    # tremolos or repeated 16ths are collapsed here too (intended).
    #
    # Replacement: every note in the chain is replaced by one note starting at onset(n_1) and ending at
    # onset(n_k) + duration(n_k) (the same in beats and in seconds). Its velocity is the max velocity in the
    # chain (as in BPE), e.g. 80, 60, 70, 60 -> 80.
    #
    # Target complexity: O(n), or O(n log n) if the voice has to be re-sorted.
    raise NotImplementedError


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
        * Patterns whose pitch order is strictly increasing, ``(1, 2, ..., k)``, or strictly decreasing,
          ``(k, ..., 2, 1)``, are not counted as repeated patterns.

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
    raise NotImplementedError


def bpe_within_instruments(corpus: Corpus, num_iterations: int = 10) -> None:
    """Byte-pair-style merging of notes that follow one another in the same voice.

    Works on the whole corpus at once, because "the most common merged note class" is counted over all pieces.
    Updates each piece's ReductionTable and the corpus-wide BPEMergeTable in place.

    Starting point
        Each note's note field encodes tax-20 instrument and pitch as ``128 * instrument + pitch`` (see step 0 of
        ``tokenize_corpus``). This step adds new tokens on top of those.

    "Follows"
        For notes ``a`` and ``b`` in the same voice, ``b`` follows ``a`` when ``a.onset < b.onset`` and
        ``|a.offset - b.onset| < 1/8 beat``, i.e. ``b`` starts within 1/8 beat of ``a`` ending (in beats, so the
        rule means the same notation at any tempo).

    Merged note
        Every pair where ``b`` follows ``a`` defines a merged note ``m``. Here ``a`` and ``b`` are either original
        notes or merged notes from earlier iterations. ``m`` has:

        * ``children = (class of a, class of b)``. The class of an original note is just its tax-20 instrument;
          the class of a merged note is defined below. Classes, not tokens, are used so that the pitch stays out
          of the class (a token also fixes the starting pitch).
        * ``onset = a.onset``
        * ``track_numbers = a.track_numbers`` and ``voice_ids = a.voice_ids``
        * ``dur = b.offset - a.onset``
        * ``vel = max(a.vel, b.vel)``
        * ``pitch = a.pitch``
        * ``pitch_gap = b.pitch - a.pitch`` (signed, not absolute)
        * ``subdivision = (b.onset - a.onset) / m.dur``, measured in beats. For example, if ``b`` starts 1/2 beat
          after ``a`` and ``m.dur`` is 3/2 beats, the subdivision is 1/3. Because ``m.dur`` ends at ``b``'s
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

    Repeat for ``num_iterations``. Later iterations may merge merged notes again, so one merged note quad can
    end up standing for many original notes.

    Target complexity: O(n) per iteration.
    """

    raise NotImplementedError


def bpe_between_instruments(corpus: Corpus, num_iterations: int = 10) -> None:
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
        * ``onset``: the earliest onset among its notes
        * ``track_numbers``: the set of tracks its notes came from
        * ``instruments``: its notes' tax-20 instruments, in the order above. This starts with two and grows as
          merged notes are merged again.
        * ``dur``: the longest duration among its notes
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

    Repeat for ``num_iterations``. Later iterations may merge merged notes again.

    Target complexity: O(n) per iteration.
    """
