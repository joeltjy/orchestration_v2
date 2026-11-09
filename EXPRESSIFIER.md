# Expressifier: expressive piano parts for SymphonyNet piano concertos

Status: **implemented; dry run on 2 pieces done, full run not started** (2026-11-09). Code:
`src/orchestration_v2/sampling/expressify_symphonynet.py`, `scripts/expressify_symphonynet.slurm`,
`tests/test_expressify_symphonynet.py`.

## Goal

Every piano concerto in SymphonyNet has a quantized piano part. We'll run each continuous stretch of that part through the
expressifier. For each stretch we store a note-by-note mapping:

    original (onset, dur, pitch, velocity)  ->  expressive (onset, dur, pitch, velocity)

All timings are relative to the start of the stretch. Original onsets run from 0 to `t_end - t_start`.

## Background: what the expressifier expects

These constraints come from training (`preprocessing/expressifier_tokenize.py`, `models/expressifier.py`). Inference has to
match them:

| Constraint | Value | Consequence |
|---|---|---|
| Context | ≤ 128 notes = 512 tokens (4 per note: onset, duration, note, velocity) | Long segments need windows |
| Training chunk lengths | 128 / 64 / 32 / 16 notes | A 128-note window is in-distribution |
| Note order | score order: by score onset, then pitch. Note *i* of the output is the performance of score note *i* | The mapping is one-to-one by position |
| Time unit | 10 ms; relative onsets 0–99.99 s; durations 0–9.99 s | Each window is re-relativised; durations over 10 s are clipped in the model's view |
| Relativisation | score onsets minus the window's earliest score onset; performance onsets minus the window's earliest performance onset | Done separately per window, on both sides |
| Score velocity | input velocity is always 80 | The original MIDI velocity is not seen by the model (it is still stored in the mapping) |
| Pitch | copied from the score during generation (`copy_pitch=True`) | Expressive pitch = original pitch |
| Encoder / decoder | encoder is bidirectional over the window's score; decoder is causal over performance tokens | Lets a window start with a forced prefix (below) |

Score time is MIDI seconds under the file's own tempo map. Training used `score_onset_s` too, so the scales match.

## Which files

SymphonyNet **contemporary** (45,632 files) has numeric filenames and no title or track-name metadata. Concertos can't be
identified there. SymphonyNet **classical** (`/data/scratch-oc40/joel/symphonynet/classical`, 728 files) has descriptive
names.

- 75 classical names contain "concert". 54 of those have an acoustic-piano track (GM program 0–1), totalling 10.8 h.
- Proposed set: those 54, **minus** the ones that aren't piano concertos:
  - `barber_violin_concerto_{1,3}`: orchestral piano, 8–9 % of notes.
  - `bach_brandeburg_concert_1050_{1,3}`: harpsichord written as program 0.
- That leaves **50 files**: Beethoven, Brahms, Galynin, Grieg, Liszt, Mozart, Rachmaninov, Reinecke and Tchaikovsky, plus
  Beethoven WoO 4. Several pieces appear twice in different arrangements (e.g. Beethoven op. 58 by galimberti and by
  nemitz). Both copies are kept, because they are different MIDI files.
- `beethoven_piano_concerto_61_1_(c)lucarelli.mid` is in the piece list but missing on disk. It is skipped.

## Step 1: extract the piano part

- Piano part = all non-drum tracks with program 0–1, merged into one note list.
  - Most concertos split the soloist across 2 tracks (right and left hand).
  - Mozart K. 242 (3 pianos, 6 tracks) and Liszt / Beethoven 61a (3–6 tracks) merge several pianos into one part.
- Each note keeps `instrument` (index into pretty_midi's `PrettyMIDI.instruments`), `note_index_in_instrument`
  (index into that instrument's `notes`), onset, offset, pitch and velocity.
- Notes with the same (onset, pitch) are exact duplicates across tracks and can't be played twice. Keep the longest one.
  The others get `dup_of = <kept note>` and receive its expressive values, so every original note still has a mapping
  row.
- Sort in score order: onset, then pitch.

## Step 2: split into continuous segments

- Walk the sorted notes, tracking `sounding_until = max(offset)` of everything so far.
- Start a new segment when `next.onset - sounding_until >= 8.0 s`. "Silence" means no piano note is sounding at all.
- Segment span: `t_start` = first onset, `t_end` = last onset.
- Original timings are stored as `onset - t_start`, so they lie in [0, t_end - t_start]. Durations are unchanged.

## Step 3: sliding-window generation within a segment

Window size W = 128 notes; hop H = 64 notes.

```
window 0:  score notes [0, 128)    decoder: sample all 128 notes freely
window 1:  score notes [64, 192)   decoder: notes 64..127 FORCED (already generated), sample 128..191
window 2:  score notes [128, 256)  decoder: notes 128..191 forced, sample 192..255
...
last:      ends at N; the forced prefix is whatever was already generated
```

Once the context is full, the first half is dropped and generation continues. The decoder is causal, so a window that
starts with forced tokens is the same as having generated them in that window.

**Per-window details:**

- **Encoder input:** score tokens for the window's notes, re-relativised to the window's earliest score onset. Each window
  re-encodes its own score, so the second half is seen with 64 notes of look-ahead.
- **Forced prefix:** the already-generated absolute performance onsets for prefix notes, minus `ref` = the earliest of
  those onsets. They are re-tokenised; durations and velocities are unchanged.
  - Training relativised to the earliest onset in the whole chunk, including later notes. With a forced prefix, the later
    notes are unknown, so the prefix minimum is the best stand-in. The first note in score order is almost always the
    earliest performed one.
- **New notes:** absolute performance onset = `ref + token / 100`.
- **Onset range guard:** if a window's score span or prefix span exceeds 99.99 s, halve the window (W → 64 → 32 → 16)
  for that step. This only happens in very sparse passages.
- **Segments of ≤ 128 notes** run as one window. A segment with fewer than 4 notes still runs, as a chunk shorter than the
  training minimum, and is flagged with `short=True`.
- **Positions:** every window starts at decoder position 0, as every training chunk did.

**Model change:** add `prefix_ids: Tensor | None` to `ExpressifierModel.generate`. For positions `t < prefix_len` it
writes the prefix token instead of sampling. The rest of the loop is unchanged.

**Sampling:** temperature 1.0, top-p 0.95. The seed is derived from (piece, segment, sample index), so reruns are
reproducible. Optionally, `--num-samples K` generates K expressive versions per segment.

Each segment has its own random stream. `generate` takes per-row `uniforms` and samples by inverse CDF, so a segment's
result doesn't depend on which other segments share its batch. A test checks this.

**Batching:** `generate` has no KV cache, so each window costs 512 sequential decoder passes. Windows at the same index
across many segments go in one batch, padded with `attention_mask`, to fill the GPU. Dry run: 2 pieces (11 segments, 110 windows,
7,356 notes) took 2.5 min on 1 L40S, about 10 s per round of up to 11 windows.

## Step 4: output

`/data/scratch-oc40/joel/expressified/symphonynet_piano_concertos/<checkpoint-tag>/`

- `<piece>.parquet`, one row per original piano note:

  | column | meaning |
  |---|---|
  | `piece`, `segment`, `sample` | ids |
  | `row` | position in score order (`dup_of` refers to it) |
  | `instrument`, `note_index_in_instrument` | `PrettyMIDI(path).instruments[instrument].notes[note_index_in_instrument]` is the original note |
  | `t_start` | absolute start of this note's segment in the original MIDI (s), repeated on every row |
  | `orig_onset`, `orig_dur`, `pitch`, `orig_velocity` | original, onset relative to the segment's `t_start` |
  | `orig_onset_abs` | absolute original onset (s), exactly as read from the MIDI. `t_start + orig_onset` recovers it only up to float rounding, so the exact value is kept too |
  | `expr_onset`, `expr_dur`, `expr_pitch`, `expr_velocity` | expressive, onset relative to the segment's earliest expressive onset (so it also starts at 0) |
  | `dup_of` | −1, or the row whose expressive values this duplicate copies |
  | `window_first` | the first window that generated this note (for debugging seams) |

- `<piece>.segments.parquet`: `segment`, absolute `t_start` and `t_end` in the original MIDI, `n_notes`,
  `n_kept_notes`, `n_windows`, `short`, `n_onset_clipped`.
- `manifest.json`: checkpoint path, sampling settings, file list, version of this script.
- `load_mapping(piece) -> dict[(segment, orig tuple) -> expr tuple]` as a convenience. The parquet is row-aligned
  because original tuples aren't guaranteed unique within a segment.

**Save as you go:** one output file per piece, written to a temp name and then renamed. Pieces already done are skipped
on rerun. Progress goes to an unbuffered log.

## Step 5: checks before the full run

- **Unit tests** (`tests/test_expressify_symphonynet.py`):
  - segmentation at exactly 8 s;
  - relativisation round-trip;
  - `prefix_ids` positions come out unchanged;
  - for a segment of ≤ 128 notes, the sliding-window path equals a plain `generate` with the same seed;
  - duplicates map to their kept note;
  - every original note has exactly one row.
- **Dry run on 2 pieces:**
  - look for discontinuities at window seams (onset deltas across window boundaries vs. inside windows);
  - render original vs. expressive for a few segments to WAV.

### Dry-run results (job 3008816, 2026-11-09)

Pieces: `beethoven-laviano_concerto_0_woo-4` (4,844 piano notes, 5 segments) and `beethoven_piano_concerto_15_2`
(2,512 notes, 6 segments, 3 duplicates).

- Every row's `instrument` / `note_index_in_instrument` points back to a MIDI note with exactly `orig_onset_abs` and
  `pitch`. Row count equals the MIDI's piano-note count.
- No onsets were clipped.
- Window seams vs. inside windows, as log(expressive / original onset delta):
  - Beethoven WoO 4: +0.05 at seams (n = 51) vs. +0.03 inside (n = 3,788).
  - Op. 15/2: −0.11 at seams (n = 17) vs. −0.04 inside (n = 1,590).
  - There's no sign of a jump at seams, but there are few seams.
- Segment tempo vs. the score: ×0.72 to ×1.20 per segment.
  - Durations are shorter than written (median 0.25 → 0.13–0.19 s in op. 15/2).
  - 0–5 % of consecutive notes are played in the reverse order of the score.
- Rendered pairs (original / expressive) for one segment per piece are in
  `orchestration/samples/expressify_symphonynet_dryrun/`.

## Files

- `src/orchestration_v2/sampling/expressify_symphonynet.py`: extraction, segmentation, sliding window, writing.
- `src/orchestration_v2/models/expressifier.py`: the `prefix_ids` argument on `generate`.
- `scripts/expressify_symphonynet.slurm`: 1 GPU, `hai-res-main`, `--exclude=huang-l40s-2`.
- `tests/test_expressify_symphonynet.py`.

## Decisions

Joel, 2026-11-09:

1. **File set:** classical only, the 50 piano concertos above. Contemporary is not used.
2. **Multiple pianos:** every program-0/1 track is merged into one part, Mozart K. 242 included.
3. **Recovering exact notes:** every mapping row carries `t_start` (and `orig_onset_abs`), so each original note can be
   recovered without the segment table.

Still open:

4. **Checkpoint:** stage-2 `best/` (= step 1000, the lowest eval loss) is the default. Stage 2 overfit from the start, so
   stage-1 checkpoint-36000 could be compared first on a few segments.
5. **Samples per segment:** 1 by default, or K > 1 for more augmentation.
