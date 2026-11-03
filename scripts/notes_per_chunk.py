"""Per-piece note-count statistics over fixed-length chunks.

For every MIDI in a directory: tokenize per track, split into chunks, count
notes per chunk (all tracks combined), and write min / max / mean / std of
those per-chunk counts to a CSV, along with the start/end seconds of the
min and max chunks (first occurrence on ties).

Usage:
    uv run python scripts/notes_per_chunk.py ~/Desktop/symphonynet_classical notes_per_chunk.csv
"""

import csv
import statistics
import sys
from pathlib import Path

from orchestration_v2.symphonynet import chunk_track_events, midi_to_track_events

CHUNK_SECONDS = 10.0


def main(midi_dir: Path, out_csv: Path) -> None:
    rows = []
    failed = []
    for path in sorted(midi_dir.glob("*.mid")):
        try:
            sequences = midi_to_track_events(path)
        except Exception as e:  # corrupt / unsupported MIDI
            failed.append((path.name, repr(e)))
            continue
        chunks = chunk_track_events(sequences, CHUNK_SECONDS)
        if not chunks:
            failed.append((path.name, "no notes"))
            continue
        # The partial final chunk is counted like any other chunk.
        counts = [sum(len(track) for track in chunk) for chunk in chunks]
        min_idx = counts.index(min(counts))
        max_idx = counts.index(max(counts))
        rows.append(
            {
                "piece": path.name,
                "n_tracks": len(sequences),
                "n_chunks": len(chunks),
                "min": counts[min_idx],
                "min_start_s": min_idx * CHUNK_SECONDS,
                "min_end_s": (min_idx + 1) * CHUNK_SECONDS,
                "max": counts[max_idx],
                "max_start_s": max_idx * CHUNK_SECONDS,
                "max_end_s": (max_idx + 1) * CHUNK_SECONDS,
                "mean": round(statistics.fmean(counts), 3),
                "std": round(statistics.pstdev(counts), 3),
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} pieces to {out_csv}")
    for name, err in failed:
        print(f"skipped {name}: {err}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
