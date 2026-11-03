"""Per-piece tokens/second statistics over fixed-length chunks.

For every MIDI in a directory: tokenize per track, split into chunks, count
tokens per chunk (all tracks combined), divide by the chunk length, and write
min / max / mean / std of those per-chunk rates to a CSV.

Usage:
    uv run python scripts/tokens_per_second.py ~/Desktop/symphonynet_classical tokens_per_second.csv
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
        # Every chunk, including the partial final one, is divided by the full chunk length.
        rates = [sum(len(track) for track in chunk) / CHUNK_SECONDS for chunk in chunks]
        rows.append(
            {
                "piece": path.name,
                "n_tracks": len(sequences),
                "n_chunks": len(chunks),
                "min": min(rates),
                "max": max(rates),
                "mean": round(statistics.fmean(rates), 3),
                "std": round(statistics.pstdev(rates), 3),
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
