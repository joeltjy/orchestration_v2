"""Run the v1 reduction over the corpus and write every output.

Outputs (in OUT_DIR, default reductions_v1/):
    note_quads_per_chunk_v1.csv   note quads per second per 10 s chunk, after steps 0, 4, 5, 6, 7
    midi/<piece>_reduced.mid      the reduction after step 7 (merged notes unpacked), same tempo map as the original
    midi/<piece>_side_by_side.mid original tracks ("orig: ...") each followed by their reduced track ("red: ...")
    viewer.html + data/           piano-roll viewer: open viewer.html in a browser
    bpe_log.csv                   every BPE iteration (class, instances, tokens created)
    patterns.csv                  every repeated pattern found in step 5, with its corpus-wide count
    run_log.txt                   progress and timings
    corpus.pkl                    the reduced corpus, so outputs can be redone with --from-pickle (no re-reduction)

Usage:
    uv run python scripts/run_reduction_v1.py [--limit N] [--out reductions_v1] [--budget 10000] [--min-count 50]
"""

import argparse
import csv
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from orchestration_v2.reduction_v1 import MEASURE_STEPS, Corpus, Piece, tokenize_corpus
from orchestration_v2.reduction_v1_outputs import (
    write_reduced_midi,
    write_side_by_side_midi,
    write_viewer,
    write_viewer_data,
    write_vocabulary,
)

REPO = Path(__file__).resolve().parents[1]
_WORKER_CORPUS: Corpus | None = None


def _init_worker(class_defs: dict) -> None:
    global _WORKER_CORPUS
    _WORKER_CORPUS = Corpus([], class_defs=class_defs)


def _write_piece(job: tuple[int, Piece, Path]) -> dict:
    """All outputs for one piece (runs in a worker; only needs the merge class definitions)."""
    k, piece, out = job
    assert _WORKER_CORPUS is not None
    stem = piece.name.removesuffix(".mid")
    write_reduced_midi(_WORKER_CORPUS, piece, out / "midi" / f"{stem}_reduced.mid")
    write_side_by_side_midi(_WORKER_CORPUS, piece, out / "midi" / f"{stem}_side_by_side.mid")
    return write_viewer_data(_WORKER_CORPUS, piece, k, out / "data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path.home() / "Desktop" / "symphonynet_classical")
    ap.add_argument("--outliers", type=Path, default=REPO / "to_analyze" / "to_analyze.csv")
    ap.add_argument("--out", type=Path, default=REPO / "reductions_v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--from-pickle", type=Path, default=None, help="skip the reduction; load this corpus.pkl")
    ap.add_argument("--pieces", type=Path, default=None, help="only these pieces (one file name per line)")
    args = ap.parse_args()

    out = args.out
    (out / "midi").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    log = (out / "run_log.txt").open("a")
    start = time.time()

    def progress(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - start:6.0f}s] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if args.from_pickle:
        progress(f"loading {args.from_pickle}")
        with args.from_pickle.open("rb") as f:
            corpus = pickle.load(f)
    else:
        progress(f"start: corpus {args.corpus}, budget {args.budget}, min_count {args.min_count}, limit {args.limit}")
        corpus = tokenize_corpus(args.corpus, args.outliers, out / "note_quads_per_chunk_v1.csv",
                                 workers=args.workers, token_budget=args.budget, min_count=args.min_count,
                                 progress=progress, limit=args.limit,
                                 only=set(args.pieces.read_text().split()) if args.pieces else None)  # fmt: skip
        progress(f"reduction done: {len(corpus.pieces)} pieces, {len(corpus.merges.entries)} merged tokens")
        for piece in corpus.pieces:  # voice lists are not needed after BPE; drop them before pickling
            piece.voices = {}
        with (out / "corpus.pkl").open("wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
        progress(f"saved {out / 'corpus.pkl'}")

    with (out / "bpe_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corpus.bpe_log[0]) if corpus.bpe_log else ["step"])
        w.writeheader()
        w.writerows(corpus.bpe_log)
    with (out / "patterns.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pitch_order", "relative_gaps_12ths", "count", "example_onset_beats_pitch"])
        for e in sorted(corpus.patterns.entries.values(), key=lambda e: -e.count):
            w.writerow([" ".join(map(str, e.pitch_order)), " ".join(map(str, e.rounded_gaps)), e.count,
                        " ".join(f"{o}:{p}" for o, p in e.example)])  # fmt: skip

    index = []
    jobs = [(k, piece, out) for k, piece in enumerate(corpus.pieces)]
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(corpus.class_defs,)) as pool:
        for k, entry in enumerate(pool.map(_write_piece, jobs, chunksize=1), start=1):
            index.append(entry)
            if k % 50 == 0 or k == len(jobs):
                progress(f"outputs: {k}/{len(jobs)} pieces")
    write_viewer(index, out)
    write_vocabulary(corpus, index, out / "data")
    totals = {s: sum(e[f"q{s}"] for e in index) for s in MEASURE_STEPS}
    progress("note quads over the corpus: " + ", ".join(f"step {s}: {totals[s]}" for s in MEASURE_STEPS))
    progress(f"finished in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
