"""Generate performances for expressifier test chunks and write MIDI (score / generated / reference).

    python -m orchestration_v2.sampling.expressifier_generate --checkpoint .../expressifier_stage2/best \
        --data-root /data/scratch-oc40/joel/expressifier_stage2_aligned_maestro_aug16 --num-chunks 16 \
        --output-dir /data/scratch-oc40/joel/samples/expressifier_stage2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orchestration_v2.constants import (
    AMT_DENSE_DURATION_OFFSET,
    AMT_DENSE_NOTE_OFFSET,
    AMT_DENSE_VELOCITY_OFFSET,
    EXPRESSIFIER_TOKENS_PER_NOTE,
)


def tokens_to_notes(tokens: np.ndarray) -> list[tuple[float, float, int, int]]:
    """Dense-AMT quads (relative onset) -> (onset s, offset s, pitch, velocity)."""
    q = np.asarray(tokens, dtype=np.int64).reshape(-1, EXPRESSIFIER_TOKENS_PER_NOTE)
    out = []
    for on, du, nt, ve in q:
        onset = on / 100.0
        out.append((onset, onset + max(1, du - AMT_DENSE_DURATION_OFFSET) / 100.0,
                    int((nt - AMT_DENSE_NOTE_OFFSET) % 128), int(np.clip(ve - AMT_DENSE_VELOCITY_OFFSET, 1, 127))))
    return out


def write_midi(notes: list[tuple[float, float, int, int]], path: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="piano")
    inst.notes = [pretty_midi.Note(v, p, s, e) for s, e, p, v in notes]
    pm.instruments.append(inst)
    pm.write(str(path))


def main(argv=None) -> None:
    import torch

    from orchestration_v2.dataloaders.expressifier_dataloader import ExpressifierDataset, collate_expressifier_batch
    from orchestration_v2.models.expressifier import load_expressifier_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-chunks", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_expressifier_checkpoint(args.checkpoint, device=device)
    ds = ExpressifierDataset(args.data_root, args.split, max_samples=args.num_chunks, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    summary = []
    for k in range(len(ds)):
        batch = {n: t.to(device) for n, t in collate_expressifier_batch([ds[k]]).items()}
        out = model.generate(batch["input_ids"], batch["attention_mask"], temperature=args.temperature,
                             top_p=args.top_p, generator=gen)[0].cpu().numpy()
        d = args.output_dir / f"chunk_{k:03d}"
        d.mkdir(exist_ok=True)
        score = tokens_to_notes(batch["input_ids"][0].cpu().numpy())
        write_midi(score, d / "score.mid")
        write_midi(tokens_to_notes(out), d / "generated.mid")
        write_midi(tokens_to_notes(batch["output_ids"][0].cpu().numpy()), d / "reference.mid")
        summary.append(dict(chunk=k, n_notes=len(score)))
        print(f"chunk {k}: {len(score)} notes -> {d}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
