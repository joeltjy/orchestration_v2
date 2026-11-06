"""Expressifier dataset: memory-mapped token shards from ``preprocessing.expressifier_tokenize``.

Each item is one chunk of N <= 128 notes:
  input_ids   [4N] score tokens (dense-AMT quads, score-onset order)
  output_ids  [4N] performance tokens for the same notes (teacher-forcing source)
  labels      [4N] = output_ids, with -100 on interpolated notes (score notes the pianist did not play)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from orchestration_v2.constants import EXPRESSIFIER_TOKENS_PER_NOTE
from orchestration_v2.constants import LABEL_IGNORE_INDEX

T = EXPRESSIFIER_TOKENS_PER_NOTE


class ExpressifierDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, *, max_samples: int | None = None, seed: int = 42):
        self.root = Path(data_root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        shards = manifest["splits"].get(split, [])
        if not shards:
            raise FileNotFoundError(f"{self.root}: no '{split}' shards in manifest.json")
        self._shards = []
        counts = []
        for s in shards:
            d = self.root / s["dir"]
            self._shards.append(dict(
                inputs=np.load(d / "inputs.npy", mmap_mode="r"), outputs=np.load(d / "outputs.npy", mmap_mode="r"),
                loss_mask=np.load(d / "loss_mask.npy", mmap_mode="r"), offsets=np.load(d / "offsets.npy")))
            counts.append(int(s["n_chunks"]))
        self._cum = np.concatenate([[0], np.cumsum(counts)])
        n = int(self._cum[-1])
        self._index = np.arange(n)
        if max_samples is not None and max_samples < n:
            self._index = np.sort(np.random.default_rng(seed).choice(n, size=max_samples, replace=False))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = int(self._index[idx])
        k = int(np.searchsorted(self._cum, g, side="right") - 1)
        sh, c = self._shards[k], g - int(self._cum[k])
        a, b = int(sh["offsets"][c]), int(sh["offsets"][c + 1])
        inp = torch.from_numpy(np.asarray(sh["inputs"][a * T:b * T], dtype=np.int64))
        out = torch.from_numpy(np.asarray(sh["outputs"][a * T:b * T], dtype=np.int64))
        keep = torch.from_numpy(np.repeat(np.asarray(sh["loss_mask"][a:b], dtype=bool), T))
        labels = out.masked_fill(~keep, LABEL_IGNORE_INDEX)
        return {"input_ids": inp, "output_ids": out, "labels": labels}


def collate_expressifier_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad to the longest chunk; attention_mask marks real tokens (same length on both sides)."""
    L = max(len(x["input_ids"]) for x in items)
    B = len(items)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    output_ids = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), LABEL_IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["input_ids"])
        input_ids[i, :n], output_ids[i, :n], labels[i, :n] = x["input_ids"], x["output_ids"], x["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "output_ids": output_ids, "labels": labels}


def build_expressifier_dataloader(data_root: str | Path, split: str, *, batch_size: int, shuffle: bool,
                                  seed: int = 42, max_samples: int | None = None, num_workers: int = 0,
                                  drop_last: bool = False) -> DataLoader:
    ds = ExpressifierDataset(data_root, split, max_samples=max_samples, seed=seed)
    sampler: Any = (RandomSampler(ds, generator=torch.Generator().manual_seed(seed)) if shuffle
                    else SequentialSampler(ds))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      collate_fn=collate_expressifier_batch, drop_last=drop_last, pin_memory=True)


__all__ = ["ExpressifierDataset", "build_expressifier_dataloader", "collate_expressifier_batch"]
