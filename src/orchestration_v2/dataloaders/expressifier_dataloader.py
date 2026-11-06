"""Expressifier dataset: memory-mapped token shards from ``preprocessing.expressifier_tokenize``.

Each item is one chunk of N <= 128 notes:
  input_ids   [4N] score tokens (dense-AMT quads, score-onset order)
  output_ids  [4N] performance tokens for the same notes (teacher-forcing source)
  labels      [4N] = output_ids, with -100 on interpolated notes (score notes the pianist did not play)
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from orchestration_v2.constants import EXPRESSIFIER_TOKENS_PER_NOTE
from orchestration_v2.constants import LABEL_IGNORE_INDEX

T = EXPRESSIFIER_TOKENS_PER_NOTE
# Each open shard holds 3 memory maps (3 file descriptors). Opening every shard up front exceeded the
# 1024-fd limit on stage 1 (1190 shards); keep only the most recently used ones open, per process.
MAX_OPEN_SHARDS = 128


class ExpressifierDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, *, max_samples: int | None = None, seed: int = 42,
                 max_open_shards: int = MAX_OPEN_SHARDS):
        self.root = Path(data_root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        shards = manifest["splits"].get(split, [])
        if not shards:
            raise FileNotFoundError(f"{self.root}: no '{split}' shards in manifest.json")
        self._dirs = [self.root / s["dir"] for s in shards]
        self._offsets = [np.load(d / "offsets.npy") for d in self._dirs]  # small; read fully, file closed
        self._open: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()  # LRU of memory-mapped shards
        self._max_open = max(1, int(max_open_shards))
        counts = [int(s["n_chunks"]) for s in shards]
        self._cum = np.concatenate([[0], np.cumsum(counts)])
        n = int(self._cum[-1])
        self._index = np.arange(n)
        if max_samples is not None and max_samples < n:
            self._index = np.sort(np.random.default_rng(seed).choice(n, size=max_samples, replace=False))

    def __len__(self) -> int:
        return len(self._index)

    def _shard(self, k: int) -> dict[str, np.ndarray]:
        sh = self._open.get(k)
        if sh is not None:
            self._open.move_to_end(k)
            return sh
        while len(self._open) >= self._max_open:
            self._open.popitem(last=False)  # dropping the last reference unmaps it and closes its fd
        d = self._dirs[k]
        sh = dict(inputs=np.load(d / "inputs.npy", mmap_mode="r"), outputs=np.load(d / "outputs.npy", mmap_mode="r"),
                  loss_mask=np.load(d / "loss_mask.npy", mmap_mode="r"))
        self._open[k] = sh
        return sh

    def __getstate__(self):  # DataLoader workers start with no maps open
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        return state

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = int(self._index[idx])
        k = int(np.searchsorted(self._cum, g, side="right") - 1)
        c = g - int(self._cum[k])
        sh, off = self._shard(k), self._offsets[k]
        a, b = int(off[c]), int(off[c + 1])
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
