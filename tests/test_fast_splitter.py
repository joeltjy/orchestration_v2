"""The fast splitter helpers must match ~/orchestration's originals exactly (skipped if it is not available)."""

import copy
import random
import sys
from pathlib import Path

import pytest

ORCH = Path.home() / "orchestration" / "src"
if not ORCH.exists():
    pytest.skip("~/orchestration not available", allow_module_level=True)
sys.path.insert(0, str(ORCH))
ps = pytest.importorskip("orchestration.reduction.polyphony_subsegments")

from orchestration_v2 import fast_splitter as fs  # noqa: E402

ORIG_WHOLE = ps._whole_life_highest_indices
ORIG_FILL = ps._fill_voice_holes
ORIG_ABSORB = ps._absorb_surrounded_gaps


def _notes(rng, n, grid=True):
    out = []
    for i in range(n):
        on = rng.randint(0, 40) * 0.25 if grid else rng.uniform(0, 10)
        dur = rng.choice([0.0, 0.25, 0.5, 1.0, 2.0]) if grid else rng.uniform(0, 2)
        out.append(ps.TimedNote(index=i, onset=on, offset=on + dur, pitch=rng.randint(55, 70), program=40,
                                start_tick=int(on * 480), end_tick=int((on + dur) * 480),
                                ticks_per_beat=480))  # fmt: skip
    return out


def test_whole_life_highest_matches():
    rng = random.Random(0)
    for trial in range(400):
        notes = _notes(rng, rng.randint(0, 30), grid=trial % 2 == 0)
        assert fs.whole_life_highest_indices(notes) == ORIG_WHOLE(notes)


def test_fill_voice_holes_matches():
    rng = random.Random(1)
    fast = fs.make_fill_voice_holes(ps.TimedNote, ps._clipped_ticks, ps.FILL_IN_MIN_S)
    for trial in range(400):
        notes = _notes(rng, rng.randint(1, 25), grid=trial % 2 == 0)
        level = rng.randint(1, 4)
        labels = {n.index: (rng.randint(1, level),) for n in notes if rng.random() < 0.9}
        g1 = ps.SubsegmentGroup(onset=0.0, offset=12.0, l=level, notes=list(notes), labels=dict(labels))
        g2 = copy.deepcopy(g1)
        ORIG_FILL(g1)
        fast(g2)
        assert [(n.index, n.onset, n.offset, n.fill_of) for n in g1.notes] == \
            [(n.index, n.onset, n.offset, n.fill_of) for n in g2.notes]  # fmt: skip
        assert g1.labels == g2.labels


def test_absorb_matches():
    rng = random.Random(2)
    fast = fs.make_absorb_surrounded_gaps(ps.ABSORB_GAP_FRACTION)
    for _ in range(600):
        intervals, t = [], 0.0
        for _ in range(rng.randint(2, 40)):
            d = rng.choice([0.25, 0.5, 1.0, 1.5])
            intervals.append(ps.ConcurrentInterval(onset=t, offset=t + d, k=rng.randint(1, 3)))
            t += d
        groups, t = [], 0.0
        while t < intervals[-1].offset:
            start = t + rng.choice([0.0, 0.25, 0.5, 1.0, 2.0])
            end = start + rng.choice([0.5, 1.0, 2.0, 3.0])
            groups.append(ps.SubsegmentGroup(onset=start, offset=end, l=rng.randint(1, 3)))
            t = end if rng.random() < 0.7 else start + 0.25
        a_groups, a_iv = copy.deepcopy(groups), copy.deepcopy(intervals)
        b_groups, b_iv = copy.deepcopy(groups), copy.deepcopy(intervals)
        ORIG_ABSORB(a_groups, a_iv)
        fast(b_groups, b_iv)
        assert [(g.onset, g.offset, g.l, g.discard_overflow) for g in a_groups] == \
            [(g.onset, g.offset, g.l, g.discard_overflow) for g in b_groups]  # fmt: skip
        assert [iv.locked for iv in a_iv] == [iv.locked for iv in b_iv]
