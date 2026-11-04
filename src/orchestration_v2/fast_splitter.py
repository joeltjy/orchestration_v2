"""O(n log n) drop-in replacements for the quadratic helpers in ~/orchestration's polyphony_subsegments.py.

``install(module)`` swaps them into the imported module for this process only; ~/orchestration's files are not
changed. Each replacement returns exactly what the original returns (same floats compared the same way, same
order of side effects); tests/test_fast_splitter.py checks them against the originals on random input.
"""

from __future__ import annotations

import bisect
import heapq
from collections import defaultdict

_INSTALLED_FLAG = "_orchestration_v2_fast"


def whole_life_highest_indices(notes) -> set[int]:
    """Same as ``_whole_life_highest_indices``: notes with no strictly higher-pitched note overlapping them.

    "Overlapping" is ``other.offset > note.onset and other.onset < note.offset``. Notes are inserted in onset
    order; a Fenwick tree over offsets answers "highest pitch among inserted notes with offset > x".
    """
    items = list(notes)
    if not items:
        return set()
    offsets = sorted({float(n.offset) for n in items})
    size = len(offsets)
    tree = [-1] * (size + 1)  # prefix max over reversed offset rank

    def insert(offset: float, pitch: int) -> None:
        pos = size - bisect.bisect_left(offsets, offset)  # 1-based, larger offsets first
        while pos <= size:
            if tree[pos] < pitch:
                tree[pos] = pitch
            pos += pos & -pos

    def max_with_offset_above(x: float) -> int:
        count = size - bisect.bisect_right(offsets, x)  # number of distinct offsets > x
        best, pos = -1, count
        while pos > 0:
            if tree[pos] > best:
                best = tree[pos]
            pos -= pos & -pos
        return best

    by_onset = sorted(items, key=lambda n: float(n.onset))
    highest: set[int] = set()
    k = 0
    for note in sorted(items, key=lambda n: float(n.offset)):
        end = float(note.offset)
        while k < len(by_onset) and float(by_onset[k].onset) < end:
            insert(float(by_onset[k].offset), int(by_onset[k].pitch))
            k += 1
        # The note itself may be inserted; its own pitch is not strictly higher, so it cannot cover itself.
        if max_with_offset_above(float(note.onset)) <= int(note.pitch):
            highest.add(int(note.index))
    return highest


def make_fill_voice_holes(TimedNote, clipped_ticks, fill_in_min_s):
    def fill_voice_holes(group) -> None:
        """Same as ``_fill_voice_holes``, with a sweep instead of rescanning every span for every note."""
        group_l = int(group.l)
        originals = [note for note in group.notes if note.fill_of is None]
        if not originals:
            return
        starts_by_label: dict[int, list[float]] = defaultdict(list)  # originals' spans only (static)
        span_events = []  # (start, end, label, index) of the originals' spans
        own_labels: dict[int, tuple[int, ...]] = {}
        for note in originals:
            labels = group.labels.get(int(note.index), ())
            own_labels[int(note.index)] = labels
            for label in labels:
                span_events.append((float(note.onset), float(note.offset), int(label), int(note.index)))
                starts_by_label[int(label)].append(float(note.onset))
        for v in starts_by_label.values():
            v.sort()
        span_events.sort(key=lambda s: s[0])
        next_index = max(int(note.index) for note in group.notes) + 1
        active: dict[int, int] = defaultdict(int)  # label -> spans with start <= t < end
        ends: list[tuple[float, int]] = []  # heap of (end, label) for active spans
        added = 0
        fill_ins = []
        for note in sorted(originals, key=lambda item: (float(item.onset), int(item.index))):
            onset = float(note.onset)
            while added < len(span_events) and span_events[added][0] <= onset:
                start, end, label, _ = span_events[added]
                active[label] += 1
                heapq.heappush(ends, (end, label))
                added += 1
            while ends and ends[0][0] <= onset:
                _, label = heapq.heappop(ends)
                active[label] -= 1
            labels = own_labels[int(note.index)]
            if not labels:
                continue
            primary = int(min(labels))
            if primary >= group_l:
                continue
            covering = 0
            for label, count in active.items():
                if label > primary and count:
                    covering += count
                    if label in labels and float(note.offset) > onset:  # its own span is not "another" span
                        covering -= 1
            if covering > 0:
                continue
            for layer in range(primary + 1, group_l + 1):
                starts = starts_by_label.get(layer, [])
                i = bisect.bisect_right(starts, onset)
                offset = min(float(note.offset), starts[i] if i < len(starts) else float(note.offset))
                if offset - onset < fill_in_min_s:
                    continue
                start_tick, end_tick = clipped_ticks(note, onset, offset)
                fill = TimedNote(
                    index=next_index, onset=onset, offset=offset, pitch=int(note.pitch), program=int(note.program),
                    note_id=int(note.note_id), velocity=int(note.velocity), start_tick=start_tick,
                    end_tick=end_tick, ticks_per_beat=int(note.ticks_per_beat), channel=int(note.channel),
                    fill_of=int(note.index),
                )  # fmt: skip
                fill_ins.append(fill)
                group.labels[next_index] = (layer,)
                if offset > onset:  # the original appends this span; it covers later notes at the same onset
                    active[layer] += 1
                    heapq.heappush(ends, (offset, layer))
                next_index += 1
        group.notes.extend(fill_ins)

    return fill_voice_holes


class _MaxTree:
    """Segment tree of max values with point updates and range-max queries."""

    def __init__(self, values: list[float]) -> None:
        self.n = max(1, len(values))
        self.t = [float("-inf")] * (2 * self.n)
        for i, v in enumerate(values):
            self.t[self.n + i] = v
        for i in range(self.n - 1, 0, -1):
            self.t[i] = max(self.t[2 * i], self.t[2 * i + 1])

    def set(self, i: int, v: float) -> None:
        i += self.n
        self.t[i] = v
        while i > 1:
            i //= 2
            self.t[i] = max(self.t[2 * i], self.t[2 * i + 1])

    def max(self, lo: int, hi: int) -> float:  # [lo, hi)
        res = float("-inf")
        lo += self.n
        hi += self.n
        while lo < hi:
            if lo & 1:
                res = max(res, self.t[lo])
                lo += 1
            if hi & 1:
                hi -= 1
                res = max(res, self.t[hi])
            lo //= 2
            hi //= 2
        return res


def make_absorb_surrounded_gaps(absorb_gap_fraction):
    def absorb_surrounded_gaps(groups, intervals) -> None:
        """Same as ``_absorb_surrounded_gaps``.

        The original restarts its scan (lowest level first) after every merge. A merge at level L only extends
        the left group and removes the right one, so other levels' pairs can only become more blocked, and at
        level L only the pair just before the merge can newly qualify. So one pointer pass per level, stepping
        back one pair after each merge, merges the same pairs in the same order.
        """
        if len(groups) < 2:
            return
        order = sorted(range(len(groups)), key=lambda i: (float(groups[i].onset), int(groups[i].l)))
        pos_of = {gi: p for p, gi in enumerate(order)}
        onsets = [float(groups[gi].onset) for gi in order]
        tree = _MaxTree([float(groups[gi].offset) for gi in order])
        removed: set[int] = set()
        iv_onsets = [float(iv.onset) for iv in intervals]  # sweep intervals are in time order
        scanned: dict[int, int] = {}  # group -> first interval index its locking scan has not passed

        def gap_has_other(li: int, ri: int) -> bool:
            start, end = float(groups[li].offset), float(groups[ri].onset)
            if end <= start:
                return False
            hi = bisect.bisect_left(onsets, end)
            a, b = sorted((pos_of[li], pos_of[ri]))
            best = float("-inf")
            for lo, up in ((0, min(a, hi)), (a + 1, min(b, hi)), (b + 1, hi)):
                if lo < up:
                    best = max(best, tree.max(lo, up))
            return best > start

        by_level: dict[int, list[int]] = defaultdict(list)
        for gi in order:
            by_level[int(groups[gi].l)].append(gi)
        for level in sorted(by_level):
            same = by_level[level]
            p = 0
            while p < len(same) - 1:
                left, right = groups[same[p]], groups[same[p + 1]]
                gap = float(right.onset) - float(left.offset)
                ok = gap > 0.0
                if ok:
                    combined = float(left.duration) + float(right.duration)
                    ok = gap < absorb_gap_fraction * combined and not gap_has_other(same[p], same[p + 1])
                if not ok:
                    p += 1
                    continue
                left.offset = float(right.offset)
                left.discard_overflow = True
                # lock every interval contained in left (as the original does). Each group remembers where its
                # previous scan stopped, so every interval is visited about once in total.
                j = scanned.get(same[p])
                if j is None:
                    j = bisect.bisect_left(iv_onsets, float(left.onset))
                while j < len(intervals) and float(intervals[j].onset) < float(left.offset):
                    iv = intervals[j]
                    if float(iv.onset) >= float(left.onset) and float(iv.offset) <= float(left.offset):
                        iv.locked = True
                    elif float(iv.offset) > float(left.offset):
                        break  # straddles the end: not contained yet; rescan it after the next extension
                    j += 1
                scanned[same[p]] = j
                tree.set(pos_of[same[p]], float(left.offset))
                tree.set(pos_of[same[p + 1]], float("-inf"))
                removed.add(id(right))
                del same[p + 1]
                p = max(p - 1, 0)
        groups[:] = [g for g in groups if id(g) not in removed]

    return absorb_surrounded_gaps


def install(module) -> None:
    """Swap the fast versions into an imported polyphony_subsegments module (this process only)."""
    if getattr(module, _INSTALLED_FLAG, False):
        return
    module._whole_life_highest_indices = whole_life_highest_indices
    module._fill_voice_holes = make_fill_voice_holes(module.TimedNote, module._clipped_ticks, module.FILL_IN_MIN_S)
    module._absorb_surrounded_gaps = make_absorb_surrounded_gaps(module.ABSORB_GAP_FRACTION)
    setattr(module, _INSTALLED_FLAG, True)
