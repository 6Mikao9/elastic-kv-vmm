from bisect import bisect_left
from itertools import groupby


class FreeSlots:
    """Sorted, disjoint half-open ranges of free positive slot IDs."""

    def __init__(self, size):
        self.ranges = [(1, size + 1)] if size else []
        self.count = size

    def peek(self, count):
        if count < 0:
            raise ValueError("slot count must be nonnegative")
        if count > self.count:
            return None
        ranges = []
        for start, stop in self.ranges:
            if not count:
                break
            end = min(stop, start + count)
            ranges.append((start, end))
            count -= end - start
        return ranges

    def claim(self, ranges):
        consumed = 0
        for start, stop in ranges:
            self.count -= stop - start
            if stop < self.ranges[consumed][1]:
                self.ranges[consumed] = (stop, self.ranges[consumed][1])
            else:
                consumed += 1
        del self.ranges[:consumed]

    def remove(self, ranges):
        """Remove arbitrary sorted, disjoint half-open ranges.

        ``claim`` follows front-to-back allocation order. Request-affine
        placement can reserve an aligned chunk away from the first free range,
        so it needs general set subtraction.
        """

        ranges = list(ranges)
        if not ranges:
            return
        previous_stop = 0
        for start, stop in ranges:
            if start < 1 or stop <= start or start < previous_stop:
                raise ValueError(
                    "removed ranges must be positive, sorted, and disjoint"
                )
            previous_stop = stop

        free_index = 0
        for start, stop in ranges:
            while free_index < len(self.ranges) and self.ranges[free_index][1] <= start:
                free_index += 1
            if (
                free_index >= len(self.ranges)
                or self.ranges[free_index][0] > start
                or self.ranges[free_index][1] < stop
            ):
                raise ValueError("removed range crosses a non-free slot")

        result = []
        remove_index = 0
        for free_start, free_stop in self.ranges:
            cursor = free_start
            while (
                remove_index < len(ranges)
                and ranges[remove_index][1] <= cursor
            ):
                remove_index += 1
            probe = remove_index
            while probe < len(ranges) and ranges[probe][0] < free_stop:
                remove_start, remove_stop = ranges[probe]
                if remove_stop <= cursor:
                    probe += 1
                    continue
                if remove_start > cursor:
                    result.append((cursor, min(remove_start, free_stop)))
                cursor = max(cursor, min(remove_stop, free_stop))
                if remove_stop > free_stop:
                    break
                probe += 1
            if cursor < free_stop:
                result.append((cursor, free_stop))
            remove_index = probe
        self.ranges = result
        self.count -= sum(stop - start for start, stop in ranges)

    def release(self, sorted_ids):
        for _, group in groupby(enumerate(sorted_ids), lambda entry: int(entry[1]) - entry[0]):
            _, first = next(group)
            start = last = int(first)
            for _, slot in group:
                last = int(slot)
            stop = last + 1
            self.count += stop - start
            index = bisect_left(self.ranges, (start,))
            if index and self.ranges[index - 1][1] == start:
                index -= 1
                start = self.ranges.pop(index)[0]
            if index < len(self.ranges) and self.ranges[index][0] == stop:
                stop = self.ranges.pop(index)[1]
            self.ranges.insert(index, (start, stop))
