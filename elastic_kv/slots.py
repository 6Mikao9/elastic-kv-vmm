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
