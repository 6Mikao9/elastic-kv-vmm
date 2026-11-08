"""Pure-CPU migration bandwidth estimation and time-cap admission."""

import math


class MigrationBandwidthEstimator:
    """Track direction-specific transfer throughput from committed migrations.

    The configured byte cap remains a hard upper bound.  A positive time cap may
    lower it to the bytes that the observed bandwidth can transfer within that
    wall-clock budget.  A conservative configured rate is used until a real
    nonzero sample exists; EWMA then follows successful transfers only.
    """

    DIRECTIONS = ("demotion", "promotion")

    def __init__(self, initial_bandwidth_bytes_per_s, ewma_alpha):
        if (
            isinstance(initial_bandwidth_bytes_per_s, bool)
            or not isinstance(initial_bandwidth_bytes_per_s, int)
            or initial_bandwidth_bytes_per_s <= 0
        ):
            raise ValueError("initial bandwidth must be a positive integer")
        if (
            isinstance(ewma_alpha, bool)
            or not isinstance(ewma_alpha, (int, float))
            or not math.isfinite(ewma_alpha)
            or not 0 < ewma_alpha <= 1
        ):
            raise ValueError("EWMA alpha must be in (0, 1]")
        self.initial_bandwidth_bytes_per_s = initial_bandwidth_bytes_per_s
        self.ewma_alpha = float(ewma_alpha)
        self._rates = {}
        self._samples = {direction: 0 for direction in self.DIRECTIONS}

    @classmethod
    def _check_direction(cls, direction):
        if direction not in cls.DIRECTIONS:
            raise ValueError(f"unknown migration direction: {direction}")

    def observe(self, direction, transfer_bytes, elapsed_seconds):
        """Update one direction and return the sampled bandwidth or ``None``."""

        self._check_direction(direction)
        if (
            isinstance(transfer_bytes, bool)
            or not isinstance(transfer_bytes, int)
            or transfer_bytes < 0
        ):
            raise ValueError("transfer bytes must be a nonnegative integer")
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(elapsed_seconds)
        ):
            raise ValueError("elapsed seconds must be finite")
        if transfer_bytes == 0 or elapsed_seconds <= 0:
            return None
        sample = transfer_bytes / float(elapsed_seconds)
        previous = self._rates.get(direction)
        if previous is None:
            self._rates[direction] = sample
        else:
            alpha = self.ewma_alpha
            self._rates[direction] = alpha * sample + (1 - alpha) * previous
        self._samples[direction] += 1
        return sample

    def bandwidth(self, direction):
        self._check_direction(direction)
        return self._rates.get(
            direction, float(self.initial_bandwidth_bytes_per_s)
        )

    def samples(self, direction):
        self._check_direction(direction)
        return self._samples[direction]

    def effective_byte_cap(self, byte_cap, time_cap_ms, direction="demotion"):
        """Return the most bytes allowed by both byte and time admission."""

        self._check_direction(direction)
        if (
            isinstance(byte_cap, bool)
            or not isinstance(byte_cap, int)
            or byte_cap < 1
        ):
            raise ValueError("byte cap must be a positive integer")
        if (
            isinstance(time_cap_ms, bool)
            or not isinstance(time_cap_ms, int)
            or time_cap_ms < 0
        ):
            raise ValueError("time cap must be a nonnegative integer")
        if time_cap_ms == 0:
            return byte_cap
        transferable = int(self.bandwidth(direction) * time_cap_ms / 1000.0)
        return min(byte_cap, max(0, transferable))

    def snapshot(self):
        return {
            direction: self.bandwidth(direction)
            for direction in self.DIRECTIONS
        }

    def sample_counts(self):
        return dict(self._samples)
