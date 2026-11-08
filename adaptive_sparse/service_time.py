"""Bounded, closed-loop service-time estimates for rollout scheduling.

The estimator consumes the latency fields already exported by the live rollout
executor.  It is deliberately independent of CUDA and SGLang so that its
filtering and update rules can be tested without changing serving state.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

DEFAULT_PREFILL_TOKENS_PER_S = 8_000.0
DEFAULT_DECODE_TOKENS_PER_S = 20_000.0
MIN_SERVICE_RATE = 1e-3
MAX_SERVICE_RATE = 1e9


@dataclass(frozen=True, slots=True)
class ServiceTimeSnapshot:
    """One lock-free view of the estimator's current calibration state."""

    prefill_tokens_per_s: float | None
    decode_tokens_per_s: float | None
    prefill_samples: int
    decode_samples: int
    clipped_prefill_samples: int
    clipped_decode_samples: int
    rejected_samples: int
    alpha: float

    def as_dict(self) -> dict[str, object]:
        return {
            "prefill_tokens_per_s": self.prefill_tokens_per_s,
            "decode_tokens_per_s": self.decode_tokens_per_s,
            "prefill_samples": self.prefill_samples,
            "decode_samples": self.decode_samples,
            "clipped_prefill_samples": self.clipped_prefill_samples,
            "clipped_decode_samples": self.clipped_decode_samples,
            "rejected_samples": self.rejected_samples,
            "alpha": self.alpha,
        }


class ServiceTimeEstimator:
    """Maintain separate EWMA rates for prefill and decode service time.

    A rate is emitted only after at least one valid completion sample.  Until
    then callers should retain their existing conservative defaults.  Samples
    below ``MIN_SERVICE_RATE`` or above ``MAX_SERVICE_RATE`` are clipped and
    counted so malformed or pathological timings cannot dominate the estimate.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        prefill_tokens_per_s: float | None = None,
        decode_tokens_per_s: float | None = None,
    ) -> None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or not 0.0 < alpha <= 1.0
        ):
            raise ValueError("alpha must be finite and in (0, 1]")
        self._alpha = float(alpha)
        self._prefill = self._optional_seed(
            prefill_tokens_per_s, "prefill_tokens_per_s"
        )
        self._decode = self._optional_seed(decode_tokens_per_s, "decode_tokens_per_s")
        self._prefill_samples = 0
        self._decode_samples = 0
        self._clipped_prefill_samples = 0
        self._clipped_decode_samples = 0
        self._rejected_samples = 0
        self._lock = threading.Lock()

    @staticmethod
    def _optional_seed(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        return _validated_rate(value, name)

    def observe(
        self,
        *,
        prompt_tokens: int,
        output_tokens: int,
        ttft_s: float,
        tpot_s: float,
    ) -> ServiceTimeSnapshot:
        """Update both directions from one successful completion."""

        for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("ttft_s", ttft_s), ("tpot_s", tpot_s)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

        prefill_sample, prefill_clipped = _clip_rate(prompt_tokens / float(ttft_s))
        decode_sample, decode_clipped = _clip_rate(1.0 / float(tpot_s))
        with self._lock:
            self._prefill = _ewma(self._prefill, prefill_sample, self._alpha)
            self._decode = _ewma(self._decode, decode_sample, self._alpha)
            self._prefill_samples += 1
            self._decode_samples += 1
            self._clipped_prefill_samples += int(prefill_clipped)
            self._clipped_decode_samples += int(decode_clipped)
            return self._snapshot_locked()

    def record_rejected_sample(self) -> None:
        """Count a malformed or unusable completion without changing rates."""

        with self._lock:
            self._rejected_samples += 1

    def rates(self) -> dict[str, float]:
        """Return only fields that have a measured or seeded value."""

        snapshot = self.snapshot()
        result: dict[str, float] = {}
        if snapshot.prefill_tokens_per_s is not None:
            result["prefill_tokens_per_s"] = snapshot.prefill_tokens_per_s
        if snapshot.decode_tokens_per_s is not None:
            result["decode_tokens_per_s"] = snapshot.decode_tokens_per_s
        return result

    def snapshot(self) -> ServiceTimeSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> ServiceTimeSnapshot:
        return ServiceTimeSnapshot(
            prefill_tokens_per_s=self._prefill,
            decode_tokens_per_s=self._decode,
            prefill_samples=self._prefill_samples,
            decode_samples=self._decode_samples,
            clipped_prefill_samples=self._clipped_prefill_samples,
            clipped_decode_samples=self._clipped_decode_samples,
            rejected_samples=self._rejected_samples,
            alpha=self._alpha,
        )


def _validated_rate(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not MIN_SERVICE_RATE <= float(value) <= MAX_SERVICE_RATE
    ):
        raise ValueError(
            f"{name} must be finite and in [{MIN_SERVICE_RATE}, {MAX_SERVICE_RATE}]"
        )
    return float(value)


def _clip_rate(value: float) -> tuple[float, bool]:
    if not math.isfinite(value):
        return MAX_SERVICE_RATE, True
    clipped = min(MAX_SERVICE_RATE, max(MIN_SERVICE_RATE, value))
    return clipped, clipped != value


def _ewma(current: float | None, sample: float, alpha: float) -> float:
    if current is None:
        return sample
    return alpha * sample + (1.0 - alpha) * current
