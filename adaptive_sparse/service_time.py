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
DEFAULT_MIN_SAMPLES = 3
MIN_SERVICE_RATE = 1e-3
MAX_SERVICE_RATE = 1e9


def estimate_service_duration_ms(
    *,
    prompt_tokens: int,
    output_tokens: int,
    fallback_ms: float,
    prefill_tokens_per_s: float | None = None,
    decode_tokens_per_s: float | None = None,
) -> float:
    """Estimate request duration without changing the no-feedback path.

    If neither measured rate is available, callers retain the planner's static
    duration exactly.  Once either rate exists, the missing direction uses the
    existing conservative default, matching the SGLang ordering hook.
    """

    for name, value in (
        ("prompt_tokens", prompt_tokens),
        ("output_tokens", output_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(fallback_ms, bool)
        or not isinstance(fallback_ms, (int, float))
        or not math.isfinite(fallback_ms)
        or fallback_ms <= 0
    ):
        raise ValueError("fallback_ms must be finite and positive")
    if prefill_tokens_per_s is None and decode_tokens_per_s is None:
        return float(fallback_ms)
    prefill_rate = (
        DEFAULT_PREFILL_TOKENS_PER_S
        if prefill_tokens_per_s is None
        else _validated_rate(prefill_tokens_per_s, "prefill_tokens_per_s")
    )
    decode_rate = (
        DEFAULT_DECODE_TOKENS_PER_S
        if decode_tokens_per_s is None
        else _validated_rate(decode_tokens_per_s, "decode_tokens_per_s")
    )
    return max(
        1.0,
        1000.0 * prompt_tokens / prefill_rate
        + 1000.0 * output_tokens / decode_rate,
    )


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
    min_samples: int

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
            "min_samples": self.min_samples,
        }


class ServiceTimeEstimator:
    """Maintain separate EWMA rates for prefill and decode service time.

    A measured rate is emitted only after ``min_samples`` valid completions.
    Until then callers should retain their existing conservative defaults.
    Samples below ``MIN_SERVICE_RATE`` or above ``MAX_SERVICE_RATE`` are clipped
    and counted so malformed or pathological timings cannot dominate the
    estimate.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        min_samples: int = DEFAULT_MIN_SAMPLES,
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
        if (
            isinstance(min_samples, bool)
            or not isinstance(min_samples, int)
            or min_samples < 1
        ):
            raise ValueError("min_samples must be a positive integer")
        self._alpha = float(alpha)
        self._min_samples = min_samples
        self._prefill = self._optional_seed(
            prefill_tokens_per_s, "prefill_tokens_per_s"
        )
        self._decode = self._optional_seed(decode_tokens_per_s, "decode_tokens_per_s")
        self._prefill_seeded = self._prefill is not None
        self._decode_seeded = self._decode is not None
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
        """Return ready feedback fields without leaking an immature EWMA."""

        with self._lock:
            result: dict[str, float] = {}
            if self._prefill is not None and (
                self._prefill_seeded or self._prefill_samples >= self._min_samples
            ):
                result["prefill_tokens_per_s"] = self._prefill
            if self._decode is not None and (
                self._decode_seeded or self._decode_samples >= self._min_samples
            ):
                result["decode_tokens_per_s"] = self._decode
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
            min_samples=self._min_samples,
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
