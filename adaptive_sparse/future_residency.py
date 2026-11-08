"""Known-future policy for claiming full-GPU KV residency slots.

The scheduler already uses rollout hints to order the waiting queue.  This
module applies the same metadata at the memory admission point so a known
background or short request does not consume the bounded full-resident pool
needed by a later rollout.

The policy is pure and deterministic.  It does not touch CUDA, allocator
state, or SGLang objects; the caller supplies the observed KV length and
whether a resident slot and physical reserve are currently available.
"""

from __future__ import annotations

from dataclasses import dataclass


RESIDENCE_HINTS = ("auto", "resident", "host_backed")


@dataclass(frozen=True, slots=True)
class ResidentDecision:
    claim: bool
    reason: str


def decide_resident_claim(
    *,
    sequence_tokens: int,
    native_buffer_tokens: int,
    slots_available: bool,
    hint_residence: str | None = None,
    expected_output_tokens: int | None = None,
    minimum_hint_output_tokens: int = 256,
) -> ResidentDecision:
    """Decide whether a request should keep its complete GPU KV history.

    ``hint_residence`` is ``None`` for legacy traffic.  No hint preserves the
    historical long-request default.  For hinted traffic, an explicit
    ``host_backed`` request yields the resident slot, while ``resident`` claims
    it when capacity exists.  ``auto`` uses the predicted output length: long
    rollouts justify full residency, while short jobs will usually spend more
    time migrating than reading the retained history.
    """

    if sequence_tokens < 1 or native_buffer_tokens < 1:
        raise ValueError("sequence token counts must be positive")
    if minimum_hint_output_tokens < 1:
        raise ValueError("minimum_hint_output_tokens must be positive")
    if expected_output_tokens is not None and expected_output_tokens < 1:
        raise ValueError("expected_output_tokens must be positive")
    if hint_residence is not None and hint_residence not in RESIDENCE_HINTS:
        raise ValueError(f"unknown residence hint: {hint_residence!r}")

    if sequence_tokens <= native_buffer_tokens:
        return ResidentDecision(False, "fits-native-hot-set")
    if not slots_available:
        return ResidentDecision(False, "no-resident-slot")
    if hint_residence is None:
        return ResidentDecision(True, "long-request-default")
    if hint_residence == "host_backed":
        return ResidentDecision(False, "hint-host-backed")
    if hint_residence == "resident":
        return ResidentDecision(True, "hint-resident")
    if expected_output_tokens is None:
        return ResidentDecision(True, "hint-auto-missing-output")
    if (
        expected_output_tokens < minimum_hint_output_tokens
    ):
        return ResidentDecision(False, "hint-short-auto")
    return ResidentDecision(True, "hint-long-auto")
