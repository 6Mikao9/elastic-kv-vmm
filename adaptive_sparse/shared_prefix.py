"""Reference-counted registry for immutable GPU prefix KV.

The registry deliberately stores metadata instead of tensor data.  SGLang's
request table already maps both requests onto the same logical slots when a
prefix hit is installed; this layer records which live full-resident request
is a safe physical source for a host-backed HiSparse request.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import torch


@dataclass(frozen=True, slots=True)
class PrefixClaim:
    """Validated identity claimed by one request."""

    group: str
    tokens: int
    fingerprint: bytes


@dataclass(frozen=True, slots=True)
class SharedPrefixSource:
    """A live resident request that owns the claimed physical prefix pages."""

    row: int
    request: Any
    claim: PrefixClaim


def complete_shared_blocks(
    block_ids: Iterable[int],
    *,
    seq_len: int,
    shared_tokens: int,
    block_size: int,
) -> tuple[bool, ...]:
    """Return whether each selected block lies wholly in the shared prefix.

    A partial block is never shared because attention addressing and VMM
    ownership are page-granular.  Invalid block ids are left to the native
    validator and never become shared hits here.
    """

    for value, name, minimum in (
        (seq_len, "seq_len", 1),
        (shared_tokens, "shared_tokens", 0),
        (block_size, "block_size", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} is outside the supported range")
    result = []
    for block in block_ids:
        if isinstance(block, bool) or not isinstance(block, int):
            raise ValueError("block ids must be integers")
        start = block * block_size
        result.append(
            block >= 0
            and start + block_size <= shared_tokens
            and start + block_size <= seq_len
        )
    return tuple(result)


def resolve_shared_block_locations(
    block_ids: torch.Tensor,
    *,
    source_row: int,
    seq_len: torch.Tensor,
    shared_tokens: int,
    block_size: int,
    request_table: torch.Tensor,
    device_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve complete shared blocks to the source request's device slots.

    The returned mask excludes invalid ids, incomplete prefix blocks, blocks
    beyond the consumer sequence, and source logical slots that have no live
    device mapping.  The second result is shaped ``[blocks, block_size]``.
    """

    if block_ids.ndim != 1:
        raise ValueError("block ids must be one-dimensional")
    if (
        isinstance(source_row, bool)
        or not isinstance(source_row, int)
        or source_row < 0
        or source_row >= request_table.shape[0]
    ):
        raise ValueError("source row is outside the request table")
    if request_table.ndim != 2 or not request_table.shape[1] or not device_mapping.numel():
        raise ValueError("shared prefix address geometry is empty")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
        or isinstance(shared_tokens, bool)
        or not isinstance(shared_tokens, int)
        or shared_tokens < 0
    ):
        raise ValueError("invalid shared block geometry")
    blocks = block_ids.to(dtype=torch.long)
    starts = blocks * block_size
    ends = starts + block_size
    valid = (blocks >= 0) & (ends <= shared_tokens) & (ends <= seq_len)
    positions = starts.unsqueeze(1) + torch.arange(
        block_size, device=block_ids.device, dtype=torch.long
    )
    safe_positions = positions.clamp(0, request_table.shape[1] - 1)
    logical = request_table[source_row, safe_positions]
    safe_logical = logical.clamp(0, device_mapping.numel() - 1)
    physical = device_mapping[safe_logical]
    valid = valid & (logical >= 0).all(dim=1) & (physical > 0).all(dim=1)
    return valid, physical.to(torch.int32)


def _prepare_prefix_value(value: Any) -> Any:
    """Make adapter and salt identities comparable in a stable way."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif not isinstance(value, (bytes, bytearray)):
        try:
            value = list(value)
        except TypeError:
            pass
    return repr(value)


def _token_ids(request: Any) -> list[int]:
    ids = getattr(request, "origin_input_ids", None)
    if ids is None:
        raise ValueError("origin_input_ids is unavailable")
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    else:
        ids = list(ids)
    result = []
    for token in ids:
        if isinstance(token, bool):
            raise ValueError("origin_input_ids must contain integers")
        try:
            converted = int(token)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("origin_input_ids must contain integers") from exc
        if converted != token:
            raise ValueError("origin_input_ids must contain integers")
        result.append(converted)
    return result


def make_prefix_claim(
    request: Any,
    *,
    group: str | None,
    shared_tokens: int,
) -> PrefixClaim | None:
    """Build a safe identity or return ``None`` for a non-sharing request.

    Input-embedding and multimodal requests are conservatively excluded: equal
    token ids do not establish equal prefill state for those modalities.
    """

    if shared_tokens == 0:
        return None
    if not isinstance(group, str) or not group or len(group) > 64:
        raise ValueError("prefix group must be a short non-empty string")
    if (
        isinstance(shared_tokens, bool)
        or not isinstance(shared_tokens, int)
        or shared_tokens < 1
    ):
        raise ValueError("shared prefix tokens must be positive")
    if getattr(request, "multimodal_inputs", None) is not None:
        return None
    if getattr(request, "input_embeds", None) is not None:
        return None

    token_ids = _token_ids(request)
    if shared_tokens > len(token_ids):
        raise ValueError("shared prefix exceeds the request prompt")
    digest = hashlib.sha256()
    digest.update(group.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shared_tokens.to_bytes(8, "big"))
    for name in (
        "extra_key",
        "cache_salt",
        "lora_id",
        "lora_path",
        "adapter_id",
    ):
        digest.update(name.encode("ascii"))
        digest.update(repr(_prepare_prefix_value(getattr(request, name, None))).encode())
    digest.update(b"\0tokens\0")
    for token in token_ids[:shared_tokens]:
        digest.update(token.to_bytes(8, "big", signed=True))
    return PrefixClaim(group, shared_tokens, digest.digest())


class SharedPrefixRegistry:
    """Process-local ownership and lookup for immutable resident prefixes.

    All mutation and lookup occurs on the scheduler thread at an eager safe
    boundary.  Consumers do not take an extra GPU reference: residency of the
    source request protects its physical pages, and removal of the source makes
    the next Top-K lookup fall back to native host staging.
    """

    def __init__(self) -> None:
        self._claims: dict[int, PrefixClaim] = {}
        self._sources: dict[int, SharedPrefixSource] = {}
        self.stats = {
            "claims": 0,
            "sources_registered": 0,
            "sources_removed": 0,
            "lookups": 0,
            "shared_hits": 0,
            "missing_sources": 0,
            "staging_sources": 0,
            "inconsistent_group": 0,
        }

    def register_claim(self, row: int, claim: PrefixClaim) -> None:
        self._validate_row(row)
        self._claims[row] = claim
        self.stats["claims"] += 1

    def register_source(self, row: int, request: Any, claim: PrefixClaim) -> None:
        self._validate_row(row)
        self._claims[row] = claim
        replaced = row in self._sources
        self._sources[row] = SharedPrefixSource(row, request, claim)
        if not replaced:
            self.stats["sources_registered"] += 1

    def remove_source(self, row: int) -> None:
        if self._sources.pop(row, None) is not None:
            self.stats["sources_removed"] += 1

    def remove_request(self, row: int) -> None:
        self.remove_source(row)
        self._claims.pop(row, None)

    def claim_for(self, row: int) -> PrefixClaim | None:
        self._validate_row(row)
        return self._claims.get(row)

    def resolve(self, row: int) -> SharedPrefixSource | None:
        """Return a deterministic live source for ``row`` or ``None``."""

        self._validate_row(row)
        claim = self._claims.get(row)
        if claim is None:
            return None
        self.stats["lookups"] += 1
        inconsistent = False
        for source in self._sources.values():
            if source.row == row or source.claim.group != claim.group:
                continue
            if source.claim.tokens != claim.tokens:
                inconsistent = True
                continue
            if source.claim.fingerprint != claim.fingerprint:
                inconsistent = True
                continue
            if getattr(source.request, "hisparse_staging", False):
                self.stats["staging_sources"] += 1
                continue
            self.stats["shared_hits"] += 1
            return source
        if inconsistent:
            self.stats["inconsistent_group"] += 1
        else:
            self.stats["missing_sources"] += 1
        return None

    def snapshot(self) -> dict[str, int]:
        result = dict(self.stats)
        result["active_claims"] = len(self._claims)
        result["active_sources"] = len(self._sources)
        return result

    @staticmethod
    def _validate_row(row: int) -> None:
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise ValueError("request row must be a nonnegative integer")
