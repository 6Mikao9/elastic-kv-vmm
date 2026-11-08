"""CUDA VMM backing with stable torch views and explicit residency control."""

import ctypes as ct
import functools
import operator
import os
import time
import weakref
from pathlib import Path

import torch

from .global_broker import GlobalAdmissionBroker

DEFAULT_CHUNK_BYTES = 2 << 20
_INVALID = (1 << 64) - 1


@functools.lru_cache(maxsize=None)
def _load_library(path):
    lib = ct.CDLL(path)
    lib.ev_create.argtypes = [ct.c_int, ct.c_uint64, ct.c_uint64]
    lib.ev_create.restype = ct.c_void_p
    for name in ("ev_ensure", "ev_precommit", "ev_drop"):
        fn = getattr(lib, name)
        fn.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_size_t]
        fn.restype = ct.c_int
    for name in ("ev_base", "ev_backed", "ev_reserved", "ev_granularity"):
        fn = getattr(lib, name)
        fn.argtypes = [ct.c_void_p]
        fn.restype = ct.c_uint64
    lib.ev_close.argtypes = [ct.c_void_p]
    lib.ev_close.restype = ct.c_int
    lib.ev_error.argtypes = []
    lib.ev_error.restype = ct.c_char_p
    return lib


def _gpu_resource_id(device):
    """Return a stable physical-device hint for broker state validation."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1", "NoDevices"):
        entries = [entry.strip() for entry in visible.split(",")]
        if 0 <= device < len(entries) and entries[device]:
            return f"cuda-visible:{entries[device]}"
    try:
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(device)
            uuid = getattr(properties, "uuid", None)
            if uuid is not None:
                return f"cuda-uuid:{uuid}"
    except (AssertionError, RuntimeError):
        pass
    return f"cuda-index:{device}"


def _resolve_broker_path(configured, resource_id):
    safe_id = "".join(
        character if character.isalnum() or character in "-._" else "_"
        for character in resource_id
    )
    expanded = Path(configured.replace("{gpu}", safe_id)).expanduser()
    if (
        expanded.is_dir()
        or configured.endswith("/")
        or configured.endswith("\\")
    ):
        expanded.mkdir(parents=True, exist_ok=True)
        return str(expanded / f"elastic-kv-{safe_id}.sqlite")
    return str(expanded)


class _BackingLifecycle:
    """Own native and broker teardown without retaining the Backing object."""

    def __init__(self, library, handle):
        self.library = library
        self.handle = handle
        self.broker = None

    def close(self):
        if self.handle:
            if self.library.ev_close(self.handle):
                error = self.library.ev_error().decode("utf-8", errors="replace")
                return "native", error or "native VMM close failed"
            self.handle = None
        if self.broker is not None:
            broker = self.broker
            try:
                broker.close()
            except Exception as exc:
                return "broker", str(exc) or type(exc).__name__
            self.broker = None
        return "closed", None


class Backing:
    """Single scheduler-thread owner of one CUDA virtual reservation.

    A tensor returned by raw can span unmapped addresses. Before reading or
    writing a tensor range, the caller must ensure every chunk it touches is
    mapped. Only the caller knows when a slot or in-flight GPU access is dead.
    reclaim() waits on the device by default, but the caller must also ensure
    no future work can access that chunk until it is ensured again.
    """

    def __init__(self, capacity_bytes, chunk_bytes=DEFAULT_CHUNK_BYTES, device=0, library=None):
        self.capacity_bytes = operator.index(capacity_bytes)
        self.chunk_bytes = operator.index(chunk_bytes)
        if not (0 < self.capacity_bytes < _INVALID and 0 < self.chunk_bytes < _INVALID):
            raise ValueError("capacity_bytes and chunk_bytes must be positive uint64 values")
        device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        if device.type != "cuda":
            raise ValueError("VMM backing requires a CUDA device")
        self.device = torch.cuda.current_device() if device.index is None else device.index
        if library is None:
            library = os.environ.get(
                "ELASTIC_VMM_LIBRARY",
                Path(__file__).resolve().parents[1] / "native" / "libelastic_vmm.so",
            )
        self._lib = _load_library(str(Path(library).expanduser().resolve()))
        self._handle = self._lib.ev_create(self.device, self.capacity_bytes, self.chunk_bytes)
        if not self._handle:
            raise RuntimeError(self._error())
        self._lifecycle = _BackingLifecycle(self._lib, self._handle)
        self._finalizer = weakref.finalize(self, self._lifecycle.close)
        self._finalizer.atexit = False
        self.base = self._number("ev_base")
        self.reserved_bytes = self._number("ev_reserved")
        self.granularity = self._number("ev_granularity")
        self._mapped = set()
        self._precommitted = set()
        self._raw_ref = None
        self._poisoned = False
        self._broker = None
        self._broker_leases = set()
        self._broker_claims = {}
        try:
            broker_path = os.environ.get("EKV_GLOBAL_BROKER_PATH")
            broker_mib = int(os.environ.get("EKV_GLOBAL_BROKER_MIB", "0"))
            fair_mode = os.environ.get(
                "EKV_GLOBAL_BROKER_FAIR_MODE", "off"
            ).strip().lower()
            if bool(broker_path) != (broker_mib > 0):
                raise ValueError(
                    "global VMM broker requires both EKV_GLOBAL_BROKER_PATH and "
                    "a positive EKV_GLOBAL_BROKER_MIB"
                )
            if fair_mode != "off" and not broker_path:
                raise ValueError(
                    "fair global VMM broker requires an enabled broker path"
                )
            if broker_path:
                stale = float(
                    os.environ.get("EKV_GLOBAL_BROKER_STALE_SECONDS", "120")
                )
                if stale <= 0:
                    raise ValueError(
                        "EKV_GLOBAL_BROKER_STALE_SECONDS must be positive"
                    )
                tenant_id = os.environ.get("EKV_GLOBAL_BROKER_TENANT") or None
                tenant_weight = float(
                    os.environ.get("EKV_GLOBAL_BROKER_TENANT_WEIGHT", "1")
                )
                demand_ttl = float(
                    os.environ.get("EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS", "10")
                )
                reserve_wait = float(
                    os.environ.get("EKV_GLOBAL_BROKER_WAIT_MS", "0")
                ) / 1000.0
                resource_id = _gpu_resource_id(self.device)
                self._broker = GlobalAdmissionBroker(
                    _resolve_broker_path(broker_path, resource_id),
                    broker_mib << 20,
                    stale_after_seconds=stale,
                    heartbeat_seconds=max(0.1, min(30.0, stale / 3)),
                    resource_id=resource_id,
                    tenant_id=tenant_id,
                    tenant_weight=tenant_weight,
                    fair_mode=fair_mode,
                    demand_ttl_seconds=demand_ttl,
                    reserve_wait_seconds=reserve_wait,
                )
                self._lifecycle.broker = self._broker
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise
        self._metrics = {
            "ensure_calls": 0,
            "ensure_chunks": 0,
            "ensure_ns": 0,
            "precommit_calls": 0,
            "precommit_chunks": 0,
            "precommit_ns": 0,
            "reclaim_calls": 0,
            "reclaim_chunks": 0,
            "reclaim_ns": 0,
            "reclaim_sync_ns": 0,
            "peak_backed_bytes": 0,
            "global_broker_reservations": 0,
            "global_broker_denials": 0,
            "global_broker_retries": 0,
            "global_broker_wait_ns": 0,
            "global_broker_released_chunks": 0,
            "global_broker_claim_checks": 0,
            "global_broker_claim_denials": 0,
            "global_broker_claim_retries": 0,
            "global_broker_claim_wait_ns": 0,
            "global_broker_claim_releases": 0,
            "global_broker_claim_shrinks": 0,
            "global_broker_claim_shrunk_bytes": 0,
            "global_broker_peak_claim_bytes": 0,
        }

    def _error(self):
        return self._lib.ev_error().decode("utf-8", errors="replace")

    def _number(self, name):
        if not self._handle:
            raise RuntimeError("VMM backing is closed")
        value = getattr(self._lib, name)(self._handle)
        if value == _INVALID:
            raise RuntimeError(self._error())
        return value

    def _check(self):
        if not self._handle or self._poisoned:
            raise RuntimeError("VMM backing is closed or has encountered a CUDA error")

    def _ids(self, indices):
        ids = {operator.index(index) for index in indices}
        if any(index < 0 or index >= self.reserved_bytes // self.chunk_bytes for index in ids):
            raise ValueError("chunk index outside reserved range")
        return ids

    def _batch(self, name, indices):
        ids = (ct.c_uint64 * len(indices))(*sorted(indices))
        if getattr(self._lib, name)(self._handle, ids, len(indices)):
            self._poisoned = True
            raise RuntimeError(self._error())

    @property
    def mapped_chunks(self):
        return frozenset(self._mapped - self._precommitted)

    @property
    def backed_bytes(self):
        return self._number("ev_backed")

    @property
    def metrics(self):
        result = {
            **self._metrics,
            "backed_bytes": self.backed_bytes,
            "reserved_bytes": self.reserved_bytes,
        }
        if self._broker is not None:
            result["global_broker"] = self._broker.snapshot()
        return result

    @property
    def broker_available_bytes(self):
        snapshot = self.broker_snapshot
        return None if snapshot is None else snapshot["available_bytes"]

    @property
    def broker_snapshot(self):
        return None if self._broker is None else self._broker.snapshot()

    @property
    def has_global_broker(self):
        return self._broker is not None

    def _broker_key(self, index):
        return f"{self._broker.owner_id}/device-{self.device}/chunk-{index}"

    def _reserve_broker(self, indices):
        if self._broker is None or not indices:
            return ()
        decision = self._broker.reserve(
            {self._broker_key(index): self.chunk_bytes for index in indices}
        )
        if not decision.admit:
            self._metrics["global_broker_denials"] += 1
            self._metrics["global_broker_retries"] += decision.retry_count
            self._metrics["global_broker_wait_ns"] += int(
                decision.waited_seconds * 1_000_000_000
            )
            raise RuntimeError(
                "global VMM capacity exhausted or fair-share yield: "
                f"reason={decision.reason} "
                f"required={decision.required_bytes} "
                f"available={decision.available_bytes}"
            )
        self._metrics["global_broker_retries"] += decision.retry_count
        self._metrics["global_broker_wait_ns"] += int(
            decision.waited_seconds * 1_000_000_000
        )
        self._broker_leases.update(decision.acquired_keys)
        self._metrics["global_broker_reservations"] += len(decision.acquired_keys)
        return decision.acquired_keys

    def _release_broker(self, indices):
        if self._broker is None or not indices:
            return
        keys = tuple(
            self._broker_key(index)
            for index in indices
            if self._broker_key(index) in self._broker_leases
        )
        if not keys:
            return
        self._broker.release(keys)
        self._broker_leases.difference_update(keys)
        self._metrics["global_broker_released_chunks"] += len(keys)

    def claim_broker_capacity(self, key, byte_count, *, kind="prefill"):
        """Reserve temporary non-mapped capacity in the aggregate broker.

        ``None`` means this backing has no broker. ``True`` or ``False``
        describes an actual broker admission decision. The caller owns the
        returned claim until ``release_broker_claim`` or backing teardown.
        """

        if self._broker is None:
            return None
        normalized = str(key)
        if not normalized:
            raise ValueError("broker claim key must be nonempty")
        normalized_kind = str(kind)
        if not normalized_kind or len(normalized_kind) > 64:
            raise ValueError("broker claim kind must be a short nonempty string")
        byte_count = operator.index(byte_count)
        if byte_count < 1:
            raise ValueError("broker claim bytes must be positive")
        full_key = (
            f"{self._broker.owner_id}/device-{self.device}/"
            f"claim/{normalized_kind}/{normalized}"
        )
        self._metrics["global_broker_claim_checks"] += 1
        decision = self._broker.claim(
            full_key, byte_count, kind=normalized_kind
        )
        self._metrics["global_broker_claim_retries"] += decision.retry_count
        self._metrics["global_broker_claim_wait_ns"] += int(
            decision.waited_seconds * 1_000_000_000
        )
        if not decision.admit:
            self._metrics["global_broker_claim_denials"] += 1
            return False
        self._broker_claims[full_key] = byte_count
        self._metrics["global_broker_peak_claim_bytes"] = max(
            self._metrics["global_broker_peak_claim_bytes"],
            sum(self._broker_claims.values()),
        )
        return True

    def release_broker_claim(self, key, *, kind="prefill"):
        if self._broker is None:
            return 0
        normalized = str(key)
        normalized_kind = str(kind)
        full_key = (
            f"{self._broker.owner_id}/device-{self.device}/"
            f"claim/{normalized_kind}/{normalized}"
        )
        released = self._broker.release_claim(full_key)
        self._broker_claims.pop(full_key, None)
        if released:
            self._metrics["global_broker_claim_releases"] += 1
        return released

    def shrink_broker_claim(self, key, maximum_bytes, *, kind="prefill"):
        """Shrink a held temporary claim after SGLang selects a concrete batch."""

        if self._broker is None:
            return None
        normalized = str(key)
        if not normalized:
            raise ValueError("broker claim key must be nonempty")
        normalized_kind = str(kind)
        if not normalized_kind or len(normalized_kind) > 64:
            raise ValueError("broker claim kind must be a short nonempty string")
        maximum_bytes = operator.index(maximum_bytes)
        if maximum_bytes < 1:
            raise ValueError("maximum broker claim bytes must be positive")
        full_key = (
            f"{self._broker.owner_id}/device-{self.device}/"
            f"claim/{normalized_kind}/{normalized}"
        )
        previous = self._broker_claims.get(full_key)
        revised = self._broker.shrink_claim(full_key, maximum_bytes)
        if revised is not None:
            self._broker_claims[full_key] = revised
            if previous is not None and revised < previous:
                self._metrics["global_broker_claim_shrinks"] += 1
                self._metrics["global_broker_claim_shrunk_bytes"] += previous - revised
        return revised

    @property
    def raw(self):
        """Borrow a contiguous uint8 view spanning the logical reservation."""
        self._check()
        if 0 not in self.mapped_chunks:
            raise RuntimeError("commit the sink chunk before building the tensor view")
        cached = self._raw_ref() if self._raw_ref else None
        if cached is not None:
            return cached
        import cupy

        with cupy.cuda.Device(self.device), torch.cuda.device(self.device):
            memory = cupy.cuda.UnownedMemory(
                self.base, self.capacity_bytes, owner=self, device_id=self.device
            )
            array = cupy.ndarray(
                (self.capacity_bytes,),
                dtype=cupy.uint8,
                memptr=cupy.cuda.MemoryPointer(memory, 0),
            )
            raw = torch.from_dlpack(array)
        if raw.data_ptr() != self.base:
            raise RuntimeError("DLPack did not preserve the reserved address")
        self._raw_ref = weakref.ref(raw)
        return raw

    def ensure(self, indices):
        """Commit and zero missing chunks, returning their count."""
        self._check()
        ids = self._ids(indices)
        pending = ids - self._mapped
        needs_initialization = pending or bool(ids & self._precommitted)
        if not needs_initialization:
            return 0
        start = time.perf_counter_ns()
        self._reserve_broker(pending)
        # A failed batch may have mapped a prefix. Keep every lease until a
        # successful close proves that no native mapping remains.
        self._batch("ev_ensure", pending)
        self._mapped.update(pending)
        self._precommitted.clear()
        self._metrics["ensure_ns"] += time.perf_counter_ns() - start
        self._metrics["ensure_calls"] += 1
        self._metrics["ensure_chunks"] += len(pending)
        self._metrics["peak_backed_bytes"] = max(
            self._metrics["peak_backed_bytes"], len(self._mapped) * self.chunk_bytes
        )
        return len(pending)

    def precommit(self, indices):
        """Map missing chunks and start zeroing without blocking the caller.

        The chunks become usable only after a later ``ensure`` waits for the
        backing initialization stream. Callers must keep the returned chunks
        out of kernel argument lists until that synchronization completes.
        """
        self._check()
        pending = self._ids(indices) - self._mapped
        if not pending:
            return 0
        start = time.perf_counter_ns()
        self._reserve_broker(pending)
        # As with ensure(), an error can follow a partial native mapping.
        self._batch("ev_precommit", pending)
        self._mapped.update(pending)
        self._precommitted.update(pending)
        self._metrics["precommit_ns"] += time.perf_counter_ns() - start
        self._metrics["precommit_calls"] += 1
        self._metrics["precommit_chunks"] += len(pending)
        self._metrics["peak_backed_bytes"] = max(
            self._metrics["peak_backed_bytes"], len(self._mapped) * self.chunk_bytes
        )
        return len(pending)

    def reclaim(self, indices, *, synchronize=True):
        """Release idle chunks, returning their count.

        synchronize=False requires the caller to have already waited for all
        in-flight GPU accesses to the affected ranges.
        """
        self._check()
        pending = self._ids(indices) & self._mapped
        if not pending:
            return 0
        start = time.perf_counter_ns()
        if synchronize:
            torch.cuda.synchronize(self.device)
            self._metrics["reclaim_sync_ns"] += time.perf_counter_ns() - start
        self._batch("ev_drop", pending)
        self._mapped.difference_update(pending)
        self._precommitted.difference_update(pending)
        self._release_broker(pending)
        self._metrics["reclaim_ns"] += time.perf_counter_ns() - start
        self._metrics["reclaim_calls"] += 1
        self._metrics["reclaim_chunks"] += len(pending)
        return len(pending)

    def close(self):
        """Close once no tensor view, kernel, or transfer can access backing."""
        stage, message = self._lifecycle.close()
        self._handle = None
        if stage != "closed":
            self._poisoned = True
            raise RuntimeError(f"{stage} VMM teardown failed: {message}")
        self._finalizer.detach()
        self._broker = None
        self._broker_leases.clear()
        self._broker_claims.clear()
        if self._lifecycle.handle is None:
            self._mapped.clear()
            self._precommitted.clear()
