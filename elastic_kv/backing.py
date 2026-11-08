"""CUDA VMM backing with stable torch views and explicit residency control."""

import ctypes as ct
import functools
import operator
import os
import time
import weakref
from pathlib import Path

import torch


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
        self._finalizer = weakref.finalize(self, self._lib.ev_close, self._handle)
        self._finalizer.atexit = False
        self.base = self._number("ev_base")
        self.reserved_bytes = self._number("ev_reserved")
        self.granularity = self._number("ev_granularity")
        self._mapped = set()
        self._precommitted = set()
        self._raw_ref = None
        self._poisoned = False
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
        return {
            **self._metrics,
            "backed_bytes": self.backed_bytes,
            "reserved_bytes": self.reserved_bytes,
        }

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
        self._metrics["reclaim_ns"] += time.perf_counter_ns() - start
        self._metrics["reclaim_calls"] += 1
        self._metrics["reclaim_chunks"] += len(pending)
        return len(pending)

    def close(self):
        """Close once no tensor view, kernel, or transfer can access backing."""
        if not self._handle:
            return
        self._handle = None
        status = self._finalizer()
        self._mapped.clear()
        self._precommitted.clear()
        if status:
            raise RuntimeError(self._error())
