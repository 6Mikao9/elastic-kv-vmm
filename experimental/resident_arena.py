"""Explicitly owned, single-threaded VMM arena for resident KV experiments."""


class VmmResidentArena:
    """Views are borrowed. Discard tail views before shrinking, and all views
    before closing. Newly exposed rows must be written before use, since a
    retained chunk can still contain bytes from a previously discarded tail.
    """

    def __init__(self, layout, device=0, backing_factory=None, *, synchronize=None):
        if layout.chunk_bytes % layout.itemsize:
            raise ValueError("chunk size must align to the element size")
        if backing_factory is None:
            from elastic_kv.backing import Backing

            backing_factory = Backing
        self.layout = layout
        self.backing = backing_factory(layout.capacity_bytes, chunk_bytes=layout.chunk_bytes, device=device)
        if synchronize is None:
            def synchronize():
                import torch

                torch.cuda.synchronize(self.backing.device)

        self._synchronize = synchronize
        self._lengths = [0] * layout.max_requests
        self._closed = False
        self._failed = False
        self.backing.ensure({0})

    def _check(self):
        if self._closed or self._failed:
            raise RuntimeError("arena is closed or has encountered a backing error")

    def reserve(self, req, length):
        """Grow the mapped prefix, preserving existing rows."""
        self._check()
        target = self.layout.prefix_chunks(req, length)
        previous = self._lengths[req]
        if length <= previous:
            return
        pending = target - self.layout.prefix_chunks(req, previous)
        if pending:
            try:
                self.backing.ensure(pending)
            except Exception:
                self._failed = True
                raise
        self._lengths[req] = length

    def resize_quiescent(self, req, length):
        """Resize a prefix. Shrinking discards tail rows and their future use."""
        self._check()
        target = self.layout.prefix_chunks(req, length)
        previous = self._lengths[req]
        if length >= previous:
            self.reserve(req, length)
            return
        dropped = self.layout.prefix_chunks(req, previous) - target
        try:
            self._synchronize()
            if dropped:
                self.backing.reclaim(dropped, synchronize=False)
        except Exception:
            self._failed = True
            raise
        self._lengths[req] = length

    def row_view(self, req, length, dtype):
        """Borrow a tensor shaped (length, row_elements) over the mapped prefix."""
        self._check()
        self.layout.prefix_chunks(req, length)
        if length > self._lengths[req]:
            raise ValueError("requested view exceeds the committed prefix")
        if dtype.itemsize != self.layout.itemsize:
            raise ValueError("dtype item size does not match the layout")
        raw = self.backing.raw
        return raw.narrow(0, self.layout.request_offset(req), length * self.layout.row_bytes).view(
            dtype
        ).reshape(length, self.layout.row_elements)

    def close(self):
        """Synchronize and release the reservation, after all views are retired."""
        if self._closed:
            return
        try:
            self._synchronize()
            self.backing.close()
        except Exception:
            self._failed = True
            raise
        self._closed = True
