"""Byte layout for independent, chunk-aligned request KV regions."""

from dataclasses import dataclass


def _integer(name, value, minimum, maximum=None):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside supported range")


@dataclass(frozen=True)
class VirtualKVLayout:
    max_requests: int
    max_tokens: int
    row_elements: int
    itemsize: int = 2
    chunk_bytes: int = 2 << 20

    def __post_init__(self):
        for name in ("max_requests", "max_tokens", "row_elements", "itemsize", "chunk_bytes"):
            _integer(name, getattr(self, name), 1)

    @property
    def row_bytes(self):
        return self.row_elements * self.itemsize

    @property
    def request_bytes(self):
        used = self.max_tokens * self.row_bytes
        return ((used + self.chunk_bytes - 1) // self.chunk_bytes) * self.chunk_bytes

    @property
    def capacity_bytes(self):
        return self.chunk_bytes + self.max_requests * self.request_bytes

    def request_offset(self, req):
        _integer("request index", req, 0, self.max_requests - 1)
        return self.chunk_bytes + req * self.request_bytes

    def token_offset(self, req, token):
        _integer("token position", token, 0, self.max_tokens - 1)
        return self.request_offset(req) + token * self.row_bytes

    def _chunk_range(self, start, size):
        if size == 0:
            return range(0)
        return range(start // self.chunk_bytes, (start + size - 1) // self.chunk_bytes + 1)

    def prefix_chunks(self, req, length):
        _integer("prefix length", length, 0, self.max_tokens)
        return frozenset(self._chunk_range(self.request_offset(req), length * self.row_bytes))

    def selected_chunks(self, req, positions):
        self.request_offset(req)
        return frozenset(
            chunk
            for token in positions
            for chunk in self._chunk_range(self.token_offset(req, token), self.row_bytes)
        )
