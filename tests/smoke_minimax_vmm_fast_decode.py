"""CPU interface test for the pinned SGLang allocator fast path."""
from types import SimpleNamespace

import torch

from adaptive_sparse.minimax_vmm import PhysicalPages, _attach_allocator


def main():
    mapped = set()
    backing = SimpleNamespace(ensure=mapped.update, mapped_chunks=mapped, backed_bytes=0)
    tracker = PhysicalPages(backing, 4, 8, [(0, 2)], lambda: None,
                            chunk_bytes=8, fast_decode=True)
    tracker.acquire([1])

    class NoHostRead:
        def tolist(self):
            raise AssertionError("non-growing decode read device IDs")

    class Allocator:
        free_pages = torch.tensor([2, 3])
        staged_pages = []
        num_staged_pages = 0

        def alloc(self):
            raise AssertionError("unexpected allocation")

        alloc_extend = alloc

        def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
            if seq_lens_cpu.item() == 5:
                self.free_pages = self.free_pages[1:]
                return torch.tensor([8])
            return NoHostRead()

        def _release_page_ids(self, *pages):
            pass

        def clear(self):
            pass

    allocator = Allocator()
    _attach_allocator(allocator, tracker)
    for length in (2, 3, 4):
        lens = torch.tensor([length])
        assert isinstance(allocator.alloc_decode(lens, lens, None), NoHostRead)
    lens = torch.tensor([5])
    assert allocator.alloc_decode(lens, lens, None).tolist() == [8]
    assert tracker.pages == {0, 1, 2} and mapped == {0, 1, 2}
    assert tracker.stats()["fast_no_new_page"] == 3
    assert tracker.stats()["decode_new_page"] == 1
    assert isinstance(tracker.chunks(2), frozenset)
    assert tracker.chunks(2) is tracker.chunks(2)
    print("VMM decode fast path CPU interface passed")


if __name__ == "__main__":
    main()
