import unittest

from adaptive_sparse.minimax_vmm import PhysicalPages


class Backing:
    def __init__(self):
        self.mapped_chunks = set()
        self.fail = False

    @property
    def backed_bytes(self):
        return len(self.mapped_chunks) * 8

    def ensure(self, indices):
        if self.fail:
            raise RuntimeError("map failed")
        self.mapped_chunks.update(indices)

    def reclaim(self, indices, *, synchronize):
        assert not synchronize
        self.mapped_chunks.difference_update(indices)
        return len(indices)


class PhysicalPagesTest(unittest.TestCase):
    def test_crossing_shared_chunks_tail_reuse_and_sink(self):
        backing = Backing()
        syncs = []
        pages = PhysicalPages(backing, 2, 4, [(0, 3), (8, 2)],
                              lambda: syncs.append(True), chunk_bytes=8)
        self.assertEqual(pages.chunks(1), {0, 1, 8})
        pages.acquire([1, 2, 2])
        pages.acquire([2])
        pages.release({1})
        self.assertEqual(pages.stats()["reclaimable_bytes"], 0)
        pages.release({2})
        self.assertEqual(pages.trim_quiescent(), 24)
        self.assertEqual(backing.mapped_chunks, {0, 8})
        self.assertEqual(syncs, [True])
        with self.assertRaises(RuntimeError):
            pages.release({0})

    def test_failed_map_does_not_change_page_ownership(self):
        backing = Backing()
        pages = PhysicalPages(backing, 1, 3, [(0, 8)], lambda: None, chunk_bytes=8)
        backing.fail = True
        with self.assertRaisesRegex(RuntimeError, "map failed"):
            pages.acquire([1])
        self.assertEqual(pages.pages, {0})
        self.assertEqual(pages.refs, {0: 1})
