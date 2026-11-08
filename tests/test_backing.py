"""GPU tests for true VMM mapping and lifetime; require the compiled native library."""

import unittest

import torch

from elastic_kv.backing import Backing, DEFAULT_CHUNK_BYTES


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class BackingTest(unittest.TestCase):
    def test_release_and_recommit_one_hole(self):
        chunk = DEFAULT_CHUNK_BYTES
        torch.cuda.set_device(0)
        backing = Backing(5 * chunk - 17, chunk_bytes=chunk, device=0)
        try:
            self._exercise(backing, chunk)
        finally:
            backing.close()

    def _exercise(self, backing, chunk):
        # Chunk 3 remains unmapped. Only explicitly mapped slices are accessed.
        self.assertEqual(backing.ensure([0, 1, 2, 4]), 4)
        self.assertEqual(backing.reserved_bytes, 5 * chunk)
        self.assertEqual(backing.backed_bytes, 4 * chunk)
        raw = backing.raw
        base = raw.data_ptr()
        self.assertEqual(base, backing.base)
        self.assertEqual(raw.numel(), 5 * chunk - 17)
        views = {i: raw[i * chunk : (i + 1) * chunk] for i in (0, 1, 2, 4)}
        for i, value in ((0, 11), (1, 22), (2, 33), (4, 55)):
            views[i].fill_(value)

        self.assertEqual(backing.reclaim([1, 3, 1]), 1)
        self.assertEqual(backing.mapped_chunks, {0, 2, 4})
        self.assertEqual(backing.backed_bytes, 3 * chunk)
        for i, value in ((0, 11), (2, 33), (4, 55)):
            self.assertTrue(views[i].eq(value).all().item())

        # Precommit maps the chunk but leaves zeroing on the initialization
        # stream. ``ensure`` is still the readiness barrier before GPU use.
        self.assertEqual(backing.precommit([1]), 1)
        self.assertEqual(backing.mapped_chunks, {0, 2, 4})
        self.assertEqual(backing.backed_bytes, 4 * chunk)
        self.assertEqual(backing.ensure([1]), 0)
        self.assertEqual(backing.mapped_chunks, {0, 1, 2, 4})
        self.assertEqual(backing.backed_bytes, 4 * chunk)
        self.assertEqual(backing.raw.data_ptr(), base)
        self.assertEqual(views[1].data_ptr(), base + chunk)
        self.assertEqual(torch.count_nonzero(views[1]).item(), 0)
        for i, value in ((0, 11), (2, 33), (4, 55)):
            self.assertEqual(views[i].data_ptr(), base + i * chunk)
            self.assertTrue(views[i].eq(value).all().item())


if __name__ == "__main__":
    unittest.main()
