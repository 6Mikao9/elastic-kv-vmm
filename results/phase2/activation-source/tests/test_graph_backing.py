"""Exercise a captured gather across native VMM reclaim and recommit."""
import unittest

import torch

from elastic_kv.backing import Backing


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class GraphBackingTest(unittest.TestCase):
    def test_captured_gather_reuses_virtual_addresses_after_recommit(self):
        chunk = 2 << 20
        with torch.cuda.device(0):
            backing = Backing(4 * chunk, chunk_bytes=chunk, device=0)
            try:
                self._exercise(backing, chunk)
            finally:
                backing.close()

    def _exercise(self, backing, chunk):
        self.assertEqual(backing.ensure([0, 1, 2]), 3)
        raw = backing.raw
        base = raw.data_ptr()
        raw[0] = 7
        raw[chunk] = 23
        raw[2 * chunk] = 41
        indices = torch.tensor([chunk, 2 * chunk, 0], device="cuda")
        input_ptr = indices.data_ptr()

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            result = torch.index_select(raw, 0, indices)

        def replay(offsets, expected):
            indices.copy_(torch.tensor(offsets, device="cuda"))
            graph.replay()
            self.assertEqual(result.tolist(), expected)
            self.assertEqual(indices.data_ptr(), input_ptr)
            self.assertEqual(backing.raw.data_ptr(), base)

        replay([chunk, 2 * chunk, 0], [23, 41, 7])
        torch.cuda.synchronize()
        self.assertEqual(backing.reclaim([1], synchronize=False), 1)
        self.assertEqual(backing.mapped_chunks, {0, 2})
        replay([2 * chunk, 0, 0], [41, 7, 7])

        self.assertEqual(backing.ensure([1]), 1)
        self.assertEqual(backing.mapped_chunks, {0, 1, 2})
        raw[chunk] = 97
        replay([chunk, 2 * chunk, 0], [97, 41, 7])


if __name__ == "__main__":
    unittest.main()
