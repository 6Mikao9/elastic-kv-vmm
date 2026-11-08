import unittest

from benchmarks.affine_placement_sim import run_scenario
from experimental.affine_placement import (
    AffineArena,
    AffineConfig,
    AffineRequest,
    Placement,
    chunks_for_prefix,
    simulate_affine_placement,
)


class AffinePlacementTest(unittest.TestCase):
    def test_packed_affine_shares_tail_chunks(self):
        requests = [
            AffineRequest(f"r{index}", 0, prompt_tokens=2, output_tokens=8)
            for index in range(4)
        ]
        config = AffineConfig(
            arena_bytes=80,
            chunk_bytes=8,
            row_bytes=1,
            max_active_requests=4,
            prefill_tokens_per_ms=8,
            decode_tokens_per_ms=20,
        )

        fixed = simulate_affine_placement(requests, config, "fixed-affine")
        private = simulate_affine_placement(requests, config, "private-affine")
        packed = simulate_affine_placement(requests, config, "packed-affine")

        self.assertEqual(fixed.peak.mapped_bytes, 64)
        self.assertEqual(fixed.peak.shared_chunks, 0)
        self.assertEqual(private.peak.mapped_bytes, 64)
        self.assertEqual(private.peak.shared_chunks, 0)
        self.assertEqual(packed.peak.mapped_bytes, 40)
        self.assertEqual(packed.peak.useful_bytes, 40)
        self.assertEqual(packed.peak.tail_waste_bytes, 0)
        self.assertEqual(packed.peak.shared_chunks, 3)
        self.assertEqual(packed.peak.shared_savings_bytes, 24)

    def test_token_mapping_stays_affine_when_rows_cross_chunks(self):
        arena = AffineArena(32, chunk_bytes=8)
        first = arena.reserve("a", total_tokens=3, row_bytes=3)
        second = arena.reserve("b", total_tokens=3, row_bytes=3)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None

        self.assertEqual(first.start_byte, 0)
        self.assertEqual(second.start_byte, 9)
        self.assertEqual(first.token_offset(1), 3)
        self.assertEqual(first.token_offset(2), 6)
        self.assertEqual(second.token_offset(0), 9)
        self.assertEqual(chunks_for_prefix(first, 3, 8), {0, 1})
        self.assertEqual(chunks_for_prefix(second, 3, 8), {1, 2})

    def test_page_boundaries_cover_every_chunk_transition(self):
        row_bytes = 32 * 1024
        placement = Placement(
            "page-boundary",
            start_byte=0,
            reserved_bytes=129 * row_bytes,
            total_tokens=129,
            row_bytes=row_bytes,
        )
        expected_chunks = {
            1: 1,
            63: 1,
            64: 1,
            65: 2,
            127: 2,
            128: 2,
            129: 3,
        }

        for pages, expected in expected_chunks.items():
            with self.subTest(pages=pages):
                self.assertEqual(
                    len(chunks_for_prefix(placement, pages, 2 << 20)),
                    expected,
                )

    def test_packed_tail_sharing_couples_reclaim(self):
        packed = AffineArena(16, chunk_bytes=8)
        first = packed.reserve("first", total_tokens=4, row_bytes=1)
        second = packed.reserve("second", total_tokens=4, row_bytes=1)
        assert first is not None and second is not None
        self.assertEqual(chunks_for_prefix(first, 4, 8), {0})
        self.assertEqual(chunks_for_prefix(second, 4, 8), {0})

        packed.release("first")

        self.assertEqual(chunks_for_prefix(packed["second"], 4, 8), {0})

        private = AffineArena(16, chunk_bytes=8, slot_bytes=8)
        first = private.reserve("first", total_tokens=4, row_bytes=1)
        second = private.reserve("second", total_tokens=4, row_bytes=1)
        assert first is not None and second is not None
        self.assertEqual(chunks_for_prefix(first, 4, 8), {0})
        self.assertEqual(chunks_for_prefix(second, 4, 8), {1})

        private.release("first")

        self.assertEqual(chunks_for_prefix(private["second"], 4, 8), {1})

    def test_private_affine_rounds_each_request_without_sharing(self):
        arena = AffineArena(32, chunk_bytes=8, round_to_chunks=True)
        first = arena.reserve("first", total_tokens=5, row_bytes=1)
        second = arena.reserve("second", total_tokens=3, row_bytes=1)
        assert first is not None and second is not None

        self.assertEqual((first.start_byte, first.reserved_bytes), (0, 8))
        self.assertEqual((second.start_byte, second.reserved_bytes), (8, 8))
        self.assertEqual(chunks_for_prefix(first, 5, 8), {0})
        self.assertEqual(chunks_for_prefix(second, 3, 8), {1})

    def test_chunk_rounded_reservation_skips_an_unaligned_free_prefix(self):
        arena = AffineArena(32, chunk_bytes=8)
        packed = arena.reserve("short", total_tokens=3, row_bytes=1)
        assert packed is not None
        private = arena.reserve(
            "long",
            total_tokens=8,
            row_bytes=1,
            round_to_chunks=True,
        )
        assert private is not None

        self.assertEqual(packed.start_byte, 0)
        self.assertEqual(private.start_byte, 8)
        self.assertEqual(arena.free_ranges, ((3, 5), (16, 16)))

    def test_private_affine_reclaims_all_touched_chunks_independently(self):
        arena = AffineArena(32, chunk_bytes=8, round_to_chunks=True)
        first = arena.reserve("first", total_tokens=9, row_bytes=1)
        second = arena.reserve("second", total_tokens=9, row_bytes=1)
        assert first is not None and second is not None
        self.assertEqual(chunks_for_prefix(first, 9, 8), {0, 1})
        self.assertEqual(chunks_for_prefix(second, 9, 8), {2, 3})

        arena.release("first")

        self.assertEqual(arena.free_ranges, ((0, 16),))
        self.assertEqual(chunks_for_prefix(arena["second"], 9, 8), {2, 3})

    def test_slot_and_chunk_rounding_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            AffineArena(16, chunk_bytes=8, slot_bytes=8, round_to_chunks=True)
        with self.assertRaisesRegex(TypeError, "boolean"):
            AffineArena(16, chunk_bytes=8, round_to_chunks="yes")  # type: ignore[arg-type]

    def test_release_coalesces_ranges_and_keeps_fragmented_ranges_apart(self):
        arena = AffineArena(32, chunk_bytes=8)
        arena.reserve("a", 8, 1)
        arena.reserve("b", 8, 1)
        arena.reserve("c", 8, 1)
        arena.release("b")
        self.assertEqual(arena.free_ranges, ((8, 8), (24, 8)))
        self.assertIsNone(arena.reserve("too-large", 9, 1))
        arena.release("a")
        self.assertEqual(arena.free_ranges, ((0, 16), (24, 8)))
        arena.release("c")
        self.assertEqual(arena.free_ranges, ((0, 32),))

    def test_fixed_slot_rejects_a_request_larger_than_its_stride(self):
        arena = AffineArena(32, chunk_bytes=8, slot_bytes=16)
        with self.assertRaisesRegex(ValueError, "configured affine slot"):
            arena.reserve("too-long", total_tokens=17, row_bytes=1)

    def test_simulation_is_deterministic(self):
        requests = [
            AffineRequest("long", 0, 100, 300),
            AffineRequest("short-a", 1, 20, 40),
            AffineRequest("short-b", 2, 20, 40),
        ]
        config = AffineConfig(16_384, 2_048, 4, 3)
        for strategy in (
            "fixed-affine",
            "private-affine",
            "packed-affine",
            "hybrid-affine",
        ):
            with self.subTest(strategy=strategy):
                first = simulate_affine_placement(requests, config, strategy)
                second = simulate_affine_placement(requests, config, strategy)
                self.assertEqual(first, second)
                self.assertEqual(first.admitted_requests, 3)
                self.assertLessEqual(first.peak.mapped_bytes, config.arena_bytes)

    def test_hybrid_keeps_small_requests_packed_and_large_requests_private(self):
        requests = [
            AffineRequest("short-a", 0, 2, 2),
            AffineRequest("short-b", 0, 2, 2),
            AffineRequest("long", 0, 7, 1),
        ]
        config = AffineConfig(
            arena_bytes=32,
            chunk_bytes=8,
            row_bytes=1,
            max_active_requests=3,
            prefill_tokens_per_ms=100,
            decode_tokens_per_ms=100,
        )
        fixed = simulate_affine_placement(requests, config, "fixed-affine")
        hybrid = simulate_affine_placement(requests, config, "hybrid-affine")

        self.assertEqual(fixed.peak.mapped_bytes, 24)
        self.assertEqual(hybrid.peak.mapped_bytes, 16)
        self.assertEqual(hybrid.peak.shared_chunks, 1)

    def test_benchmark_payload_includes_private_reclaim_control(self):
        payload = run_scenario(
            "short-heavy",
            request_count=20,
            seed=17,
            config=AffineConfig(32 << 20, 2 << 20, 512, 8),
        )

        self.assertEqual(
            set(payload),
            {
                "scenario",
                "seed",
                "request_count",
                "total_tokens",
                "min_total_tokens",
                "max_total_tokens",
                "fixed_affine",
                "private_affine",
                "hybrid_affine",
                "packed_affine",
                "comparison",
            },
        )
        self.assertEqual(
            payload["private_affine"]["peak"]["shared_chunks"],  # type: ignore[index]
            0,
        )
        self.assertGreater(
            payload["packed_affine"]["peak"]["shared_chunks"],  # type: ignore[index]
            0,
        )

    def test_rejects_invalid_configuration_and_strategy(self):
        with self.assertRaisesRegex(ValueError, "arena_bytes"):
            AffineConfig(10, 8, 1, 1)
        with self.assertRaisesRegex(ValueError, "strategy"):
            simulate_affine_placement(
                [AffineRequest("r", 0, 1, 1)],
                AffineConfig(8, 8, 1, 1),
                "page-table",
            )


if __name__ == "__main__":
    unittest.main()
