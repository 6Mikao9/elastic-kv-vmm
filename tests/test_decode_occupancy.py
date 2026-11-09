import unittest

from adaptive_sparse.decode_occupancy import compare_decode_occupancy


class DecodeOccupancyTest(unittest.TestCase):
    def test_compares_planned_width_with_per_request_observations(self):
        rows = [
            {"request_id": "mixed", "kind": "chat", "decode_batch_size": 4},
            {"request_id": "matching", "kind": "tool", "decode_batch_size": 2},
            {"request_id": "missing", "kind": "tool", "decode_batch_size": 1},
        ]
        result = compare_decode_occupancy(
            rows,
            {
                "decode_batch_sizes_by_request": {
                    "mixed": {"4": 3, "2": 1},
                    "matching": {"2": 2},
                    "unplanned": {"1": 1},
                },
                "decode_trace": [
                    {
                        "observed_at_s": 10.0,
                        "batch_size": 4,
                        "mixed_prefill_decode": False,
                    },
                    {
                        "observed_at_s": 10.125,
                        "batch_size": 2,
                        "mixed_prefill_decode": True,
                    },
                ],
            },
        )

        by_id = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(by_id["mixed"]["exact_events"], 3)
        self.assertEqual(by_id["mixed"]["observed_events"], 4)
        self.assertEqual(by_id["mixed"]["exact_event_fraction"], 0.75)
        self.assertFalse(by_id["mixed"]["exactly_matches_plan"])
        self.assertTrue(by_id["matching"]["exactly_matches_plan"])
        self.assertIsNone(by_id["missing"]["weighted_mean_batch_size"])
        self.assertEqual(
            result["summary"],
            {
                "planned_request_count": 3,
                "observed_request_count": 2,
                "missing_request_ids": ["missing"],
                "unplanned_observed_request_ids": ["unplanned"],
                "observed_decode_events": 6,
                "exact_decode_events": 5,
                "exact_event_fraction": 5 / 6,
                "requests_fully_matching": 1,
                "requests_with_mismatch": 1,
                "complete_request_coverage": False,
            },
        )
        self.assertEqual(
            result["trace"],
            {
                "event_count": 2,
                "mixed_event_count": 1,
                "batch_size_histogram": {"2": 1, "4": 1},
                "start_s": 10.0,
                "end_s": 10.125,
                "duration_s": 0.125,
            },
        )

    def test_rejects_malformed_plan_histogram_and_trace(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compare_decode_occupancy(
                [{"request_id": "bad", "decode_batch_size": 0}],
                {},
            )
        with self.assertRaisesRegex(TypeError, "must be an object"):
            compare_decode_occupancy(
                [{"request_id": "bad", "decode_batch_size": 1}],
                {"decode_batch_sizes_by_request": []},
            )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            compare_decode_occupancy(
                [{"request_id": "bad", "decode_batch_size": 1}],
                {"decode_batch_sizes_by_request": {"bad": {"1.5": 1}}},
            )
        with self.assertRaisesRegex(ValueError, "invalid observation time"):
            compare_decode_occupancy(
                [{"request_id": "bad", "decode_batch_size": 1}],
                {
                    "decode_batch_sizes_by_request": {},
                    "decode_trace": [
                        {"observed_at_s": None, "batch_size": 1}
                    ],
                },
            )


if __name__ == "__main__":
    unittest.main()
