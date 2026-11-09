import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from adaptive_sparse.future_plan import PlannerConfig, ResidencyMode
from adaptive_sparse.profile_calibration import calibrate_profiles
from scripts.calibrate_future_profiles import main as calibrate_main


class ProfileCalibrationTest(unittest.TestCase):
    def test_builds_profiles_only_from_complete_unambiguous_measurements(self):
        rows = [
            *[
                {
                    "request_id": f"resident-{index}",
                    "runtime_residence": "resident",
                    "tpot_s": 0.001,
                }
                for index in range(2)
            ],
            *[
                {
                    "request_id": f"host-{index}",
                    "runtime_residence": "host_backed",
                    "tpot_s": 0.004,
                }
                for index in range(2)
            ],
            {
                "request_id": "mixed-width",
                "runtime_residence": "resident",
                "tpot_s": 0.002,
            },
            {
                "request_id": "missing",
                "runtime_residence": "resident",
                "tpot_s": 0.002,
            },
            *[
                {
                    "request_id": f"cohort-{cohort}-{member}",
                    "cohort_id": f"cohort-{cohort}",
                    "prefill_batch_size": 2,
                    "prompt_tokens": 100,
                    "ttft_s": 0.02,
                }
                for cohort in range(2)
                for member in range(2)
            ],
            {
                "request_id": "incomplete",
                "cohort_id": "incomplete",
                "prefill_batch_size": 2,
                "prompt_tokens": 100,
                "ttft_s": 0.02,
            },
        ]
        stats = {
            "decode_batch_sizes_by_request": {
                "resident-0": {"4": 3},
                "resident-1": {"4": 2},
                "host-0": {"2": 2},
                "host-1": {"2": 3},
                "mixed-width": {"2": 1, "4": 2},
            },
            "complete_cohort_ids": ["cohort-0", "cohort-1"],
        }

        result = calibrate_profiles(rows, stats, min_samples=2)

        self.assertTrue(result["available"])
        self.assertEqual(
            result["decode_batch_profiles"],
            {
                "2": {"sparse_tokens_per_ms": 0.5},
                "4": {"resident_tokens_per_ms": 4.0},
            },
        )
        self.assertEqual(result["prefill_batch_profiles"], {"2": 10.0})
        evidence = result["evidence"]
        self.assertEqual(evidence["decode"]["accepted_requests"], 4)
        self.assertEqual(evidence["decode"]["ambiguous_batch_size"], 1)
        self.assertEqual(evidence["decode"]["missing_observation"], 6)
        self.assertEqual(evidence["prefill"]["complete_cohorts"], 2)
        self.assertEqual(evidence["prefill"]["incomplete_cohorts"], 1)

        planner = PlannerConfig(
            memory_budget_bytes=1024,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            decode_batch_profiles=result["decode_batch_profiles"],
            prefill_batch_profiles=result["prefill_batch_profiles"],
        )
        self.assertEqual(
            planner.decode_rate_for_batch("generic", ResidencyMode.RESIDENT, 4),
            1.0,
        )
        self.assertEqual(
            planner.decode_rate_for_batch("generic", ResidencyMode.HI_SPARSE, 2),
            0.25,
        )
        self.assertEqual(planner.prefill_rate_for_batch("generic", 2), 5.0)

    def test_retains_samples_below_the_configured_threshold(self):
        result = calibrate_profiles(
            [
                {
                    "request_id": "single",
                    "runtime_residence": "resident",
                    "tpot_s": 0.001,
                }
            ],
            {"decode_batch_sizes_by_request": {"single": {"1": 4}}},
            min_samples=3,
        )

        self.assertFalse(result["available"])
        self.assertEqual(result["decode_batch_profiles"], {})
        self.assertEqual(
            result["evidence"]["decode"]["below_min_samples"],
            ["1:resident"],
        )

    def test_exports_observed_activation_workspace_evidence(self):
        result = calibrate_profiles(
            [],
            {
                "activation_prefill_bytes_per_token_max_observed": 9,
                "activation_prefill_bytes_per_token_samples": 4,
            },
            min_samples=3,
        )

        self.assertTrue(result["available"])
        self.assertEqual(
            result["activation_workspace"],
            {
                "bytes_per_token": 9,
                "samples": 4,
                "aggregation": "observed-maximum",
            },
        )
        self.assertFalse(
            result["evidence"]["activation"]["below_min_samples"]
        )

    def test_holds_activation_profile_below_the_sample_threshold(self):
        result = calibrate_profiles(
            [],
            {
                "activation_prefill_bytes_per_token_max_observed": 9,
                "activation_prefill_bytes_per_token_samples": 2,
            },
            min_samples=3,
        )

        self.assertFalse(result["available"])
        self.assertIsNone(result["activation_workspace"])
        self.assertTrue(
            result["evidence"]["activation"]["below_min_samples"]
        )

    def test_rejects_malformed_activation_workspace_evidence(self):
        with self.assertRaisesRegex(ValueError, "bytes per token"):
            calibrate_profiles(
                [],
                {"activation_prefill_bytes_per_token_max_observed": True},
            )
        with self.assertRaisesRegex(ValueError, "sample count"):
            calibrate_profiles(
                [],
                {
                    "activation_prefill_bytes_per_token_max_observed": 1,
                    "activation_prefill_bytes_per_token_samples": 0,
                },
            )

    def test_calibrates_kind_specific_decode_profiles_with_thresholds(self):
        rows = [
            *[
                {
                    "request_id": f"chat-{index}",
                    "runtime_residence": "resident",
                    "kind": "chat",
                    "tpot_s": 0.002,
                }
                for index in range(2)
            ],
            *[
                {
                    "request_id": f"tool-{index}",
                    "runtime_residence": "resident",
                    "kind": "tool",
                    "tpot_s": 0.001,
                }
                for index in range(2)
            ],
            {
                "request_id": "rare",
                "runtime_residence": "resident",
                "kind": "rare",
                "tpot_s": 0.0005,
            },
        ]
        stats = {
            "decode_batch_sizes_by_request": {
                row["request_id"]: {"4": 2} for row in rows
            }
        }

        result = calibrate_profiles(rows, stats, min_samples=2)

        self.assertEqual(
            result["decode_batch_profiles"],
            {
                "4": {
                    "resident_tokens_per_ms": 4.0,
                    "kind_profiles": {
                        "chat": {"resident_tokens_per_ms": 2.0},
                        "tool": {"resident_tokens_per_ms": 4.0},
                    },
                }
            },
        )
        self.assertEqual(
            result["evidence"]["decode"]["kind_below_min_samples"],
            ["4:resident:rare"],
        )
        planner = PlannerConfig(
            memory_budget_bytes=1024,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            decode_batch_profiles=result["decode_batch_profiles"],
        )
        self.assertEqual(
            planner.decode_rate_for_batch("chat", ResidencyMode.RESIDENT, 4),
            0.5,
        )
        self.assertEqual(
            planner.decode_rate_for_batch("tool", ResidencyMode.RESIDENT, 4),
            1.0,
        )
        self.assertEqual(
            planner.decode_rate_for_batch("rare", ResidencyMode.RESIDENT, 4),
            1.0,
        )

    def test_rejects_invalid_threshold_and_histogram_shape(self):
        with self.assertRaisesRegex(ValueError, "min_samples"):
            calibrate_profiles([], {}, min_samples=0)
        with self.assertRaisesRegex(TypeError, "must be an object"):
            calibrate_profiles(
                [],
                {"decode_batch_sizes_by_request": []},
            )

    def test_cli_writes_a_plan_compatible_profile_file(self):
        rows = [
            {
                "request_id": f"resident-{index}",
                "runtime_residence": "resident",
                "tpot_s": 0.001,
            }
            for index in range(2)
        ]
        payload = {
            "requests": rows,
            "future_stats": {
                "decode_batch_sizes_by_request": {
                    row["request_id"]: {"4": 2} for row in rows
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.json"
            output = root / "planner-profiles.json"
            source.write_text(json.dumps(payload), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                status = calibrate_main(
                    [
                        "--result",
                        str(source),
                        "--min-samples",
                        "2",
                        "--require-available",
                        "--output",
                        str(output),
                    ]
                )

            result = json.loads(output.read_text(encoding="utf-8"))
            planner = PlannerConfig(
                memory_budget_bytes=1024,
                kv_bytes_per_token=1,
                prefill_workspace_bytes_per_token=0,
                decode_batch_profiles=result["decode_batch_profiles"],
                prefill_batch_profiles=result["prefill_batch_profiles"],
            )
            self.assertEqual(status, 0)
            self.assertEqual(
                planner.decode_rate_for_batch(
                    "generic", ResidencyMode.RESIDENT, 4
                ),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
