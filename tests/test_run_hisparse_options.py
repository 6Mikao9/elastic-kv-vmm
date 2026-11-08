import argparse
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_hisparse_smoke import (
    resident_broker_environment,
    resident_broker_manifest,
    resident_hint_environment,
    resident_hint_manifest,
    resident_hint_sources,
    load_future_submissions,
    validate_resident_hint_options,
    validate_resident_broker_options,
)


class ResidentHintRunnerOptionsTest(unittest.TestCase):
    def test_future_workload_and_emitted_plan_are_executable(self):
        workload = {
            "config": {
                "memory_budget_bytes": 1_000_000,
                "kv_bytes_per_token": 100,
                "prefill_workspace_bytes_per_token": 1,
            },
            "requests": [
                {
                    "request_id": "tool-0",
                    "arrival_ms": 10,
                    "prompt_tokens": 128,
                    "output_tokens": 16,
                    "kind": "tool",
                    "deadline_ms": 500,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload_path = root / "workload.json"
            workload_path.write_text(json.dumps(workload), encoding="utf-8")
            submissions, manifest, policy = load_future_submissions(
                None, workload_path, "future"
            )
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps({"rollout": manifest}), encoding="utf-8"
            )
            loaded, loaded_manifest, loaded_policy = load_future_submissions(
                plan_path, None, "fifo"
            )

        self.assertEqual(policy, "future")
        self.assertEqual(loaded_policy, "future")
        self.assertEqual(submissions, loaded)
        self.assertEqual(manifest, loaded_manifest)
        self.assertEqual(loaded[0].deadline_ms, 500)

    def test_environment_manifest_and_sources_are_consistent(self):
        self.assertEqual(
            resident_hint_environment("adaptive", 128, 2),
            {
                "EKV_RESIDENT_HINT_POLICY": "adaptive",
                "EKV_RESIDENT_MIN_HINT_OUTPUT_TOKENS": "128",
            },
        )
        self.assertEqual(
            resident_hint_manifest("adaptive", 128),
            {
                "resident_hint_policy": "adaptive",
                "resident_min_hint_output_tokens": 128,
            },
        )
        self.assertIn("adaptive_sparse/future_residency.py", resident_hint_sources(2))
        self.assertIn("adaptive_sparse/residency_broker.py", resident_hint_sources(2))
        self.assertEqual(resident_hint_environment("off", 128, 0), {})
        self.assertEqual(resident_hint_sources(0), ())

    def test_broker_environment_and_manifest_are_explicit(self):
        self.assertEqual(
            resident_broker_environment("enforce", 256, 2),
            {
                "EKV_RESIDENT_BROKER": "enforce",
                "EKV_RESIDENT_BROKER_LOOKAHEAD_TOKENS": "256",
            },
        )
        self.assertEqual(
            resident_broker_manifest("dry_run", 256),
            {
                "resident_broker_mode": "dry_run",
                "resident_broker_lookahead_tokens": 256,
            },
        )
        self.assertEqual(resident_broker_environment("enforce", 256, 0), {})
        self.assertEqual(resident_broker_environment("off", 256, 2), {})

    def test_adaptive_requires_future_policy_and_residency(self):
        parser = argparse.ArgumentParser()
        cases = (
            ("adaptive", 128, 0, "deadline"),
            ("adaptive", 128, 2, "off"),
            ("off", 0, 2, "deadline"),
        )
        for policy, minimum, limit, future in cases:
            with self.subTest(policy=policy, minimum=minimum, limit=limit, future=future):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    validate_resident_hint_options(
                        parser,
                        policy=policy,
                        minimum_tokens=minimum,
                        resident_limit=limit,
                        future_policy=future,
                    )

    def test_valid_adaptive_configuration_is_accepted(self):
        validate_resident_hint_options(
            argparse.ArgumentParser(),
            policy="adaptive",
            minimum_tokens=128,
            resident_limit=2,
            future_policy="deadline",
        )

    def test_broker_requires_residency_and_positive_lookahead(self):
        parser = argparse.ArgumentParser()
        cases = (("enforce", 256, 0), ("dry_run", 0, 2))
        for mode, lookahead, limit in cases:
            with self.subTest(mode=mode, lookahead=lookahead, limit=limit):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    validate_resident_broker_options(
                        parser,
                        mode=mode,
                        lookahead_tokens=lookahead,
                        resident_limit=limit,
                    )
        validate_resident_broker_options(
            parser, mode="enforce", lookahead_tokens=256, resident_limit=2
        )


if __name__ == "__main__":
    unittest.main()
