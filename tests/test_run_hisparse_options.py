import argparse
from contextlib import redirect_stderr
from io import StringIO
import unittest

from scripts.run_hisparse_smoke import (
    resident_broker_environment,
    resident_broker_manifest,
    resident_hint_environment,
    resident_hint_manifest,
    resident_hint_sources,
    validate_resident_hint_options,
    validate_resident_broker_options,
)


class ResidentHintRunnerOptionsTest(unittest.TestCase):
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
