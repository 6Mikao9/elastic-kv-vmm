import argparse
from contextlib import redirect_stderr
from io import StringIO
import unittest

from scripts.run_hisparse_smoke import (
    resident_hint_environment,
    resident_hint_manifest,
    resident_hint_sources,
    validate_resident_hint_options,
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
        self.assertEqual(resident_hint_environment("off", 128, 0), {})
        self.assertEqual(resident_hint_sources(0), ())

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


if __name__ == "__main__":
    unittest.main()
