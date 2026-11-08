import unittest

from scripts.run_case import server_environment


class RunCaseTest(unittest.TestCase):
    def test_baseline_uses_nvml_without_adapter_stats(self):
        env = server_environment(
            "baseline",
            "/tmp/adapter.json",
            {"EKV_STATS_PATH": "/stale/value", "KEEP": "yes"},
        )
        self.assertEqual(env["EKV_MODE"], "baseline")
        self.assertEqual(env["KEEP"], "yes")
        self.assertNotIn("EKV_STATS_PATH", env)

    def test_elastic_mode_publishes_its_own_stats_path(self):
        env = server_environment(
            "v5", "/tmp/v5.json", {"EKV_STATS_PATH": "/stale/value"}
        )
        self.assertEqual(env["EKV_STATS_PATH"], "/tmp/v5.json")
        self.assertEqual(env["EKV_MODE"], "v5")


if __name__ == "__main__":
    unittest.main()
