import sys
from pathlib import Path
import unittest

from scripts.run_case import (
    HOST,
    PORT,
    build_server_command,
    server_environment,
)


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

    def test_future_stats_have_a_separate_path(self):
        env = server_environment(
            "baseline",
            "/tmp/adapter.json",
            {"EKV_FUTURE_STATS_PATH": "/stale/future.json"},
            future_stats_path="/tmp/future.json",
        )
        self.assertEqual(env["EKV_FUTURE_STATS_PATH"], "/tmp/future.json")
        self.assertNotIn("EKV_STATS_PATH", env)

    def test_overlap_opt_in_replaces_a_stale_value(self):
        env = server_environment(
            "v5",
            "/tmp/v5.json",
            {"EKV_ALLOW_OVERLAP_SCHEDULE": "0"},
            allow_overlap_schedule=True,
        )
        self.assertEqual(env["EKV_ALLOW_OVERLAP_SCHEDULE"], "1")

    def test_overlap_is_disabled_unless_requested(self):
        env = server_environment(
            "v5",
            "/tmp/v5.json",
            {"EKV_ALLOW_OVERLAP_SCHEDULE": "1"},
        )
        self.assertNotIn("EKV_ALLOW_OVERLAP_SCHEDULE", env)

    def test_server_command_keeps_host_value_adjacent_by_default(self):
        command = build_server_command(
            "models/qwen", "eager", prefill_chunk=1024
        )
        host_index = command.index("--host")
        self.assertEqual(command[host_index + 1], HOST)
        self.assertGreater(command.index("--disable-overlap-schedule"), host_index)
        self.assertIn("--disable-cuda-graph", command)
        self.assertEqual(command[0], sys.executable)
        self.assertIn(str(Path("models/qwen").resolve()), command)

    def test_server_command_omits_disable_flag_for_overlap(self):
        command = build_server_command(
            "models/qwen", "graph", allow_overlap_schedule=True
        )
        self.assertNotIn("--disable-overlap-schedule", command)
        port_index = command.index("--port")
        self.assertEqual(command[port_index + 1], str(PORT))
        self.assertIn("--disable-cuda-graph-padding", command)


if __name__ == "__main__":
    unittest.main()
