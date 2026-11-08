import argparse
import sqlite3
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.run_dual import (
    adapter_stats,
    broker_environment,
    phase_stem,
    read_broker_snapshot,
    run_correctness_probes,
    run_synchronized,
    scenario_order,
    server_launch_spec,
    source_hashes,
    source_manifest,
    summarize_phase,
    validate_broker_options,
    wait_for_broker_owners,
    weighted_targets,
    write_phase_artifacts,
)


class DualBrokerRunnerTest(unittest.TestCase):
    def test_broker_environment_is_explicit_per_tenant(self):
        self.assertEqual(
            broker_environment(
                "broker.sqlite",
                6144,
                "observe",
                "model-a",
                3.0,
                120,
                10.0,
                0,
            ),
            {
                "EKV_GLOBAL_BROKER_PATH": "broker.sqlite",
                "EKV_GLOBAL_BROKER_MIB": "6144",
                "EKV_GLOBAL_BROKER_FAIR_MODE": "observe",
                "EKV_GLOBAL_BROKER_TENANT_WEIGHT": "3.0",
                "EKV_GLOBAL_BROKER_STALE_SECONDS": "120",
                "EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS": "10.0",
                "EKV_GLOBAL_BROKER_WAIT_MS": "0",
                "EKV_GLOBAL_BROKER_TENANT": "model-a",
            },
        )
        self.assertEqual(
            broker_environment(None, 0, "off", None, 1, 120, 10, 0), {}
        )

    def test_weighted_targets_match_broker_integer_split(self):
        self.assertEqual(weighted_targets(100, {"a": 3, "b": 1}), {"a": 75, "b": 25})
        self.assertEqual(weighted_targets(10, {"a": 1, "b": 1, "c": 1}),
                         {"a": 4, "b": 3, "c": 3})
        self.assertEqual(weighted_targets(100, {}), {})

    def test_snapshot_reads_capacity_usage_demand_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broker.sqlite"
            now = time.time()
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO metadata VALUES(?, ?)",
                    (("capacity_bytes", "100"), ("fair_mode", "enforce")),
                )
                connection.execute(
                    "CREATE TABLE owners(owner TEXT PRIMARY KEY, pid INTEGER, "
                    "last_seen REAL, tenant_id TEXT)"
                )
                connection.executemany(
                    "INSERT INTO owners VALUES(?, ?, ?, ?)",
                    (("owner-a", 101, now, "a"), ("owner-b", 102, now, "b")),
                )
                connection.execute(
                    "CREATE TABLE leases(owner TEXT, lease_key TEXT, bytes INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO leases VALUES(?, ?, ?)",
                    (("owner-a", "a-0", 70), ("owner-b", "b-0", 20)),
                )
                connection.execute(
                    "CREATE TABLE tenants(tenant_id TEXT PRIMARY KEY, weight REAL)"
                )
                connection.executemany(
                    "INSERT INTO tenants VALUES(?, ?)", (("a", 3), ("b", 1))
                )
                connection.execute(
                    "CREATE TABLE demands(owner TEXT, tenant_id TEXT, "
                    "requested_bytes INTEGER, expires_at REAL)"
                )
                connection.execute(
                    "INSERT INTO demands VALUES(?, ?, ?, ?)",
                    ("owner-b-demand", "b", 30, now + 10),
                )
                connection.commit()
            finally:
                connection.close()
            snapshot = read_broker_snapshot(path)
        self.assertEqual(snapshot["status"], "ready")
        self.assertEqual(snapshot["outstanding_bytes"], 90)
        self.assertEqual(snapshot["available_bytes"], 10)
        self.assertEqual(snapshot["tenant_usage_bytes"], {"a": 70, "b": 20})
        self.assertEqual(snapshot["tenant_demand_bytes"], {"b": 30})
        self.assertEqual(snapshot["fair_target_bytes"], {"a": 75, "b": 25})
        self.assertEqual(snapshot["tenant_deficit_bytes"], {"a": 5, "b": 5})
        self.assertEqual(snapshot["starved_tenants"], ["b"])
        self.assertEqual(snapshot["tenants_over_target_with_demand"], [])
        self.assertFalse(snapshot["overcommitted"])
        self.assertEqual(snapshot["owners"]["owner-a"]["leased_bytes"], 70)
        self.assertEqual(snapshot["owners"]["owner-b"]["lease_count"], 1)

    def test_missing_snapshot_is_uninitialized(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = read_broker_snapshot(Path(directory) / "missing.sqlite")
        self.assertEqual(snapshot, {"status": "uninitialized"})

    def test_broker_options_reject_partial_or_ambiguous_configuration(self):
        parser = argparse.ArgumentParser()
        cases = (
            dict(mode="v2", path=None, mib=1, fair_mode="off", tenants=(None, None)),
            dict(mode="v2", path=None, mib=-1, fair_mode="off", tenants=(None, None)),
            dict(mode="v2", path=Path("broker.sqlite"), mib=0,
                 fair_mode="off", tenants=(None, None)),
            dict(mode="v2", path=Path("broker.sqlite"), mib=-1,
                 fair_mode="off", tenants=(None, None)),
            dict(mode="baseline", path=Path("broker.sqlite"), mib=1,
                 fair_mode="off", tenants=(None, None)),
            dict(mode="v2", path=Path("broker.sqlite"), mib=1,
                 fair_mode="enforce", tenants=(None, None)),
            dict(mode="v2", path=Path("broker.sqlite"), mib=1,
                 fair_mode="enforce", tenants=("same", "same")),
            dict(mode="v2", path=Path("broker.sqlite"), mib=1,
                 fair_mode="off", tenants=("a", None)),
            dict(mode="v2", path=Path("broker.sqlite"), mib=1,
                 fair_mode="enforce", tenants=("", "b")),
        )
        for values in cases:
            with self.subTest(values=values):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    validate_broker_options(
                        parser,
                        weights=(1.0, 1.0),
                        stale_seconds=120,
                        demand_ttl_seconds=10,
                        wait_ms=0,
                        **values,
                    )

    def test_valid_broker_configuration_is_accepted(self):
        validate_broker_options(
            argparse.ArgumentParser(),
            mode="v2",
            path=Path("broker.sqlite"),
            mib=1024,
            fair_mode="enforce",
            tenants=("model-a", "model-b"),
            weights=(3.0, 1.0),
            stale_seconds=120,
            demand_ttl_seconds=10,
            wait_ms=0,
        )

    def test_source_manifest_contains_serving_control_plane(self):
        hashes = source_hashes()
        self.assertIn("elastic_kv/adapter.py", hashes)
        self.assertIn("elastic_kv/config.py", hashes)
        self.assertIn("scripts/run_dual.py", hashes)
        self.assertIn("requirements.txt", hashes)

    def test_git_source_manifest_handles_non_ascii_worktree(self):
        manifest = source_manifest()
        self.assertTrue(manifest["available"])
        self.assertRegex(manifest["commit"], r"^[0-9a-f]{40}$")
        self.assertIn("elastic_kv/adapter.py", manifest["files"])
        self.assertRegex(manifest["tree_sha256"], r"^[0-9a-f]{64}$")

    def test_adapter_stats_preserve_broker_and_mapping_counters(self):
        stats = adapter_stats({
            "backed_bytes": 4096,
            "live_tokens": 3,
            "ensure_calls": 2,
            "ensure_chunks": 4,
            "reclaim_calls": 1,
            "reclaim_chunks": 2,
            "global_broker_reservations": 6,
            "global_broker_denials": 1,
            "global_broker_wait_ns": 17,
            "activation_reserved_bytes": 8192,
            "mode": "v2",
            "layout": "kv-sink",
            "admission_policy": "graceful",
            "unsafe/path": "/private/model",
        })
        self.assertEqual(stats["global_broker_reservations"], 6)
        self.assertEqual(stats["global_broker_denials"], 1)
        self.assertEqual(stats["ensure_chunks"], 4)
        self.assertEqual(stats["activation_reserved_bytes"], 8192)
        self.assertNotIn("unsafe/path", stats)

    def test_adapter_stats_reject_invalid_token_count(self):
        with self.assertRaisesRegex(ValueError, "live_tokens"):
            adapter_stats({"backed_bytes": 0, "live_tokens": -1})

    def test_load_order_reverses_hot_instances_but_keeps_joint_phase_last(self):
        self.assertEqual(
            scenario_order(1), ("a-hot", "b-hot", "both-active")
        )
        self.assertEqual(
            scenario_order(2), ("b-hot", "a-hot", "both-active")
        )
        self.assertEqual(phase_stem("a-hot", 2), "repeat-02-a-hot")
        self.assertEqual(phase_stem("a-hot", 0, True), "warmup-a-hot")

    def test_launch_spec_records_command_and_explicit_environment(self):
        spec = server_launch_spec(
            Path("/models/a"), "v2", 31080, Path("/tmp/a.stats"),
            {"EKV_GLOBAL_BROKER_PATH": "/tmp/broker.sqlite"},
        )
        self.assertIn("--disable-overlap-schedule", spec["command"])
        self.assertEqual(spec["environment"]["EKV_MODE"], "v2")
        self.assertEqual(
            spec["environment"]["EKV_GLOBAL_BROKER_PATH"],
            "/tmp/broker.sqlite",
        )

    def test_synchronized_dispatch_releases_all_workers_together(self):
        records, starts = run_synchronized(list(range(6)), lambda value: value * 2)
        self.assertEqual(records, [0, 2, 4, 6, 8, 10])
        self.assertEqual(len(starts), 6)
        self.assertLess(max(starts) - min(starts), 0.5)

    def test_phase_summary_exposes_raw_artifacts_and_aggregates(self):
        request = {
            "ttft_s": 0.01,
            "tpot_s": 0.02,
            "end_to_end_s": 0.5,
        }
        result = {
            "status": "ok",
            "scenario": "a-hot",
            "repeat": 1,
            "warmup": False,
            "elapsed_s": 0.5,
            "drain_s": 0.1,
            "dispatch_skew_s": 0.001,
            "input_tokens": 1024,
            "output_tokens": 128,
            "output_tokens_per_s": 256,
            "requests": [request],
            "samples": [{
                "offset_s": 0.2,
                "device_used_bytes": 4096,
                "instances": {"a": {"backed_bytes": 2048, "live_tokens": 3}},
            }],
            "active_window_s": [0.1, 0.3],
            "active_sample_count": 1,
            "peak_device_used_bytes": 4096,
            "broker_snapshot_status": {"ready": 1},
            "broker_failure_ratio": 0.0,
            "idle": {"device_used_bytes": 2048, "instances": {}},
        }
        summary = summarize_phase(result)
        self.assertEqual(summary["scenario"], "a-hot")
        self.assertNotIn("requests", summary)
        self.assertEqual(summary["summary"]["instances"]["a"]["peak_live_tokens"], 3)
        self.assertIn("p95", summary["summary"]["ttft_s"])

    def test_phase_artifacts_are_split_from_summary(self):
        result = {
            "status": "ok", "scenario": "a-hot", "repeat": 1,
            "warmup": False, "elapsed_s": 0.5, "drain_s": 0.1,
            "dispatch_skew_s": 0.001,
            "input_tokens": 1024, "output_tokens": 128,
            "output_tokens_per_s": 256, "requests": [{
                "request_id": "r", "ttft_s": 0.01,
                "tpot_s": 0.02, "end_to_end_s": 0.5,
            }],
            "samples": [{"offset_s": 0.2, "device_used_bytes": 4096,
                          "instances": {"a": {"backed_bytes": 0, "live_tokens": 0}}}],
            "active_window_s": [0.1, 0.3], "active_sample_count": 1,
            "peak_device_used_bytes": 4096,
            "broker_snapshot_status": {}, "broker_failure_ratio": None,
            "idle": {"device_used_bytes": 2048, "instances": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = write_phase_artifacts(output, result)
            requests = Path(output / "repeat-01-a-hot.requests.json")
            samples = Path(output / "repeat-01-a-hot.samples.json")
            self.assertTrue(requests.is_file())
            self.assertTrue(samples.is_file())
            self.assertEqual(summary["scenario"], "a-hot")

    def test_container_dual_forwards_image_identity(self):
        script = Path("scripts/container_dual.sh").read_text(encoding="utf-8")
        self.assertIn("docker image inspect --format '{{.Id}}'", script)
        self.assertIn("EKV_IMAGE_ID=$image_id", script)
        self.assertIn("EKV_IMAGE_DIGEST=$image_digest", script)

    def test_correctness_probe_runs_fixed_nonstreaming_matrix(self):
        def fake_generate(url, spec, timeout, stream=True):
            self.assertFalse(stream)
            return {
                **spec,
                "_started_perf_s": 0.0,
                "output_ids_complete": True,
                "output_ids_sha256": "a" * 64,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.run_dual.generate", side_effect=fake_generate):
                summary = run_correctness_probes(output)
            artifact = Path(output / "correctness.json")
            self.assertTrue(artifact.is_file())
            self.assertEqual(summary["instances"]["a"]["probe_count"], 6)
            self.assertEqual(summary["instances"]["b"]["probe_count"], 6)

    def test_broker_wait_requires_both_configured_tenants(self):
        snapshots = [
            {"status": "ready", "owners": {"a": {"tenant_id": "tenant-a"}}},
            {"status": "ready", "owners": {
                "a": {"tenant_id": "tenant-a"},
                "b": {"tenant_id": "tenant-b"},
            }},
        ]
        with patch("scripts.run_dual.read_broker_snapshot", side_effect=snapshots):
            snapshot = wait_for_broker_owners(
                Path("broker.sqlite"), ("tenant-a", "tenant-b"), timeout=1
            )
        self.assertEqual(set(snapshot["owners"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
