import argparse
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from scripts.run_dual import (
    broker_environment,
    read_broker_snapshot,
    validate_broker_options,
    weighted_targets,
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


if __name__ == "__main__":
    unittest.main()
