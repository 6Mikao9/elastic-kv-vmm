import io
import json
from contextlib import redirect_stdout
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from elastic_kv.config import ElasticConfig
from elastic_kv import launch
from elastic_kv.launch import main, resolve_config


class ElasticConfigTest(unittest.TestCase):
    def test_defaults_and_environment(self):
        config = ElasticConfig.from_env({})
        self.assertEqual(config.as_dict()["mode"], "baseline")
        config = ElasticConfig.from_env(
            {
                "EKV_MODE": "v5",
                "EKV_CHUNK_MIB": "4",
                "EKV_RETAIN_CHUNKS": "8",
                "EKV_ACTIVATION_POLICY": "both",
                "EKV_ADMISSION_POLICY": "physical",
                "EKV_PREFILL_HEADROOM_MIB": "512",
                "EKV_PREFILL_HEADROOM_SCOPED": "1",
                "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN": "1536",
                "EKV_PREFILL_WORKSPACE_MAX_MIB": "768",
                "EKV_PRECOMMIT_CHUNKS": "2",
                "EKV_PRECOMMIT_MAX_CHUNKS": "3",
                "EKV_FUTURE_POLICY": "deadline",
                "EKV_FUTURE_ORDERING_HORIZON_MS": "2000",
                "EKV_FUTURE_STATS_PATH": "/tmp/future.json",
                "EKV_GLOBAL_BROKER_PATH": "/tmp/gpu0-broker.sqlite",
                "EKV_GLOBAL_BROKER_MIB": "6144",
                "EKV_GLOBAL_BROKER_STALE_SECONDS": "90",
                "EKV_GLOBAL_BROKER_FAIR_MODE": "enforce",
                "EKV_GLOBAL_BROKER_TENANT": "qwen-a",
                "EKV_GLOBAL_BROKER_TENANT_WEIGHT": "3.5",
                "EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS": "7.5",
                "EKV_GLOBAL_BROKER_WAIT_MS": "125",
            }
        )
        self.assertEqual(config.chunk_mib, 4)
        self.assertEqual(config.retain_chunks, 8)
        self.assertEqual(config.activation_policy, "both")
        self.assertEqual(config.admission_policy, "physical")
        self.assertEqual(config.prefill_headroom_mib, 512)
        self.assertTrue(config.prefill_headroom_scoped)
        self.assertEqual(config.prefill_workspace_bytes_per_token, 1536)
        self.assertEqual(config.prefill_workspace_max_mib, 768)
        self.assertEqual(config.precommit_chunks, 2)
        self.assertEqual(config.precommit_max_chunks, 3)
        self.assertEqual(config.future_policy, "deadline")
        self.assertEqual(config.future_ordering_horizon_ms, 2000)
        self.assertEqual(config.future_stats_path, "/tmp/future.json")
        self.assertEqual(config.global_broker_path, "/tmp/gpu0-broker.sqlite")
        self.assertEqual(config.global_broker_mib, 6144)
        self.assertEqual(config.global_broker_stale_seconds, 90)
        self.assertEqual(config.global_broker_fair_mode, "enforce")
        self.assertEqual(config.global_broker_tenant, "qwen-a")
        self.assertEqual(config.global_broker_tenant_weight, 3.5)
        self.assertEqual(config.global_broker_demand_ttl_seconds, 7.5)
        self.assertEqual(config.global_broker_wait_ms, 125)

    def test_activation_policy_requires_v5(self):
        with self.assertRaisesRegex(ValueError, "require elastic mode v5"):
            ElasticConfig(mode="v4", activation_policy="telemetry")
        with self.assertRaisesRegex(ValueError, "physical admission requires"):
            ElasticConfig(mode="v4", admission_policy="physical")

    def test_invalid_integer_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "EKV_CHUNK_MIB must be an integer"):
            ElasticConfig.from_env({"EKV_CHUNK_MIB": "2MiB"})

    def test_future_policy_is_validated(self):
        with self.assertRaisesRegex(ValueError, "unknown future policy"):
            ElasticConfig(future_policy="predict-everything")

    def test_precommit_budget_is_validated(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 4"):
            ElasticConfig(precommit_chunks=5)
        with self.assertRaisesRegex(ValueError, "must cover"):
            ElasticConfig(precommit_chunks=2, precommit_max_chunks=1)

    def test_dynamic_prefill_workspace_requires_v5_physical_admission(self):
        with self.assertRaisesRegex(ValueError, "dynamic prefill workspace requires"):
            ElasticConfig(
                mode="v4",
                prefill_headroom_scoped=True,
            )
        with self.assertRaisesRegex(ValueError, "requires physical admission"):
            ElasticConfig(
                mode="v5",
                prefill_workspace_bytes_per_token=1,
            )

    def test_global_broker_requires_path_and_positive_capacity_together(self):
        with self.assertRaisesRegex(ValueError, "requires both a path"):
            ElasticConfig(global_broker_path="/tmp/gpu0.sqlite")
        with self.assertRaisesRegex(ValueError, "requires both a path"):
            ElasticConfig(global_broker_mib=1024)

    def test_fair_global_broker_requires_tenant_and_matching_mode(self):
        with self.assertRaisesRegex(ValueError, "requires a tenant"):
            ElasticConfig(global_broker_fair_mode="enforce")
        with self.assertRaisesRegex(ValueError, "enabled broker path"):
            ElasticConfig(
                global_broker_fair_mode="enforce",
                global_broker_tenant="qwen",
            )
        config = ElasticConfig(
            global_broker_path="/tmp/gpu0.sqlite",
            global_broker_mib=1024,
            global_broker_fair_mode="enforce",
            global_broker_tenant="qwen",
        )
        self.assertEqual(config.as_env()["EKV_GLOBAL_BROKER_TENANT"], "qwen")
        with self.assertRaisesRegex(ValueError, "between 0 and 5000"):
            ElasticConfig(global_broker_wait_ms=5001)

    def test_cli_overrides_leave_sglang_arguments_untouched(self):
        config, upstream, dry_run = resolve_config(
            [
                "--elastic-mode",
                "v5",
                "--elastic-retain-chunks",
                "8",
                "--elastic-admission-policy",
                "physical",
                "--elastic-prefill-headroom-scoped",
                "--elastic-prefill-workspace-bytes-per-token",
                "1536",
                "--elastic-prefill-workspace-max-mib",
                "768",
                "--elastic-future-policy",
                "deadline",
                "--elastic-precommit-chunks",
                "2",
                "--elastic-precommit-max-chunks",
                "4",
                "--elastic-future-ordering-horizon-ms",
                "1500",
                "--elastic-future-stats-path",
                "/tmp/future.json",
                "--elastic-global-broker-path",
                "/tmp/gpu0.sqlite",
                "--elastic-global-broker-mib",
                "6144",
                "--elastic-global-broker-stale-seconds",
                "90",
                "--elastic-global-broker-fair-mode",
                "enforce",
                "--elastic-global-broker-tenant",
                "qwen-a",
                "--elastic-global-broker-tenant-weight",
                "2.5",
                "--elastic-global-broker-demand-ttl-seconds",
                "8",
                "--elastic-global-broker-wait-ms",
                "150",
                "--model-path",
                "/models/qwen",
                "--port",
                "30000",
            ]
        )
        self.assertEqual(
            (
                config.mode,
                config.retain_chunks,
                config.admission_policy,
                config.future_policy,
                config.precommit_chunks,
                config.precommit_max_chunks,
                config.future_ordering_horizon_ms,
                config.future_stats_path,
                config.prefill_headroom_scoped,
                config.prefill_workspace_bytes_per_token,
                config.prefill_workspace_max_mib,
                config.global_broker_path,
                config.global_broker_mib,
                config.global_broker_stale_seconds,
                config.global_broker_fair_mode,
                config.global_broker_tenant,
                config.global_broker_tenant_weight,
                config.global_broker_demand_ttl_seconds,
                config.global_broker_wait_ms,
            ),
            (
                "v5",
                8,
                "physical",
                "deadline",
                2,
                4,
                1500,
                "/tmp/future.json",
                True,
                1536,
                768,
                "/tmp/gpu0.sqlite",
                6144,
                90,
                "enforce",
                "qwen-a",
                2.5,
                8.0,
                150,
            ),
        )
        self.assertEqual(upstream, ["--model-path", "/models/qwen", "--port", "30000"])
        self.assertFalse(dry_run)

    def test_dry_run_does_not_import_sglang(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--elastic-mode", "v5", "--elastic-dry-run"])
        self.assertEqual(status, 0)
        payload = output.getvalue().split(": ", 1)[1]
        self.assertEqual(json.loads(payload)["mode"], "v5")

    def test_spawn_worker_installs_only_after_parent_sentinel(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            launch, "_install_adapter"
        ) as install:
            launch._install_spawned_worker()
            install.assert_not_called()
        with patch.dict(
            os.environ,
            {launch.SPAWN_INSTALL_ENV: "1", "EKV_MODE": "baseline"},
            clear=True,
        ), patch.object(launch, "_install_adapter") as install:
            launch._install_spawned_worker()
            install.assert_called_once_with()

    def test_baseline_future_policy_skips_pinned_vmm_adapter(self):
        vmm_install = unittest.mock.Mock()
        future_install = unittest.mock.Mock()
        modules = {
            "elastic_kv.adapter": SimpleNamespace(install=vmm_install),
            "adaptive_sparse.sglang_future_hook": SimpleNamespace(install=future_install),
        }
        with patch.dict(
            os.environ,
            {"EKV_MODE": "baseline", "EKV_FUTURE_POLICY": "deadline"},
            clear=True,
        ), patch.dict(sys.modules, modules):
            launch._install_adapter()
        vmm_install.assert_not_called()
        future_install.assert_called_once_with()

    def test_nonbaseline_mode_installs_vmm_adapter(self):
        vmm_install = unittest.mock.Mock()
        modules = {"elastic_kv.adapter": SimpleNamespace(install=vmm_install)}
        with patch.dict(os.environ, {"EKV_MODE": "v5"}, clear=True), patch.dict(
            sys.modules, modules
        ):
            launch._install_adapter()
        vmm_install.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
