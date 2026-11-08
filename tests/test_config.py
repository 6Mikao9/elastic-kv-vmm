import io
import json
from contextlib import redirect_stdout
import os
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
                "EKV_PREFILL_HEADROOM_MIB": "512",
            }
        )
        self.assertEqual(config.chunk_mib, 4)
        self.assertEqual(config.retain_chunks, 8)
        self.assertEqual(config.activation_policy, "both")
        self.assertEqual(config.prefill_headroom_mib, 512)

    def test_activation_policy_requires_v5(self):
        with self.assertRaisesRegex(ValueError, "require elastic mode v5"):
            ElasticConfig(mode="v4", activation_policy="telemetry")

    def test_invalid_integer_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "EKV_CHUNK_MIB must be an integer"):
            ElasticConfig.from_env({"EKV_CHUNK_MIB": "2MiB"})

    def test_cli_overrides_leave_sglang_arguments_untouched(self):
        config, upstream, dry_run = resolve_config(
            [
                "--elastic-mode",
                "v5",
                "--elastic-retain-chunks",
                "8",
                "--model-path",
                "/models/qwen",
                "--port",
                "30000",
            ]
        )
        self.assertEqual((config.mode, config.retain_chunks), ("v5", 8))
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


if __name__ == "__main__":
    unittest.main()
