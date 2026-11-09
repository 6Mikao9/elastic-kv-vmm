import argparse
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_hisparse_smoke import (
    configure_profile_environment,
    diagnostic_capture_arguments,
    global_broker_environment,
    global_broker_manifest,
    resident_broker_environment,
    resident_broker_manifest,
    resident_hint_environment,
    resident_hint_manifest,
    resident_hint_sources,
    load_future_submissions,
    server_base_url,
    shared_prefix_environment,
    shared_prefix_manifest,
    validate_profile_output_options,
    validate_resident_hint_options,
    validate_resident_broker_options,
    validate_shared_prefix_reuse_options,
    validate_global_broker_options,
    validate_server_endpoint,
)


class ResidentHintRunnerOptionsTest(unittest.TestCase):
    def test_profile_output_defaults_and_opt_in(self):
        parser = argparse.ArgumentParser()
        for workload, cap, policy, expected in (
            (None, None, None, (None, None)),
            ("mixed", None, None, (16, "clip")),
            ("mixed12k_unequal", 32, "proportional", (32, "proportional")),
        ):
            with self.subTest(workload=workload, cap=cap, policy=policy):
                args = argparse.Namespace(profile_workload=workload, profile_output_token_cap=cap,
                                          profile_output_length_policy=policy)
                validate_profile_output_options(parser, args)
                self.assertEqual((args.profile_output_token_cap, args.profile_output_length_policy),
                                 expected)

    def test_profile_output_options_require_a_profile_and_valid_cap(self):
        parser = argparse.ArgumentParser()
        for workload, cap, policy, message in (
            (None, None, "clip", "requires --profile-workload"),
            (None, 32, None, "token cap requires profiling"),
            ("mixed", 33, None, "value from 1 to 32"),
        ):
            with self.subTest(workload=workload, cap=cap, policy=policy):
                args = argparse.Namespace(profile_workload=workload, profile_output_token_cap=cap,
                                          profile_output_length_policy=policy)
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    validate_profile_output_options(parser, args)
                self.assertIn(message, stderr.getvalue())

    def test_profile_ranges_cannot_leak_into_performance_runs(self):
        environment = {"EKV_PROFILE_RANGES": "1", "CUDA_VISIBLE_DEVICES": "GPU-test"}
        configure_profile_environment(environment, None)
        self.assertNotIn("EKV_PROFILE_RANGES", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-test")
        configure_profile_environment(environment, "mixed")
        self.assertEqual(environment["EKV_PROFILE_RANGES"], "1")
        self.assertEqual(environment["SGLANG_PROFILE_V2"], "0")
        configure_profile_environment(environment, "")
        self.assertNotIn("EKV_PROFILE_RANGES", environment)

    def test_server_base_url_supports_explicit_ports_and_wildcards(self):
        self.assertEqual(server_base_url("127.0.0.1", 31081), "http://127.0.0.1:31081")
        self.assertEqual(server_base_url("0.0.0.0", 31082), "http://127.0.0.1:31082")
        self.assertEqual(server_base_url("::", 31083), "http://[::1]:31083")

    def test_server_endpoint_rejects_invalid_host_and_port(self):
        parser = argparse.ArgumentParser()
        for host, port in (("", 31080), (" localhost", 31080), ("127.0.0.1", 0),
                           ("127.0.0.1", 65536)):
            with self.subTest(host=host, port=port):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    validate_server_endpoint(parser, host=host, port=port)
        validate_server_endpoint(parser, host="127.0.0.1", port=31081)

    def test_diagnostic_capture_uses_the_launched_endpoint(self):
        self.assertEqual(
            diagnostic_capture_arguments(
                "http://127.0.0.1:31082", Path("run/determinism.json"), 3
            ),
            [
                "--url", "http://127.0.0.1:31082/generate",
                "--output", str(Path("run/determinism.json")),
                "--captures", "3",
            ],
        )

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

    def test_precomputed_prefix_reuse_plan_requires_live_radix_cache(self):
        manifest = {
            "policy": "fifo",
            "planner_config": {
                "memory_budget_bytes": 1_000_000,
                "kv_bytes_per_token": 100,
                "prefill_workspace_bytes_per_token": 0,
                "active_prefix_prefill_reuse": True,
            },
            "submissions": [
                {
                    "request_id": "shared",
                    "planned_arrival_ms": 0,
                    "submit_at_ms": 0,
                    "estimated_finish_ms": 10,
                    "kind": "generic",
                    "prompt_tokens": 16,
                    "output_tokens": 1,
                    "hint": {
                        "release_after_ms": 0,
                        "expected_output_tokens": 1,
                        "kind": "generic",
                        "prefix_group": "system",
                        "shared_prefix_tokens": 8,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan_path.write_text(
                json.dumps({"rollout": manifest}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "radix cache"):
                load_future_submissions(plan_path, None, "fifo")

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
        self.assertIn("adaptive_sparse/shared_prefix.py", resident_hint_sources(2))
        self.assertEqual(resident_hint_environment("off", 128, 0), {})
        self.assertEqual(resident_hint_sources(0), ())

    def test_broker_environment_and_manifest_are_explicit(self):
        self.assertEqual(
            resident_broker_environment("enforce", 256, 2, "enforce"),
            {
                "EKV_RESIDENT_BROKER": "enforce",
                "EKV_RESIDENT_BROKER_LOOKAHEAD_TOKENS": "256",
                "EKV_RESIDENT_BROKER_GLOBAL_MODE": "enforce",
            },
        )
        self.assertEqual(
            resident_broker_manifest("dry_run", 256, "observe"),
            {
                "resident_broker_mode": "dry_run",
                "resident_broker_lookahead_tokens": 256,
                "resident_broker_global_mode": "observe",
            },
        )
        self.assertEqual(resident_broker_environment("enforce", 256, 0), {})
        self.assertEqual(resident_broker_environment("off", 256, 2), {})

    def test_shared_prefix_environment_and_manifest_are_explicit(self):
        self.assertEqual(
            shared_prefix_environment("direct", 2),
            {"EKV_SHARED_PREFIX_REUSE": "direct"},
        )
        self.assertEqual(
            shared_prefix_manifest("direct"), {"shared_prefix_reuse": "direct"}
        )
        self.assertEqual(shared_prefix_environment("direct", 0), {})
        self.assertEqual(shared_prefix_environment("off", 2), {})

    def test_global_broker_environment_and_manifest_are_explicit(self):
        self.assertEqual(
            global_broker_environment(
                "/run/ekv/gpu-{gpu}.sqlite",
                6144,
                120,
                "enforce",
                "model-a",
                3.0,
                10.0,
                0,
            ),
            {
                "EKV_GLOBAL_BROKER_PATH": "/run/ekv/gpu-{gpu}.sqlite",
                "EKV_GLOBAL_BROKER_MIB": "6144",
                "EKV_GLOBAL_BROKER_STALE_SECONDS": "120",
                "EKV_GLOBAL_BROKER_FAIR_MODE": "enforce",
                "EKV_GLOBAL_BROKER_TENANT": "model-a",
                "EKV_GLOBAL_BROKER_TENANT_WEIGHT": "3.0",
                "EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS": "10.0",
                "EKV_GLOBAL_BROKER_WAIT_MS": "0",
            },
        )
        manifest = global_broker_manifest(
            "/run/ekv/gpu-{gpu}.sqlite",
            6144,
            120,
            "observe",
            "model-a",
            3.0,
            10.0,
            25,
        )
        self.assertEqual(manifest["global_broker_fair_mode"], "observe")
        self.assertEqual(manifest["global_broker_wait_ms"], 25)
        self.assertEqual(
            global_broker_environment(None, 0, 120, "off", None, 1, 10, 0),
            {},
        )

    def test_global_broker_options_require_a_consistent_vmm_configuration(self):
        parser = argparse.ArgumentParser()
        base = {
            "path": None,
            "mib": 0,
            "stale_seconds": 120,
            "fair_mode": "off",
            "tenant": None,
            "tenant_weight": 1.0,
            "demand_ttl_seconds": 10.0,
            "wait_ms": 0,
            "vmm_backing": True,
        }
        invalid = (
            {"path": "/tmp/broker.sqlite"},
            {"mib": 1024},
            {
                "path": "/tmp/broker.sqlite",
                "mib": 1024,
                "fair_mode": "enforce",
            },
            {
                "path": "/tmp/broker.sqlite",
                "mib": 1024,
                "vmm_backing": False,
            },
            {
                "path": "/tmp/broker.sqlite",
                "mib": 1024,
                "fair_mode": "observe",
                "tenant": "a",
                "wait_ms": 5001,
            },
        )
        for override in invalid:
            with self.subTest(override=override):
                case = {**base, **override}
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    validate_global_broker_options(parser, **case)
        validate_global_broker_options(
            parser,
            **{
                **base,
                "path": "/tmp/broker.sqlite",
                "mib": 1024,
                "fair_mode": "observe",
                "tenant": "a",
            },
        )

    def test_shared_prefix_requires_full_runtime_enablement(self):
        parser = argparse.ArgumentParser()
        invalid = (
            ("invalid", "adaptive", "deadline", 2, False),
            ("direct", "off", "deadline", 2, False),
            ("direct", "adaptive", "off", 2, False),
            ("direct", "adaptive", "deadline", 0, False),
            ("direct", "adaptive", "deadline", 2, True),
        )
        for mode, hints, future, limit, full_batch in invalid:
            with self.subTest(
                mode=mode, hints=hints, future=future, limit=limit,
                full_batch=full_batch,
            ):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    validate_shared_prefix_reuse_options(
                        parser,
                        mode=mode,
                        hint_policy=hints,
                        future_policy=future,
                        resident_limit=limit,
                        full_batch_swap=full_batch,
                    )
        validate_shared_prefix_reuse_options(
            parser,
            mode="direct",
            hint_policy="adaptive",
            future_policy="deadline",
            resident_limit=2,
            full_batch_swap=False,
        )

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
        cases = (
            ("enforce", 256, "off", 0, False, None),
            ("dry_run", 0, "off", 2, False, None),
            ("off", 256, "observe", 2, True, "/run/ekv/gpu.sqlite"),
            ("dry_run", 256, "enforce", 2, True, "/run/ekv/gpu.sqlite"),
            ("enforce", 256, "observe", 2, False, "/run/ekv/gpu.sqlite"),
            ("enforce", 256, "observe", 2, True, None),
        )
        for mode, lookahead, global_mode, limit, vmm, path in cases:
            with self.subTest(
                mode=mode,
                lookahead=lookahead,
                global_mode=global_mode,
                limit=limit,
                vmm=vmm,
                path=path,
            ):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    validate_resident_broker_options(
                        parser,
                        mode=mode,
                        lookahead_tokens=lookahead,
                        global_mode=global_mode,
                        resident_limit=limit,
                        vmm_backing=vmm,
                        global_broker_path=path,
                    )
        validate_resident_broker_options(
            parser,
            mode="dry_run",
            lookahead_tokens=256,
            global_mode="observe",
            resident_limit=2,
            vmm_backing=True,
            global_broker_path="/run/ekv/gpu.sqlite",
        )
        validate_resident_broker_options(
            parser,
            mode="enforce",
            lookahead_tokens=256,
            global_mode="off",
            resident_limit=2,
            vmm_backing=False,
            global_broker_path=None,
        )
        validate_resident_broker_options(
            parser,
            mode="enforce",
            lookahead_tokens=256,
            global_mode="enforce",
            resident_limit=2,
            vmm_backing=True,
            global_broker_path="/run/ekv/gpu.sqlite",
        )


if __name__ == "__main__":
    unittest.main()
