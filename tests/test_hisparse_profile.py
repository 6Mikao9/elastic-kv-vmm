import gzip
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.profile_hisparse_live import capture, inspect_traces
from scripts.summarize_hisparse import comparison_signature


def write_trace(directory, rank, *, include_gpu=True, multiple_devices=False):
    def event(category, start, duration, name=None, **args):
        return {"ph": "X", "cat": category, "ts": start, "dur": duration,
                "name": name or category, "args": args}

    events = [
        event("cpu_op", 0, 20),
        event("cuda_runtime", 9, 2, "cudaStreamSynchronize"),
        event("cuda_runtime", 12, 1, "cudaStreamWaitEvent"),
    ]
    if include_gpu:
        events.extend([
            event("kernel", 2, 8, device=rank),
            event("kernel", 5, 10, device=rank + multiple_devices),
            event("gpu_memcpy", 14, 2, "Memcpy HtoD", bytes=128),
            event("gpu_memcpy", 16, 1, "Memcpy HtoD"),
        ])
    with gzip.open(directory / f"test-TP-{rank}.trace.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"traceEvents": events}, stream)


class HiSparseProfileTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)

    def test_trace_union_and_incomplete_traces(self):
        for rank in range(3):
            write_trace(self.output, rank)
        with self.assertRaisesRegex(ValueError, "Expected TP ranks"):
            inspect_traces(self.output)
        write_trace(self.output, 3, include_gpu=False)
        with self.assertRaisesRegex(ValueError, "missing.*CUDA kernels"):
            inspect_traces(self.output)
        write_trace(self.output, 3)
        ranks = inspect_traces(self.output)
        self.assertEqual(set(ranks), {"0", "1", "2", "3"})
        self.assertEqual(ranks["0"]["gpu_activity_union_us"], 15)
        self.assertEqual(ranks["0"]["gpu_activity_sum_us"], 21)
        self.assertEqual(ranks["0"]["cuda_device_from_trace"], 0)
        self.assertEqual(ranks["0"]["cuda_kernels_by_name"], {
            "kernel": {"count": 2, "sum_us": 18, "max_us": 10},
        })
        self.assertEqual(ranks["0"]["cuda_wait_api_cpu_durations_by_name"], {
            "cudaStreamSynchronize": {"count": 1, "sum_us": 2, "max_us": 2},
            "cudaStreamWaitEvent": {"count": 1, "sum_us": 1, "max_us": 1},
        })
        self.assertEqual(ranks["0"]["explicit_gpu_memcpy"]["Memcpy HtoD"], {
            "events": 2, "events_with_bytes": 1, "reported_bytes": 128,
        })
        write_trace(self.output, 3, multiple_devices=True)
        with self.assertRaisesRegex(ValueError, "multiple devices"):
            inspect_traces(self.output)

    def run_capture(self, *, fail_phase=False, fail_stop=False):
        calls = []
        trace_dir = self.output / "profile_traces"
        payloads = []
        phase_specs = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                calls.append(self.path)
                if self.path == "/start_profile":
                    payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                elif not fail_stop:
                    for rank in range(4):
                        write_trace(trace_dir, rank)
                self.send_response(500 if fail_stop and self.path == "/stop_profile" else 200)
                self.end_headers()

            def log_message(self, *_):
                pass

        def fake_run_requests(url, specs, concurrency, *, timeout, stream):
            calls.append("/generate")
            phase_specs.append(specs)
            if fail_phase and "/start_profile" in calls:
                raise RuntimeError("generation failed")
            return [dict(spec, output_tokens=spec["requested_output_tokens"],
                         prompt_sha256="prompt") for spec in specs], 0, 1

        client = SimpleNamespace(run_requests=fake_run_requests,
                                 write_json=lambda path, value: path.write_text(json.dumps(value)))
        workloads = SimpleNamespace(
            WORKLOADS={"mixed": (4, [(128, 8), (8192, 128)] * 2)},
            spec=lambda key, index, inputs, outputs: {
                "request_id": f"hisparse/{key}/{index}", "input_tokens": inputs,
                "requested_output_tokens": outputs,
            },
            verify=lambda record: record,
        )
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with patch.dict(sys.modules, {"benchmarks.client": client,
                                              "benchmarks.hisparse_client": workloads}):
                    if fail_phase:
                        with self.assertLogs("scripts.profile_hisparse_live", "WARNING") as logs:
                            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                                capture(base_url, self.output, "mixed", opener,
                                        output_token_cap=12)
                        self.assertIn("cleanup also failed", logs.output[0])
                    else:
                        report = capture(base_url, self.output, "mixed", opener,
                                         output_token_cap=12)
                        self.assertEqual(len(report["ranks"]), 4)
                        self.assertEqual(len(report["requests"]), 4)
                        self.assertEqual(report["output_token_cap"], 12)
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(calls, ["/generate", "/start_profile", "/generate", "/stop_profile"])
        self.assertEqual(phase_specs[0], phase_specs[1])
        self.assertEqual([spec["requested_output_tokens"] for spec in phase_specs[0]],
                         [8, 12, 8, 12])
        self.assertEqual(payloads, [{"output_dir": str(trace_dir.resolve()),
                                     "activities": ["CPU", "GPU"], "merge_profiles": False,
                                     "with_stack": False, "record_shapes": False}])
        self.assertEqual((self.output / "profile_summary.json").exists(), not fail_phase)

    def test_capture(self):
        self.run_capture()

    def test_failed_generation_stops_profile_and_preserves_original_error(self):
        self.run_capture(fail_phase=True, fail_stop=True)

    def test_summary_rejects_profile_diagnostic(self):
        (self.output / "results.json").write_text('{"status":"complete","mode":"native"}')
        (self.output / "source_manifest.json").write_text(
            '{"mode":"native","live_profile_diagnostic":"mixed"}'
        )
        with self.assertRaisesRegex(ValueError, "live profiling diagnostic"):
            comparison_signature(self.output, {})
