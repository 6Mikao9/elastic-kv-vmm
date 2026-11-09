import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_hisparse_timeline import analyze_path


def event(category, start, duration, name="event", **args):
    return {"ph": "X", "cat": category, "ts": start, "dur": duration,
            "name": name, "pid": 42, "args": args}


class HiSparseTimelineTest(unittest.TestCase):
    def analyze(self, events, *, directory=False, expected_tp_size=None):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "test-TP-0.trace.json.gz"
            with gzip.open(trace, "wt", encoding="utf-8") as output:
                json.dump({"traceEvents": events}, output)
            return analyze_path(trace.parent if directory else trace,
                                expected_tp_size=expected_tp_size)["ranks"]["0"]

    def test_gpu_envelope_gaps_and_cpu_overlap(self):
        timeline = self.analyze([
            event("kernel", 0, 0, device=3, stream=99),
            event("kernel", 10, 8, device=0, stream=1),
            event("gpu_memcpy", 15, 5, device=0, stream=2),
            event("kernel", 24, 6, device=0, stream=1),
            event("gpu_memset", 31, 2, device=0),
            event("kernel", 100, 0, device=3, stream=99),
            event("cpu_op", 8, 9),
            event("cpu_op", 11, 4),
            event("cpu_op", 16, 10),
            event("cuda_runtime", 24, 2, "cudaStreamSynchronize"),
            event("cuda_driver", 25, 1, "cuStreamWaitEvent"),
        ], directory=True)
        self.assertEqual(timeline["gpu_envelope_us"], 23)
        self.assertEqual(timeline["cuda_device_from_trace"], 0)
        self.assertEqual(timeline["gpu_events_without_device_metadata"], 0)
        self.assertEqual(timeline["gpu_activity_union_us"], 18)
        self.assertEqual(timeline["interior_gpu_idle_us"], 5)
        self.assertEqual(timeline["interior_gpu_gap_count"], 2)
        self.assertEqual(timeline["max_interior_gpu_gap_us"], 4)
        self.assertEqual(timeline["largest_interior_gpu_gaps"], [
            {"start_from_first_gpu_us": 10, "duration_us": 4},
            {"start_from_first_gpu_us": 20, "duration_us": 1},
        ])
        self.assertEqual(timeline["gpu_activity_union_by_category_us"],
                         {"kernel": 14, "gpu_memcpy": 5, "gpu_memset": 2})
        self.assertEqual(timeline["gpu_activity_union_by_stream_us"],
                         {"1": 14, "2": 5, "unknown": 2})
        self.assertEqual(timeline["recorded_cpu_op_overlap_with_gpu_us"], 12)
        self.assertEqual(timeline["blocking_sync_cpu_calls_by_name"], {
            "cudaStreamSynchronize": {"count": 1, "sum_us": 2, "max_us": 2},
        })
        self.assertEqual(timeline["stream_wait_event_cpu_enqueue_calls_by_name"], {
            "cuStreamWaitEvent": {"count": 1, "sum_us": 1, "max_us": 1},
        })
        self.assertEqual(timeline["zero_duration_events_by_category"], {"kernel": 2})

    def test_explicit_launch_correlation_and_ambiguity(self):
        timeline = self.analyze([
            event("cuda_runtime", 0, 5, "cudaLaunchKernel", correlation=1),
            event("kernel", 4, 1, correlation=1),
            event("cuda_runtime", 5, 1, "cudaLaunchKernel", correlation=2),
            event("cuda_driver", 5, 1, "cuLaunchKernel", correlation=2),
            event("kernel", 6, 1, correlation=2),
            event("kernel", 7, 1, correlation=3),
            event("kernel", 8, 1),
            event("cuda_driver", 9, 1, "cuLaunchKernel", correlation=4),
            event("kernel", 11, 1, correlation=4),
            event("kernel", 13, 1, correlation=4),
            event("cuda_driver", 12, 1, "cuLaunchKernel", correlation=5),
            event("kernel", 16, 1, correlation=5),
        ])
        self.assertEqual(timeline["correlated_kernel_start_minus_launch_end"], {
            "matched_kernel_events": 2, "unmatched_kernel_events_with_id": 1,
            "ambiguous_kernel_events": 3, "kernel_events_without_id": 1,
            "negative_delta_events": 1, "min_us": -1, "max_us": 3, "mean_us": 1,
        })
        self.assertIsNone(timeline["cuda_device_from_trace"])
        self.assertEqual(timeline["gpu_events_without_device_metadata"], 7)

    def test_multiple_launch_processes_skip_correlation(self):
        timeline = self.analyze([
            event("cuda_runtime", 0, 1, "cudaLaunchKernel", correlation=1),
            dict(event("cuda_runtime", 0, 1, "cudaLaunchKernel", correlation=2), pid=43),
            event("kernel", 1, 1, correlation=1),
        ])
        self.assertEqual(timeline["correlated_kernel_start_minus_launch_end"], {
            "status": "skipped_multiple_launch_processes", "kernel_events_without_id": 0,
        })

    def test_invalid_events(self):
        with self.assertRaisesRegex(ValueError, "invalid trace event list"):
            self.analyze({"unexpected": "object"})
        with self.assertRaisesRegex(ValueError, "invalid kernel interval end"):
            self.analyze([event("kernel", 1e308, 1e308)])
        with self.assertRaisesRegex(ValueError, "invalid kernel interval"):
            self.analyze([event("kernel", 10, -1)])
        with self.assertRaisesRegex(ValueError, "GPU intervals span multiple devices"):
            self.analyze([event("kernel", 10, 1, device=0),
                          event("gpu_memcpy", 11, 1, device=1)])

    def test_expected_tp_size_requires_exact_ranks(self):
        events = [event("kernel", 10, 1)]
        self.analyze(events, directory=True, expected_tp_size=1)
        with self.assertRaisesRegex(ValueError, "Expected TP ranks 0 through 3, found 0"):
            self.analyze(events, directory=True, expected_tp_size=4)
        with self.assertRaisesRegex(ValueError, "expected TP size must be positive"):
            self.analyze(events, expected_tp_size=0)


if __name__ == "__main__":
    unittest.main()
