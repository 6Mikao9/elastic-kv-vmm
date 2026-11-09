import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks import hisparse_client


UUIDS = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]


class RankSamplerTimingTest(unittest.TestCase):
    def test_sweep_records_lock_wait_and_slow_rank(self):
        nvml = Mock()
        nvml.nvmlDeviceGetHandleByUUID.side_effect = lambda uuid: uuid
        nvml.nvmlDeviceGetMemoryInfo.side_effect = [
            SimpleNamespace(used=value) for value in (10, 20, 30, 40)
        ]
        times = [
            1_000_000_000, 1_000_000_010,  # UTC anchor bracket
            1_100_000_000, 1_200_000_000, 1_200_000_010,  # request, lock, start
            1_200_000_020, 1_200_000_030,
            1_200_000_040, 6_200_000_040,  # slow rank
            6_200_000_050, 6_200_000_060,
            6_200_000_070, 6_200_000_080,
            6_200_000_100,  # sweep end
        ]
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.multiple(
            hisparse_client.time,
            perf_counter=Mock(side_effect=[1.0, 6.20000009]),
            perf_counter_ns=Mock(side_effect=times),
            time_ns=Mock(return_value=1_700_000_000_000_000_000),
        ):
            sampler = hisparse_client.RankSampler(UUIDS)
            sampler.take()

        self.assertEqual(sampler.clock_anchor, {
            "origin_perf_counter_s": 1.0,
            "utc_unix_ns": 1_700_000_000_000_000_000,
            "perf_counter_before_ns": times[0],
            "perf_counter_after_ns": times[1],
        })
        self.assertEqual([call.args[0] for call in nvml.nvmlDeviceGetMemoryInfo.call_args_list], UUIDS)
        sample, = sampler.samples
        self.assertAlmostEqual(sample["offset_s"], 5.20000009)
        self.assertEqual(sample["rank_used_bytes"], [10, 20, 30, 40])
        self.assertEqual(sample["sum_used_bytes"], 100)
        self.assertEqual(sample["timing"], {
            "requested_perf_counter_ns": times[2],
            "lock_acquired_perf_counter_ns": times[3],
            "sample_window_perf_counter_ns": [times[4], times[-1]],
            "rank_query_windows_perf_counter_ns": [
                times[index:index + 2] for index in range(5, 13, 2)
            ],
        })

    def test_failed_rank_does_not_publish_partial_sweep(self):
        nvml = Mock()
        nvml.nvmlDeviceGetMemoryInfo.side_effect = [
            SimpleNamespace(used=10), RuntimeError("NVML query failed")
        ]
        with patch.dict(sys.modules, {"pynvml": nvml}):
            sampler = hisparse_client.RankSampler(UUIDS)
        with self.assertRaisesRegex(RuntimeError, "NVML query failed"):
            sampler.take()
        self.assertEqual(sampler.samples, [])
        self.assertEqual(nvml.nvmlDeviceGetMemoryInfo.call_count, 2)

    def test_phase_keeps_completion_based_window_and_adds_rank_order(self):
        def sample(offset, used):
            return {"offset_s": offset, "rank_used_bytes": [used] * 4,
                    "sum_used_bytes": used * 4}

        samples = [sample(0.9, 99), sample(1.0, 2), sample(2.0, 3), sample(2.1, 99)]

        class PhaseSampler:
            origin = 10.0
            clock_anchor = {"origin_perf_counter_s": 10.0}

            def __init__(self):
                self.samples = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def take(self):
                self.samples.append(samples[0] if not self.samples else samples[-1])

        sampler = PhaseSampler()

        def run_requests(url, specs, concurrency, timeout, stream):
            sampler.samples.extend(samples[1:3])
            records = [dict(spec, _started_perf_s=11.0,
                            output_tokens=spec["requested_output_tokens"],
                            ttft_s=0.1, tpot_s=0.01, end_to_end_s=0.73) for spec in specs]
            return records, 11.0, 12.0

        with patch.object(hisparse_client, "RankSampler", return_value=sampler), patch.object(
            hisparse_client, "run_requests", side_effect=run_requests
        ), patch.object(hisparse_client.time, "sleep"):
            phase = hisparse_client.run_phase("http://unused", "short_serial", 0, UUIDS, 1)
        memory = phase["memory"]
        self.assertEqual(phase["phase_wall_s"], 1.0)
        self.assertEqual(memory["active_window_s"], [1.0, 2.0])
        self.assertEqual(memory["active_sample_count"], 2)
        self.assertEqual(memory["peak_sum_used_bytes"], 12)
        self.assertEqual(memory["rank_gpu_uuids"], UUIDS)
        self.assertEqual(memory["clock_anchor"], sampler.clock_anchor)
