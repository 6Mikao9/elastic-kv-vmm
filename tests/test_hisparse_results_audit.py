import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hisparse_results import audit, audit_nvml, main, read_json


NATIVE = Path(__file__).resolve().parents[1] / "results/final-native-greedy"
ARTIFACTS = (
    "results.json", "source_manifest.json", "performance.json",
    "semantic_smoke.json", "performance.correctness.json",
)


class HiSparseResultsAuditTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.arm = Path(temp.name)
        for name in ARTIFACTS:
            shutil.copyfile(NATIVE / name, self.arm / name)

    def edit(self, name, change):
        path = self.arm / name
        data = read_json(path)
        change(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_historical_native_artifacts(self):
        _, _, _, report = audit(self.arm, 3)
        self.assertEqual(report["result_artifacts"], "verified")
        self.assertEqual(report["gpu_runtime_and_resource_evidence"], "not_verified")
        with self.assertRaisesRegex(ValueError, "2 measured repeats"):
            audit(self.arm, 2)

    def test_duplicate_semantic_case(self):
        self.edit("semantic_smoke.json", lambda data: data["checks"][1].update(case="arithmetic"))
        with self.assertRaisesRegex(ValueError, "semantic smoke"):
            audit(self.arm, 3)

    def test_corrupt_correctness_token_digest(self):
        self.edit("performance.correctness.json",
                  lambda data: data["probes"][0]["output_ids"].__setitem__(0, 0))
        with self.assertRaisesRegex(ValueError, "correctness probe 0"):
            audit(self.arm, 3)

    def test_inconsistent_request_tpot(self):
        self.edit("performance.json",
                  lambda data: data["phases"][0]["requests"][0].update(tpot_s=0))
        with self.assertRaisesRegex(ValueError, "request latency"):
            audit(self.arm, 3)

    def test_changed_warmup_prompt_is_rejected_across_arms(self):
        self.edit("performance.json",
                  lambda data: data["warmups"][0]["requests"][0].update(prompt_sha256="0" * 64))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main([str(NATIVE), str(self.arm)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("warmup prompt identities differ", stderr.getvalue())

    def test_changed_correctness_prompt_is_rejected_across_arms(self):
        self.edit("performance.correctness.json",
                  lambda data: data["probes"][0].update(prompt_sha256="0" * 64))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main([str(NATIVE), str(self.arm)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Correctness probe prompt identities differ", stderr.getvalue())

    def test_warmup_nvml_samples_must_be_ordered(self):
        def change(data):
            samples = data["warmups"][0]["memory"]["samples"]
            samples[0], samples[1] = samples[1], samples[0]

        self.edit("performance.json", change)
        with self.assertRaisesRegex(ValueError, "Invalid NVML sample for warmups"):
            audit(self.arm, 3)

    def test_measured_nvml_stored_peak_must_match_samples(self):
        self.edit("performance.json",
                  lambda data: data["phases"][0]["memory"].update(peak_sum_used_bytes=0))
        with self.assertRaisesRegex(ValueError, "stored peaks differ for phases"):
            audit(self.arm, 3)

    def test_nvml_sampling_gaps_include_both_window_boundaries(self):
        phase = {
            "workload": "long_serial", "repeat": 0, "phase_wall_s": 1.0,
            "memory": {
                "active_window_s": [0.0, 1.0],
                "samples": [
                    {"offset_s": offset, "rank_used_bytes": [1] * 4, "sum_used_bytes": 4}
                    for offset in (0.4, 0.5, 0.7)
                ],
                "active_sample_count": 3, "rank_peak_used_bytes": [1] * 4,
                "peak_sum_used_bytes": 4,
            },
        }
        coverage = audit_nvml(phase, 4, "phases/long_serial/0", 0.05)
        self.assertEqual(coverage["gaps_over_5x_interval"], 2)
        self.assertAlmostEqual(coverage["max_gap_s"], 0.4)
        self.assertAlmostEqual(coverage["gaps_over_5x_interval_sum_s"], 0.7)

    def test_historical_nvml_sampling_quality_reports_each_phase(self):
        _, _, _, report = audit(self.arm, 3)
        quality = report["nvml_sampling_quality"]
        self.assertEqual(quality["sample_interval_s"], 0.05)
        self.assertEqual(len(quality["warmups"]), 5)
        self.assertEqual(len(quality["phases"]), 15)
        for phase in quality["warmups"] + quality["phases"]:
            self.assertGreater(phase["max_gap_s"], 0)
            self.assertLess(phase["max_gap_s"], 0.1)
            self.assertEqual(phase["gaps_over_5x_interval"], 0)
            self.assertEqual(phase["gaps_over_5x_interval_sum_s"], 0)

    def setup_vmm(self, *, pending_reclamation=False):
        snapshots = [
            {"tp_rank": rank, "tp_size": 4, "idle": True, "owned_pages": 0,
             "reclaimable_bytes": 0, "backed_bytes": 2, "peak_backed_bytes": 6,
             "request_residency": {"resident_layer_rows": 1}}
            for rank in range(4)
        ]
        self.edit("source_manifest.json", lambda data: data.update(vmm_backing=True, resident_limit=2))
        self.edit("results.json",
                  lambda data: data.update(vmm_backing=True, resident_limit=2, vmm_idle=snapshots))
        stats = self.arm / "vmm_stats"
        stats.mkdir()
        for rank, snapshot in enumerate(snapshots):
            disk = dict(snapshot)
            if pending_reclamation:
                pending = 720 << 20
                disk.update(reclaimable_bytes=pending, backed_bytes=pending + 2,
                            peak_backed_bytes=pending + 6)
            (stats / f"tp-{rank}.json").write_text(json.dumps(disk), encoding="utf-8")
        return stats

    def test_vmm_requires_all_four_rank_files(self):
        stats = self.setup_vmm()
        _, _, _, report = audit(self.arm, 3)
        self.assertEqual(report["vmm_rank_evidence"], "verified")
        (stats / "tp-3.json").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete VMM rank evidence"):
            audit(self.arm, 3)

    def test_vmm_allows_later_pending_reclamation_after_semantic_probes(self):
        self.setup_vmm(pending_reclamation=True)
        _, _, _, report = audit(self.arm, 3)
        self.assertEqual(report["vmm_rank_evidence"], "verified")

    def test_vmm_still_requires_cleanup_at_benchmark_idle_checkpoint(self):
        self.setup_vmm()
        self.edit("results.json",
                  lambda data: data["vmm_idle"][0].update(reclaimable_bytes=1))
        with self.assertRaisesRegex(ValueError, "Incomplete benchmark idle cleanup"):
            audit(self.arm, 3)
