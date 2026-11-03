import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.summarize_hisparse import load, require_matched


EVIDENCE = Path(__file__).resolve().parents[1] / "results" / "final-native-greedy"


class MeasuredRunValidityTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.documents = {
            name: json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
            for name in ("results.json", "performance.json", "source_manifest.json")
        }

    def load_fixture(self):
        for name, value in self.documents.items():
            (self.directory / name).write_text(json.dumps(value), encoding="utf-8")
        return load(self.directory)

    def test_failed_outer_run_rejects_completed_performance(self):
        self.documents["results.json"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "outer run is incomplete"):
            self.load_fixture()

    def test_wrong_actual_capacity_rejected(self):
        self.documents["results.json"]["capacity"]["derived_logical_capacity"] -= 128
        with self.assertRaisesRegex(ValueError, "actual logical capacity"):
            self.load_fixture()

    def test_diagnostic_timing_rejected(self):
        self.documents["source_manifest.json"]["logical_shadow_diagnostic"] = True
        with self.assertRaisesRegex(ValueError, "logical_shadow_diagnostic"):
            self.load_fixture()

    def test_inference_mismatch_rejected_but_physical_capacity_can_differ(self):
        reference = self.load_fixture()
        capacity = self.documents["results.json"]["capacity"]
        capacity.update(reported_pool_capacity=65536, resolved_max_total_tokens=65536,
                        host_to_device_ratio=1)
        manifest = self.documents["source_manifest.json"]
        argv = manifest["server_args"]
        argv[argv.index("--max-total-tokens") + 1] = "65536"
        manifest["requested_pool_capacity"] = 65536
        manifest["mode"] = self.documents["results.json"]["mode"] = "resident"
        argv.remove("--enable-hisparse")
        index = argv.index("--hisparse-config")
        del argv[index:index + 2]
        require_matched(reference, self.load_fixture())
        argv[argv.index("--random-seed") + 1] = "31"
        with self.assertRaisesRegex(ValueError, "common inference configuration differs"):
            require_matched(reference, self.load_fixture())

    def test_hisparse_ratio_disagreement_rejected(self):
        argv = self.documents["source_manifest.json"]["server_args"]
        index = argv.index("--hisparse-config") + 1
        config = json.loads(argv[index])
        config["host_to_device_ratio"] = 1
        argv[index] = json.dumps(config)
        with self.assertRaisesRegex(ValueError, "HiSparse config ratio"):
            self.load_fixture()

    def test_different_hisparse_hot_buffer_rejected_by_comparator(self):
        reference = self.load_fixture()
        argv = self.documents["source_manifest.json"]["server_args"]
        index = argv.index("--hisparse-config") + 1
        config = json.loads(argv[index])
        config["device_buffer_size"] *= 2
        argv[index] = json.dumps(config)
        with self.assertRaisesRegex(ValueError, "HiSparse configuration differs"):
            require_matched(reference, self.load_fixture())