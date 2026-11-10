import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.summarize_hisparse_locality import main, summarize, supplementary_thresholds


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class LocalityReportTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run = self.root / "private"
        self.access = self.run / "access"
        self.access.mkdir(parents=True)
        write_rows(self.run / "requests.jsonl", [
            dict(request_id="PRIVATE-RID", origin_id="PRIVATE-ORIGIN", workload="chatbot",
                 input_tokens=8000, output_tokens=35, input_ids=[79194], output_ids=[123456]),
        ])
        (self.run / "results.json").write_text(json.dumps(dict(
            status="ok", hisparse={"top_k": 2048, "device_buffer_size": 4096},
            capture_verification={"request_rank_coverage_ok": True},
        )), encoding="utf-8")

    def capture(self, rank, count=34, truncated=False):
        rows = [dict(type="metadata", rank=rank, schema="minimax-hisparse-access-v1", layers=[5])]
        for step in range(count):
            token = 1 if rank == 1 and step == 3 else 0
            rows.append(dict(type="access", rank=rank, rid="PRIVATE-RID", generation=1,
                             step=step, layer=5, hot_capacity=4096, seq_len=8192,
                             selected_blocks=[0, -1], block_size=128,
                             selected_tokens=[token, token], prestate_hit_tokens=[token] if rank else []))
        if truncated:
            rows.append(dict(type="truncation", reason="max_steps"))
        write_rows(self.access / f"hisparse-rank{rank}.jsonl", rows)

    def test_rank_dedup_warmup_and_private_values(self):
        self.capture(0)
        self.capture(1)
        output = self.root / "public"
        main([str(self.run), str(output)])
        payload = (output / "summary.json").read_text(encoding="utf-8")
        result = json.loads(payload)
        self.assertEqual(result["coverage"], "complete")
        self.assertNotIn("PRIVATE", payload)
        self.assertNotIn("79194", payload)
        self.assertNotIn("123456", payload)
        self.assertEqual(result["capture"]["rank_selection_comparison"], [
            dict(rank=1, matched_steps=34, selection_mismatch_steps=1,
                 primary_only_steps=0, other_only_steps=0),
        ])
        totals = {row["period"]: row for row in result["aggregates"] if row["workload"] == "all"}
        self.assertEqual(totals["all"]["selected"], 34)
        self.assertEqual(totals["all"]["simulated_misses_4096"], 1)
        self.assertEqual(totals["after_first_16"]["selected"], 18)
        self.assertEqual(totals["after_first_16"]["simulated_misses_4096"], 0)
        self.assertEqual(totals["after_first_16"]["native_prestate_miss_rate"], 1)
        windows = [row for row in result["windows"] if row["rank"] == 0 and row["window_steps"] == 32]
        self.assertEqual([row["steps"] for row in windows], [32, 2])
        self.assertTrue((output / "windows.csv").is_file())
        self.assertTrue((output / "report.md").is_file())

    def test_partial_capture_reports_truncation_and_gaps(self):
        self.capture(0)
        self.capture(1, count=3, truncated=True)
        result = summarize(self.run)
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual(result["capture"]["truncation_markers"], {"max_steps": 1})
        self.assertEqual(result["capture"]["rank_layer_coverage_issues"][0]["missing_steps"], 31)

    def test_request_average_preserves_worst_shrink_window(self):
        write_rows(self.run / "requests.jsonl", [
            dict(request_id="PRIVATE-RID", origin_id="PRIVATE-ORIGIN", workload="chatbot",
                 input_tokens=8000, output_tokens=49),
        ])
        for rank in (0, 1):
            rows = [dict(type="metadata", rank=rank, schema="minimax-hisparse-access-v1", layers=[5])]
            for step in range(48):
                start = 2048 * (step % 2 if step < 16 or step >= 32 else 1)
                selected = list(range(start, start + 2048))
                rows.append(dict(type="access", rank=rank, rid="PRIVATE-RID", generation=1,
                                 step=step, layer=5, hot_capacity=4096, seq_len=8192,
                                 selected_blocks=list(range(start // 128, start // 128 + 16)),
                                 block_size=128,
                                 selected_tokens=selected,
                                 prestate_hit_tokens=selected if step < 32 else []))
            write_rows(self.access / f"hisparse-rank{rank}.jsonl", rows)
        result = summarize(self.run)
        request, = result["primary_request_after_first_16"]
        self.assertEqual(request["steps"], 32)
        self.assertEqual(request["simulated_shrink_cost_pp_4096_to_2048"], 50)
        self.assertEqual(request["simulated_expansion_benefit_pp_4096_to_8192"], 0)
        self.assertEqual(request["offline_hindsight_shrink_cost_within_pp"],
                         {"1": False, "2": False, "5": False})
        burst, = result["primary_request_layer_complete_16_step_bursts"]
        self.assertEqual(burst["native_prestate_miss_rate_range_pp"], 100)
        self.assertEqual(burst["simulated_shrink_cost_max_pp"], 100)
        self.assertEqual(burst["simulated_shrink_cost_max_window_start_step"], 32)

    def test_oversized_selection_preserves_native_and_other_capacities(self):
        self.capture(0)
        self.capture(1)
        path = self.access / "hisparse-rank0.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[17].update(selected_tokens=list(range(2176)), selected_blocks=list(range(17)))
        write_rows(path, rows)
        output = self.root / "public"
        main([str(self.run), str(output)])
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        limits = result["selection_limits"]
        self.assertEqual(limits["max_unique_selected_tokens"], 2176)
        self.assertEqual(limits["max_block_slots"], 17)
        self.assertEqual(limits["max_valid_block_slots"], 17)
        self.assertEqual(limits["capacities"]["2048"], {
            "status": "infeasible", "oversized_access_rows_all_ranks": 1, "supplementary": False,
        })
        self.assertEqual(limits["capacities"]["2176"]["status"], "feasible")
        request, = result["primary_request_after_first_16"]
        self.assertEqual(request["selected"], 2176 + 17)
        self.assertEqual(request["native_prestate_miss_rate"], 1)
        self.assertIsNone(request["simulated_miss_rate_2048"])
        self.assertIsNone(request["simulated_shrink_cost_pp_4096_to_2048"])
        self.assertIsNotNone(request["simulated_miss_rate_4096"])
        self.assertIsNotNone(request["simulated_miss_rate_2176"])
        self.assertIsNotNone(request["simulated_expansion_benefit_pp_4096_to_8192"])
        burst, = result["primary_request_layer_complete_16_step_bursts"]
        self.assertIsNone(burst["simulated_shrink_cost_max_pp"])

    def test_supplementary_cli_keeps_rank_workload_and_request_layer_separate(self):
        self.capture(0)
        self.capture(1)
        output = self.root / "public"
        main([str(self.run), str(output), "--supplementary-capacities",
              "2560", "3072", "3584", "6144"])
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(set(result["selection_limits"]["capacities"]),
                         {"2048", "2176", "2560", "3072", "3584", "4096", "6144", "8192"})
        chatbot = [row for row in result["rank_workload_after_first_16"]
                   if row["workload"] == "chatbot"]
        self.assertEqual([(row["rank"], row["steps"]) for row in chatbot], [(0, 18), (1, 18)])
        self.assertEqual([row["native_prestate_miss_rate"] for row in chatbot], [1, 0])
        self.assertEqual([row["simulated_miss_rate_3072"] for row in chatbot], [0, 0])
        by_layer = result["rank_request_layer_after_first_16"]
        self.assertEqual([(row["rank"], row["request"], row["layer"]) for row in by_layer],
                         [(0, "R01", 5), (1, "R01", 5)])
        self.assertEqual(by_layer[1]["offline_hindsight_min_supplementary_capacity_by_pp"],
                         {"1": 2176, "2": 2176, "5": 2176})

    def test_hindsight_threshold_uses_smallest_feasible_tested_shrink_capacity(self):
        counts = dict(selected=1000, simulated_misses_4096=10, simulated_misses_2560=60,
                      simulated_misses_3072=30, simulated_misses_3584=20,
                      simulated_misses_6144=0)
        self.assertEqual(supplementary_thresholds(counts, (6144, 3584, 2176, 2560, 3072)),
                         {"1": 3584, "2": 3072, "5": 2560})
        self.assertEqual(supplementary_thresholds(counts, (2176, 6144)),
                         {"1": None, "2": None, "5": None})
