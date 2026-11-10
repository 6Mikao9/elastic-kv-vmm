import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_hisparse_locality_runs import HIT_SEMANTICS, SCHEMA, compare, digest_ids, main


RID = "private-request-id"
PROMPT = [887766554, 2, 3]
OUTPUT = list(range(55000, 55020))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_run(run, hot, output=OUTPUT, *, prompt=PROMPT, runner="same", bad_trace=None):
    access = run / "access"
    access.mkdir(parents=True)
    write_json(run / "source_manifest.json", {
        "dataset": str(run / "private-dataset"), "model": str(run / "private-model"),
        "dataset_sha256": "dataset", "model_metadata_sha256": {"config": "model"},
        "runner_sha256": {"runner.py": runner},
    })
    write_rows(run / "requests.jsonl", [{
        "request_id": RID, "input_ids": prompt, "input_tokens": len(prompt),
        "input_ids_sha256": digest_ids(prompt), "output_ids": output,
        "output_tokens": len(output), "output_ids_sha256": digest_ids(output),
        "max_output_tokens": 25, "finish_reason": {
            "type": "length" if hot == 4096 else "stop", "matched": "private-stop-string",
        },
    }])
    write_json(run / "results.json", {
        "status": "ok", "diagnostic_capture": True, "requests": [RID],
        "hisparse": {"top_k": 2048, "device_buffer_size": hot},
        "capacity": {"reported_pool_capacity": 12345},
        "capture_verification": {"request_rank_coverage_ok": True, "truncation_markers": {},
                                 "access_rows": 2 * (len(output) - 1)},
    })
    for rank in (0, 1):
        header = {
            "type": "metadata", "schema": SCHEMA, "rank": rank, "layers": [5],
            "diagnostic_only": True, "performance_evidence": False,
            "prestate_hit_semantics": HIT_SEMANTICS,
            "coordinator_sha256": "coordinator", "kernel_sha256": "kernel",
            "wrapper_sha256": "different" if bad_trace == "metadata" else "wrapper",
        }
        rows = []
        for step in range(len(output) - 1):
            selected = [0, 1]
            if hot == 8192:
                selected = [1, 2] if rank == 0 and step == 17 else [1, 0]
            rows.append({
                "type": "access", "rid": RID, "rank": rank, "layer": 5, "step": step,
                "seq_len": len(prompt) + step + (0 if bad_trace == "sequence" else 1),
                "hot_capacity": hot + (1 if bad_trace == "capacity" else 0),
                "selected_tokens": selected, "prestate_hit_tokens": selected if hot == 8192 else [0],
            })
        write_rows(access / f"hisparse-rank{rank}.jsonl", [
            header, *rows, {"type": "summary", "rows": len(rows), "dropped": {}},
        ])
    return run


class CompareHiSparseLocalityRunsTest(unittest.TestCase):
    def test_pairs_only_observed_shared_history_and_separates_selection_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = make_run(root / "left", 4096)
            right = make_run(root / "right", 8192, OUTPUT[:18] + [91, 92, 93])
            third = make_run(root / "third", 2048)
            summary = compare([left, right, third])

        row, = summary["requests"]
        self.assertEqual(row["request"], "R01")
        self.assertEqual([(run["native_hot_capacity"], run["output_tokens"], run["finish_reason"])
                          for run in row["runs"]],
                         [(4096, 20, "length"), (8192, 21, "stop"), (2048, 20, "stop")])
        self.assertEqual(row["runs"][0]["native_after_first_16"]["0"], {
            "rows": 3, "selected_tokens": 6, "native_prestate_misses": 3,
            "native_prestate_miss_rate": 0.5,
        })
        pair, identical, _ = row["pairs"]
        self.assertFalse(pair["output_ids_exact"])
        self.assertEqual(pair["output_common_prefix_tokens"], 18)
        self.assertEqual(pair["paired_all"]["0"]["same_history_rows"], 18)
        self.assertEqual(pair["paired_after_first_16"]["0"], {
            "same_history_rows": 2, "same_selection_rows": 1, "selected_tokens": 2,
            "left_native_prestate_misses": 1, "right_native_prestate_misses": 0,
        })
        self.assertEqual(pair["paired_after_first_16"]["1"]["same_selection_rows"], 2)
        self.assertTrue(identical["output_ids_exact"])
        self.assertEqual(identical["output_common_prefix_tokens"], 20)
        self.assertEqual(identical["paired_all"]["0"]["same_history_rows"], 19)
        serialized = json.dumps(summary)
        for private in (RID, "private-stop-string", str(PROMPT[0]), str(OUTPUT[0]), digest_ids(PROMPT)):
            self.assertNotIn(private, serialized)

    def test_rejects_mismatched_prompts_manifest_capacity_and_decode_history(self):
        cases = [
            ({"prompt": [6, 2, 3]}, "different prompts"),
            ({"runner": "changed"}, "metadata differs"),
            ({"bad_trace": "metadata"}, "metadata differs"),
            ({"bad_trace": "capacity"}, "disagrees with run configuration"),
            ({"bad_trace": "sequence"}, "disagrees with decode input history"),
        ]
        for options, error in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                left = make_run(root / "left", 4096)
                right = make_run(root / "right", 8192, **options)
                with self.assertRaisesRegex(ValueError, error):
                    compare([left, right])

    def test_cli_refuses_invalid_digest_without_printing_private_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = make_run(root / "left", 4096)
            right = make_run(root / "right", 8192)
            path = right / "requests.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["input_ids_sha256"] = "invalid"
            write_rows(path, [row])
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    main([str(left), str(right)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn(RID, stderr.getvalue())
        self.assertNotIn(str(PROMPT[0]), stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
