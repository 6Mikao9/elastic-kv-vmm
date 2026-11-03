"""Repeat identical greedy probes and record logprobs at the first divergence."""

import argparse
import json
from pathlib import Path
import urllib.request

from benchmarks.client import digest_ids, prompt_ids, write_json


def capture(opener, url, prompt, max_tokens):
    body = {
        "input_ids": prompt, "stream": False, "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 5, "return_text_in_logprobs": False,
        "sampling_params": {"temperature": 0, "ignore_eos": True, "max_new_tokens": max_tokens},
    }
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with opener.open(request, timeout=600) as response:
        result = json.load(response)
    meta = result["meta_info"]
    selected = meta["output_token_logprobs"]
    ids = result.get("output_ids", [row[1] for row in selected])
    top = meta["output_top_logprobs"]
    if (len(ids) != max_tokens or meta["completion_tokens"] != max_tokens or
            meta["prompt_tokens"] != len(prompt) or len(top) != max_tokens or
            ids != [row[1] for row in selected]):
        raise ValueError("incomplete or inconsistent token probe")
    return {
        "output_ids": ids, "output_ids_sha256": digest_ids(ids),
        "selected_logprobs": [[row[1], row[0]] for row in selected],
        "top_logprobs": [[[row[1], row[0]] for row in group] for group in top],
    }


def first_divergence(left, right):
    for position, (a, b) in enumerate(zip(left["output_ids"], right["output_ids"], strict=True)):
        if a != b:
            comparison = []
            for run in (left, right):
                scores = dict(run["top_logprobs"][position])
                margin = scores[a] - scores[b] if a in scores and b in scores else None
                comparison.append({"top5": run["top_logprobs"][position],
                                   "left_minus_right_logprob": margin})
            return {"zero_based_position": position, "left_id": a, "right_id": b,
                    "captures": comparison}
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31080/generate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--captures", type=int, default=4)
    parser.add_argument("--reference", type=Path, help="compare first captures with a prior arm")
    args = parser.parse_args(argv)
    if args.output.exists() or args.captures < 2:
        parser.error("require a fresh output path and at least two captures")
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    if reference is not None and reference["status"] != "complete":
        parser.error("reference must be complete")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {"status": "running", "probes": []}
    try:
        for length in (128, 8192):
            prompt = prompt_ids(f"hisparse-smoke/{length}", length)
            probe = {"input_tokens": length, "prompt_sha256": digest_ids(prompt),
                     "captures": [], "vs_first": []}
            report["probes"].append(probe)
            for index in range(args.captures):
                current = capture(opener, args.url, prompt, 16)
                if index:
                    probe["vs_first"].append(first_divergence(probe["captures"][0], current))
                probe["captures"].append(current)
            if reference is not None:
                previous = next(item for item in reference["probes"]
                                if item["prompt_sha256"] == probe["prompt_sha256"])
                probe["vs_reference_first"] = first_divergence(previous["captures"][0],
                                                               probe["captures"][0])
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
