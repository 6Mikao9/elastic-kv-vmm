"""Benchmark native token-ID completions from the vLLM OpenAI endpoint."""

import json
import os
import time

import requests

from benchmarks import client as common


def generate(url, spec, timeout, stream=True):
    prompt = common.prompt_ids(spec["request_id"], spec["input_tokens"])
    payload = {
        "model": "bench", "prompt": prompt, "temperature": 0,
        "ignore_eos": True, "max_tokens": spec["requested_output_tokens"],
        "return_token_ids": True, "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    started = time.perf_counter()
    ids, events = [], []
    first = count = None
    with requests.Session() as session:
        session.trust_env = False
        with session.post(url, json=payload, timeout=timeout, stream=stream) as response:
            response.raise_for_status()
            if stream:
                done = False
                for raw in common.sse_data(response):
                    if raw == "[DONE]":
                        done = True
                        break
                    data = json.loads(raw)
                    if data.get("error"):
                        raise common.BenchmarkError("vLLM stream returned an error")
                    for choice in data.get("choices", []):
                        delta = choice.get("token_ids") or []
                        if delta:
                            now = time.perf_counter()
                            if first is None:
                                first = now
                            ids.extend(delta)
                            events.append({"arrival_s": now - started,
                                           "completion_tokens": len(ids)})
                    if data.get("usage") is not None:
                        count = data["usage"]["completion_tokens"]
                if not done or first is None:
                    raise common.BenchmarkError("incomplete vLLM token stream")
            else:
                data = response.json()
                ids = data["choices"][0]["token_ids"]
                count = data["usage"]["completion_tokens"]
    ended = time.perf_counter()
    if (type(count) is not int or count != spec["requested_output_tokens"]
            or len(ids) != count or any(type(token) is not int or token < 0 for token in ids)):
        raise common.BenchmarkError("vLLM token IDs, usage and requested count do not agree")
    elapsed = ended - started
    return {
        **spec, "_started_perf_s": started, "output_tokens": count,
        "prompt_sha256": common.digest_ids(prompt),
        "output_ids_sha256": common.digest_ids(ids), "output_ids_complete": True,
        "output_ids": ids if not stream else None,
        "ttft_s": first - started if stream else None,
        "end_to_end_s": elapsed,
        "tpot_s": (ended - first) / (count - 1) if stream and count > 1 else None,
        "request_output_tokens_per_s": count / elapsed,
        "token_count_events": events,
        "first_event_tokens": events[0]["completion_tokens"] if events else None,
    }


class MemorySampler(common.MemorySampler):
    def __init__(self):
        super().__init__()
        self.stats_path = os.environ.get("EKV_VLLM_STATS_PATH")

    def take(self):
        with self.lock:
            stats = None
            if self.stats_path:
                with open(self.stats_path, encoding="utf-8") as source:
                    raw = json.load(source)
                stats = {key: raw[key] for key in (
                    "backed_bytes", "peak_backed_bytes", "logical_capacity_bytes",
                    "live_requests", "num_blocks", "ensure_calls", "reclaim_calls",
                )}
                if any(type(value) is not int or value < 0 for value in stats.values()):
                    raise common.BenchmarkError("invalid vLLM adapter statistics")
            self.samples.append({
                "offset_s": time.perf_counter() - self.origin,
                "device_used_bytes": int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used),
                "kv_stats": stats,
            })
