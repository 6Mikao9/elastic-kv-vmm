"""CUDA-event microbenchmark for MiniMax resident lookup only."""

import json
import statistics

import torch

from adaptive_sparse.resident_lookup import resolve_resident


def main():
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one idle GPU")
    device = "cuda"
    results = []
    for batch in (1, 2, 4):
        selected = torch.arange(batch, device=device)
        rows = selected.flip(0)
        lengths = torch.full((batch,), 8192, dtype=torch.int32, device=device)
        table = torch.arange(batch * 8192, device=device).reshape(batch, 8192) + 1
        mapping = torch.arange(batch * 8192 + 1, device=device) * 3 + 5
        topk = (torch.arange(16, device=device) * 4).int().expand(1, batch, 16).contiguous()

        def torch_lookup():
            result = torch.empty((batch, 2048), device=device, dtype=torch.int32)
            blocks = topk[0, selected].long()
            positions = (blocks.unsqueeze(-1) * 128 + torch.arange(128, device=device)).flatten(1)
            valid = (positions >= 0) & (positions < lengths[selected, None])
            logical = table[rows[selected, None], positions.masked_fill(~valid, 0)]
            result[selected] = mapping[logical].masked_fill(~valid, -1).int()
            return result.unsqueeze(0)

        def triton_lookup():
            result = torch.empty((1, batch, 2048), device=device, dtype=torch.int32)
            resolve_resident(topk, table, mapping, rows, lengths, selected, result, 128)
            return result

        def reference():
            return torch.tensor([[
                ((batch - 1 - row) * 8192 + block * 512 + offset + 1) * 3 + 5
                for block in range(16) for offset in range(128)
            ] for row in range(batch)], dtype=torch.int32, device=device).unsqueeze(0)

        expected = reference()
        assert torch.equal(torch_lookup(), expected)
        assert torch.equal(triton_lookup(), expected)
        timings = {"torch": [], "triton": []}
        methods = [("torch", torch_lookup), ("triton", triton_lookup)]
        for _, call in methods:
            for _ in range(100):
                call()
        torch.cuda.synchronize()
        for round_number in range(40):
            for name, call in methods[::(-1 if round_number % 2 else 1)]:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):
                    call()
                end.record()
                end.synchronize()
                timings[name].append(begin.elapsed_time(end) * 10)
        stats = {name: {"p50_us": statistics.median(values),
                        "p95_us": statistics.quantiles(values, n=20, method="inclusive")[18]}
                 for name, values in timings.items()}
        results.append({"batch": batch, "correctness_passed": True, **stats,
                        "p50_speedup": stats["torch"]["p50_us"] / stats["triton"]["p50_us"]})
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "length": 8192, "topk_blocks": 16, "block_size": 128, "warmup_calls": 100,
                      "rounds": 40, "calls_per_round": 100,
                      "timing": "CUDA event time per call within each round",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
