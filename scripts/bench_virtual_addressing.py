"""GPU gather microbenchmark. No model/serving speedup is measured here."""

import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--selected", type=int, default=2048)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--exclusive-gpu-confirmed", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve earlier attempts")
    if not args.exclusive_gpu_confirmed:
        parser.error("verify exclusive idle GPU access before using --exclusive-gpu-confirmed")
    if (min(args.length, args.selected, args.width, args.batch, args.rounds, args.calls) <= 0
            or args.selected > args.length or args.width > 4096):
        parser.error("invalid shape or repetitions")

    import torch
    from experimental.virtual_layout import VirtualKVLayout
    from experimental.resident_arena import VmmResidentArena
    from experimental.sparse_read import gather_rows
    from scripts.analyze_virtual_layout import support_sets

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one independently verified idle GPU")
    torch.cuda.set_device(0)
    layout = VirtualKVLayout(args.batch, args.length, args.width)
    arena = VmmResidentArena(layout)
    rows = raw = view = expected = output = table = offsets = None
    results = []
    try:
        for req in range(args.batch):
            arena.reserve(req, args.length)
            view = arena.row_view(req, args.length, torch.bfloat16)
            # A bounded integer pattern is exactly representable in BF16.
            values = (torch.arange(args.length * args.width, device="cuda") + req * 17) % 97
            view.copy_(values.reshape(args.length, args.width))
            del values
        view = None
        raw = arena.backing.raw.view(torch.bfloat16)
        request_ids = torch.arange(args.batch, device="cuda", dtype=torch.int64).flip(0)
        lengths = torch.full((args.batch,), args.length, device="cuda", dtype=torch.int64)
        # Reverse logical IDs to exercise two true dependent lookups, while
        # leaving all payload addresses and data identical in both modes.
        count = args.batch * args.length
        table = torch.arange(count, 0, -1, device="cuda", dtype=torch.int64).reshape(args.batch, args.length)
        flat_position = torch.arange(count, device="cuda", dtype=torch.int64)
        addresses = (layout.chunk_bytes // layout.itemsize
                     + (flat_position // args.length) * (layout.request_bytes // layout.itemsize)
                     + (flat_position % args.length) * args.width)
        offsets = torch.zeros(count + 1, device="cuda", dtype=torch.int64)
        offsets[table.flatten()] = addresses
        del addresses, flat_position
        output = torch.empty((args.batch, args.selected, args.width), device="cuda", dtype=torch.bfloat16)
        for name, support in support_sets(args.length, args.selected).items():
            positions = torch.tensor(support, device="cuda", dtype=torch.int64).expand(args.batch, -1).contiguous()
            columns = torch.arange(args.width, device="cuda")
            expected = ((positions.clamp_min(0)[..., None] * args.width + columns
                         + request_ids[:, None, None] * 17) % 97).to(torch.bfloat16)
            expected.masked_fill_(positions[..., None] < 0, 0)
            def call(indirect, read_positions=positions):
                gather_rows(raw, read_positions, request_ids, lengths, table, offsets,
                            output, layout, indirect=indirect)
            for indirect in (False, True):
                call(indirect)
                if not torch.equal(output, expected):
                    raise AssertionError(f"payload mismatch: {name}, indirect={indirect}")
                # Check invalid positions separately. All timed lanes remain
                # real reads, including the supported --selected=1 case.
                invalid_expected = expected.clone()
                invalid_expected[:, 0, :] = 0
                for invalid in (-1, args.length):
                    invalid_positions = positions.clone()
                    invalid_positions[:, 0] = invalid
                    call(indirect, invalid_positions)
                    if not torch.equal(output, invalid_expected):
                        raise AssertionError(f"invalid-lane mismatch: {invalid}, {indirect}")
                del invalid_expected, invalid_positions
                for _ in range(30):
                    call(indirect)
            torch.cuda.synchronize()
            timings = {"direct": [], "software_table": []}
            for repeat in range(args.rounds):
                for indirect in ((False, True) if repeat % 2 == 0 else (True, False)):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.calls):
                        call(indirect)
                    end.record()
                    end.synchronize()
                    timings["software_table" if indirect else "direct"].append(
                        start.elapsed_time(end) * 1000 / args.calls)
            stats = {k: {"p50_us": statistics.median(v), "samples_us": v} for k, v in timings.items()}
            results.append({"pattern": name, "exact_gather_check": True, **stats})

        # A separate capacity operation, outside the timed kernels. No borrowed
        # tensor or future kernel accesses the discarded request afterwards.
        pointer = arena.backing.base
        before = arena.backing.backed_bytes
        output = expected = raw = rows = view = None
        started = time.perf_counter_ns()
        arena.resize_quiescent(0, 0)
        reclaim_ns = time.perf_counter_ns() - started
        after = arena.backing.backed_bytes
        if arena.backing.base != pointer or before - after != layout.request_bytes:
            raise AssertionError("independent request reclamation changed VA or unexpected bytes")
        report = {"scope": __doc__, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "length": args.length, "selected": args.selected, "width": args.width,
                  "batch": args.batch, "rounds": args.rounds, "calls_per_round": args.calls,
                  "timing": "CUDA events around host-launched calls; may include host-dispatch gaps",
                  "comparison": "identical payload addresses/data; two software lookups vs fixed VA arithmetic",
                  "table_bytes": table.numel() * table.element_size() + offsets.numel() * offsets.element_size(),
                  "results": results, "capacity": {"before_bytes": before, "after_bytes": after,
                      "released_bytes": before - after, "reclaim_ns": reclaim_ns, "stable_va": True},
                  "limitations": ["Gather only, not full attention or end-to-end serving",
                                  "No LouisKV semantic clustering or HiSparse offload",
                                  "Does not include real paging-layout locality changes",
                                  "Very short kernels may require graph replay to isolate GPU execution time",
                                  "No performance claim until this GPU benchmark completes"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2))
    finally:
        rows = raw = view = expected = output = table = offsets = None  # noqa: F841
        arena.close()


if __name__ == "__main__":
    main()
