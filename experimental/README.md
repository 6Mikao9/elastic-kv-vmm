# VMM addressing and runtime-budget experiments

This directory is an **isolated research scaffold**, not an installed SGLang
adapter. The existing MiniMax serving path and all engine comparison flags
remain unchanged. The CPU checks passed locally on 2026-11-04; the new CUDA
arena and Triton gather have **not yet run on a GPU**. The four-A100 formal
HiSparse comparisons still take priority when the agreed window becomes free.

## Fixed virtual positions

`VirtualKVLayout` gives every request an independent, chunk-aligned virtual
range. Token `t` has a fixed byte offset within that range. `VmmResidentArena`
uses the existing CUDA `Backing` to commit growing prefixes and release whole
tail chunks at an explicit quiescent boundary. Chunk zero is a mapped sink.
No cross-request tail sharing, prefix sharing, offload, or model integration
is implemented here. New rows must be initialized before reading, including
regrowth within a retained chunk. Views are borrowed and must be retired before
the ranges they reference are discarded or the arena closes.

The GPU microbenchmark uses the **same VMM allocation, payload addresses,
support set, output order and gather computation** in both arms:

- `software_table`: request position to logical ID to element offset;
- `direct`: request base plus token position times row stride.

This isolates two dependent software lookups. It does not compare different
paged layouts, compute full attention, select Top-K, or measure model throughput.
Both arms still use GPU hardware address translation. The ordinary software
tables remain allocated during both arms, so it is not a table-memory savings
experiment. Payloads are checked exactly, invalid lanes are checked separately,
warmup precedes timing, and arm order alternates across rounds.

Only after verifying exclusive access to an idle GPU, in an environment with
the compiled `native/libelastic_vmm.so`, CUDA PyTorch, CuPy and Triton:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.bench_virtual_addressing \
  --exclusive-gpu-confirmed --output results/va-gather-attempt-1.json
```

The flag acknowledges a resource check; it is **not** a GPU lease or an
automatic contention detector. Use a fresh result filename. Do not run this
on a card occupied by another service. The output includes per-round CUDA-event
times and a separate real VMM release check. It does not claim an end-to-end gain.
These events bracket host-launched calls and can include host-dispatch gaps;
very short kernels will need a graph-replay follow-up to isolate device time.

## Sparse and clustered support

```bash
python -m scripts.analyze_virtual_layout \
  --output results/va-layout-attempt-1.json
```

This is CPU geometry, not GPU execution or a LouisKV reproduction. In the
recorded 65,536-token example, each selected row is 512 bytes and 2,048 selected
tokens contain 1 MiB of payload. A 2 MiB VMM chunk size requires 2 MiB for the
contiguous support, 32 MiB for a uniform scattered support, and 16 MiB for the
synthetic clustered support, excluding the shared sink. These figures describe
the chunks touched at fixed token positions; they are not transfer volumes.
The compact-payload lower bound excludes metadata, packing work and allocation
slack. The full-resident arena currently commits full prefixes, not only these
selected chunks.

LouisKV clusters identify which rows to retrieve. VMM capacity management can
support a changing retrieved working set, but does not remove cluster-membership
indices or packing. A semantic cluster need not be contiguous in original token
position. Allocating a separate VMM chunk per cluster can increase padding.
Compare identical real selection traces before claiming a benefit from cluster
placement, direct addressing, or sparse page caching. See the
[related-work review](../docs/HISPARSE_RELATED_WORK.md).

## Runtime budget ledger

`BudgetLedger` is a single-scheduler-thread accounting component. It can label
weights, sparse indexer, resident KV, hot KV, activations, workspace, Graph memory
and temporary migrations by category. The ledger distinguishes actual committed
allocations, pending peak reservations, known unmanaged usage and a safety margin.
It rejects new tickets when their added peak does not fit. Old and new migration
buffers remain charged until the allocator confirms release of the old allocation.

Use canonical allocation keys: aliases of one physical allocation must not be
registered twice. External usage must exclude allocations already charged to the
ledger. A ticket's final charge must include all still-live buffers; cancel only
after attempted allocations have rolled back. An observed over-budget condition
keeps all real charges and blocks new reservations instead of inventing free memory.

The ledger is **not wired to CUDA or PyTorch allocation hooks**. It does not yet
predict activations, reserve driver memory, coordinate TP, prevent other processes
from allocating, or guarantee absence of OOM. It is a tested accounting contract
for subsequent admission hooks. The first integration must occur after an actual
batch plan is known and before its KV/temporary allocations or kernels launch.

The serving hypothesis is: better peak accounting and timely reclamation may
reduce conservative unused headroom, permitting more concurrent requests under
the same latency and stability targets. Compare the normal baseline, its highest
stable tuned memory budget, and the VMM system with matched algorithms and quality.
Report achieved concurrency, throughput, TTFT/TPOT, failure/OOM rate, peak memory,
mapping and migration cost. A higher utilization setting alone is not evidence
of a better memory manager.

Separate the instance's total memory cap from the split inside that cap. More
KV capacity during decode can come from reusing idle activation capacity without
raising the total cap. Raising the cap is a second experiment, justified only by
validated peak reservations and a retained safety margin. For tensor parallelism,
admission must fit every rank, including the most constrained rank. Reclaimed
logical slots do not count as free physical bytes until whole-chunk release has
actually completed. Extra concurrency helps throughput only while the workload
can benefit from it; measure latency and transfer pressure as the batch grows.

## Verification

```bash
python -m unittest tests.test_virtual_layout tests.test_budget_ledger \
  tests.test_scheduler_budget tests.test_adaptive_policy tests.test_adaptive_runtime -v
```

CPU tests cover byte/element boundaries, rows crossing chunks, independent request
ranges, safe tail release, failed allocation state, migration peak reservations,
external pressure, and unchanged whole-chunk candidate results across randomized
layouts. They do not validate CUDA compilation, graph safety, quality or throughput.
