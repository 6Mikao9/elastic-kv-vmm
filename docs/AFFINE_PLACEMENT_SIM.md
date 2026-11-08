# Request-affine placement simulation

This CPU experiment asks whether a stable per-request virtual base can reduce
VMM tail waste without introducing a software KV block table. It compares two
layouts while keeping the same direct token formula:

```text
byte_offset = request_base + token_position * row_bytes
```

- `fixed-affine` gives every request a chunk-aligned slot sized for the largest
  final sequence in the controlled workload.
- `packed-affine` reserves each request's exact final byte extent in one global
  virtual arena. Best-fit allocation places neighboring requests at arbitrary
  chunk offsets, so their live prefixes may share a physical chunk.

Both layouts reserve the full final extent at admission, which keeps the base
stable while decoding. VMM backing is charged only for chunks touched by the
live prompt/output prefix. Reclaiming one request removes a chunk only after no
other live request touches it.

## Reproduce

```bash
python -m benchmarks.affine_placement_sim --scenario all
python -m benchmarks.affine_placement_sim \
  --arena-mib 64 \
  --output results/affine-placement-cpu/summary-capacity-pressure.json
python -m benchmarks.affine_placement_sim \
  --arena-mib 128 \
  --output results/affine-placement-cpu/summary-same-active-limit.json
```

The checked-in fixtures use 96 requests, a deterministic seed, 2 MiB chunks,
512-byte token rows and at most 16 active requests. They cover mixed lengths,
a short-heavy mix, and equal 4,096-token sequences.

## Results

At a 64 MiB arena, fixed slots consume 8 MiB each because the mixed workloads
contain a roughly 16K-token request. Packed intervals fit more requests in the
same virtual capacity. The following table uses the peak mapping sample for
each arm; `max active` is the maximum reached anywhere in the run.

| scenario | layout | max active | peak active | peak mapped | peak useful | max start delay | makespan |
|---|---|---:|---:|---:|---:|---:|---:|
| mixed | fixed-affine | 8 | 7 | 26 MiB | 19.60 MiB | 3,796.7 ms | 5,375.4 ms |
| mixed | packed-affine | 16 | 16 | 54 MiB | 30.82 MiB | 1,618.3 ms | 3,294.8 ms |
| short-heavy | fixed-affine | 8 | 8 | 48 MiB | 39.12 MiB | 1,315.0 ms | 3,091.3 ms |
| short-heavy | packed-affine | 16 | 8 | 54 MiB | 44.89 MiB | 1,207.7 ms | 3,016.7 ms |
| equal-length | fixed-affine | 16 | 16 | 32 MiB | 24.64 MiB | 2,037.4 ms | 2,726.2 ms |
| equal-length | packed-affine | 16 | 16 | 32 MiB | 24.64 MiB | 2,037.4 ms | 2,726.2 ms |

The packed arm completed the mixed workload with a 38% lower makespan and a
57% lower maximum admission delay. Its raw peak backing was higher because it
kept twice as many requests active. The result is therefore a capacity and
queueing effect, not an isolated reduction in per-request mapping cost.

To remove the active-request-count difference, the second run uses a 128 MiB
arena. Both arms reach the configured 16-request limit and have the same
admission and completion times.

| scenario | fixed peak | packed peak | packed shared savings at peak | fixed virtual reservation | packed virtual reservation |
|---|---:|---:|---:|---:|---:|
| mixed | 52 MiB | 52 MiB | 12 MiB / 5 chunks | 128 MiB | 52.72 MiB |
| short-heavy | 60 MiB | 60 MiB | 8 MiB / 4 chunks | 128 MiB | 61.51 MiB |
| equal-length | 32 MiB | 32 MiB | 0 MiB | 32 MiB | 32 MiB |

At matched concurrency, packing removes 8-12 MiB of independently counted
chunk backing by co-occupying chunks. It does not automatically reduce the
union of physical chunks touched: the mixed and short-heavy runs still peak at
the same union. Exact-length equality is a useful negative control because a
4,096-token request already occupies exactly 2 MiB.

## Interpretation

The simulation supports one narrow design choice: an exact-length packed arena
can retain a single affine base per request without paying the fixed maximum-
length slot reservation. This reduces virtual capacity consumed by sparse
workloads and can allow more active requests before physical backing pressure.

It does not make tail chunks free. Whole-chunk VMM mapping still rounds the
union of live prefixes, and a final partial chunk remains charged until every
request touching it finishes. Packing also changes reclaim coupling: adjacent
requests can hold a shared chunk after one side exits. Reservations must be
exact and immutable for the request lifetime; growing into unplanned space
would change the affine base unless a separate mapping indirection is added.

The next step should replay a real serving trace with known prompt and output
lengths, then compare actual mapped bytes, admission delay, mapping calls and
end-to-end latency on an exclusive GPU. This CPU geometry result is not a
throughput or attention-kernel claim.

## Boundaries

- The prompt and final output length must be known when the request is admitted.
- The global arena uses stable, non-overlapping request bases; there is no
  cross-request prefix sharing or offload.
- Best-fit fragmentation is modeled, but scheduler reordering and compaction
  are not.
- The service timeline uses fixed token rates and excludes batching effects.
- TP ranks need the same geometry and admission result; this fixture models
  one rank only.
