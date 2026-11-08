# Joint-demotion owner index microbenchmark

This benchmark isolates scheduler-side candidate search. It does not measure
CUDA migration, model serving, TTFT or end-to-end throughput.

## Method

The synthetic case contains 32 eligible requests and permits groups of two to
four requests. The exhaustive control enumerates combinations up to the
configured search limit. The indexed path maintains request-to-page/chunk and
chunk-to-owner maps, expands only owner closures whose summed reference count
can release a physical chunk, and then applies the same exact page-union
validation as the control.

Each scenario ran 500 times in the pinned SGLang 0.5.17 CPU test container:

```bash
python3 benchmarks/group_search_micro.py --scenario all --repeats 500
```

Raw output is in [group-index-micro.json](../results/future-planning/group-index-micro.json).

## Results

| scenario | exhaustive examined | indexed examined | exhaustive median | indexed median |
|---|---:|---:|---:|---:|
| no candidate | 256 | 0 | 0.731 ms | 0.090 ms |
| one shared pair at the tail | 256, no group found in this rotation | 1 | 0.748 ms | 0.097 ms |
| sixteen disjoint shared pairs | 256, 118 groups | 16 | 0.946 ms | 0.203 ms |

The no-candidate and tail cases reduce scheduler-side examination by roughly
an order of magnitude. The many-pair case avoids redundant three- and
four-request supersets when the complete two-request owner closure already
releases the chunk. These are deterministic search counts plus local CPU
timings; they are evidence for the control-plane optimization only.

## Boundaries

- The scheduler still computes candidate pages and exact reference counts for
  eligible requests before calling the index.
- Overlapping request pages use the exact bounded-enumeration fallback because
  additive per-request chunk counts differ from page-union semantics.
- TP agreement, migration preparation, transfer cost, dwell and byte caps are
  unchanged.
- Production value requires the pending real-GPU paired run with the same
  requests and a comparison of `group_search_evaluated`, migration time,
  decode latency and released bytes.
