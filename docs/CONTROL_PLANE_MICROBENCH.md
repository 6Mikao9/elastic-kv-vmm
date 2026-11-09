# Future-reservation control-plane microbenchmark

This benchmark isolates the CPU accounting used by known-future prefill
reservations and request-growth admission. It does not execute a model and does
not measure TTFT, TPOT or throughput.

The interval planner previously rescanned every active request and every
shared-prefix group at each event. The updated path maintains active bytes and
the per-group shared-prefix deduction incrementally with a lazy minimum heap.
The reservation broker similarly maintains its outstanding byte total instead
of summing every reservation for each admission decision. The managed rollout
executor now computes each deadline-priority key once, performs one sort per
scheduling pass, indexes cohorts instead of rescanning all pending requests for
each group, and removes a dispatched cohort from its queue in one pass.

Run from the repository root:

```bash
python benchmarks/control_plane_micro.py \
  --output results/future-planning/control-plane-micro.json
```

The benchmark checks the optimized interval result against the previous scan on
every generated workload before timing both implementations. Treat the timings
as scheduler control-plane evidence only; real serving benefit still requires
paired SGLang measurements on an exclusive GPU window.

One local Windows CPU run used a Python 3.12 runtime, 21 timing repeats and
1,000 reservation-total reads per sample. The checked-in raw result is
[control-plane-micro-20261109.json](../results/future-planning/control-plane-micro-20261109.json).

| requests | interval scan | incremental interval | speedup |
|---:|---:|---:|---:|
| 64 | 0.308 ms | 0.126 ms | 2.4x |
| 256 | 0.976 ms | 0.498 ms | 2.0x |
| 1024 | 3.716 ms | 1.777 ms | 2.1x |

| reservations | 1,000 scanned reads | 1,000 incremental reads | speedup |
|---:|---:|---:|---:|
| 64 | 0.258 ms | 0.028 ms | 9.1x |
| 256 | 0.808 ms | 0.029 ms | 28.1x |
| 1024 | 3.426 ms | 0.028 ms | 121.9x |

The worst-case rollout benchmark makes every request due immediately and keeps
every request as its own cohort. This exposes the previous restart-from-head
ordering, which recomputed a parsed deadline key and `min()` scan repeatedly.

| due requests | repeated `min` ordering | single sort | speedup |
|---:|---:|---:|---:|
| 64 | 9.256 ms | 0.289 ms | 32.0x |
| 256 | 146.853 ms | 1.169 ms | 125.7x |
| 1024 | 2,353.714 ms | 4.869 ms | 483.4x |

The execution-group benchmark uses eight requests per cohort. The previous
implementation scanned the whole request list once for every cohort; the index
pass fills each group in one traversal.

| requests / cohorts | full scan per cohort | indexed grouping | speedup |
|---:|---:|---:|---:|
| 64 / 8 | 0.018 ms | 0.009 ms | 2.0x |
| 256 / 32 | 0.167 ms | 0.035 ms | 4.8x |
| 1024 / 128 | 2.256 ms | 0.135 ms | 16.7x |

These numbers show that the control-plane work no longer repeatedly walks every
active request or promised reservation. They do not establish model latency or
throughput benefit; deployment value depends on how often the scheduler enters
these paths and must be measured in a paired serving run.
