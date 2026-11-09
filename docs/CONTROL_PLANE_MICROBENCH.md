# Future-reservation control-plane microbenchmark

This benchmark isolates the CPU accounting used by known-future prefill
reservations and request-growth admission. It does not execute a model and does
not measure TTFT, TPOT or throughput.

The interval planner previously rescanned every active request and every
shared-prefix group at each event. The updated path maintains active bytes and
the per-group shared-prefix deduction incrementally with a lazy minimum heap.
The reservation broker similarly maintains its outstanding byte total instead
of summing every reservation for each admission decision.

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
| 64 | 1.787 ms | 0.342 ms | 5.2x |
| 256 | 2.473 ms | 1.263 ms | 2.0x |
| 1024 | 10.143 ms | 5.190 ms | 2.0x |

| reservations | 1,000 scanned reads | 1,000 incremental reads | speedup |
|---:|---:|---:|---:|
| 64 | 0.779 ms | 0.079 ms | 9.9x |
| 256 | 2.351 ms | 0.083 ms | 28.4x |
| 1024 | 9.388 ms | 0.080 ms | 116.9x |

These numbers show that the control-plane work no longer repeatedly walks every
active request or promised reservation. They do not establish model latency or
throughput benefit; deployment value depends on how often the scheduler enters
these paths and must be measured in a paired serving run.
