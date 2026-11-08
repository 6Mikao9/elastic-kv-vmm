# Future-aware SGLang scheduler smoke

This report records one real SGLang scheduler run on an A100 with the opt-in
`elastic.future` request hint. It validates scheduler integration and failure
fallback. It is not a VMM memory benchmark and does not establish a general
latency or throughput gain.

## Setup

- One A100 80GB, GPU2, verified idle before and after the run.
- Existing `lmsysorg/sglang:dev-minimax-m31-flash` image. Its SGLang source
  exposes `kv_hints` and reports version `0.0.0.dev0`.
- Existing Qwen3-0.6B snapshot mounted read-only.
- Single GPU, TP1, page size 1, Triton attention, FCFS scheduling, overlap and
  radix cache disabled, prefill chunk 2048, and `max_running_requests=1`.
- Native baseline. The VMM adapter remains pinned to SGLang 0.5.2 and is not
  active in this newer-revision smoke.

The client first occupies the only running slot with a long request. It then
queues a slow request without a deadline and, 30 ms later, a short urgent
request with a 3,000 ms deadline. Both queued requests carry a versioned
`elastic.future` action.

## Result

The hook observed three scheduler ticks, saw a two-request hinted queue, and
reordered it once. The urgent request completed 297.6 ms before the slower
request even though it arrived second. All three responses produced their
requested token counts.

| Request | Prompt tokens | Output tokens | Deadline | Elapsed |
|---|---:|---:|---:|---:|
| initial | 256 | 256 | none | 730.8 ms |
| slow | 4,096 | 64 | none | 951.3 ms |
| urgent | 128 | 8 | 3,000 ms | 622.5 ms |

Hook telemetry:

```json
{
  "ticks": 3,
  "queue_requests": 2,
  "reordered_ticks": 1,
  "released_requests": 2,
  "fallback_missing_hint": 1,
  "fallback_invalid_hint": 0,
  "fallback_exception": 0,
  "telemetry_errors": 0
}
```

One scheduler tick contained traffic without the hint and therefore retained
the native order. This is the intended atomic fallback: ordinary application
traffic is not reordered when a mixed queue has incomplete planning metadata.

The first end-to-end attempt failed because release-count telemetry dereferenced
a missing hint after the scheduler had correctly chosen the fallback path. The
bug was fixed, covered by a unit test, and the complete smoke was rerun in fresh
output directories. Failed remote output is intentionally retained.

## Reproduce

```bash
bash scripts/container_future_smoke.sh \
  2 /path/to/Qwen3-0.6B results/future-smoke/my-run
```

The script checks that the selected GPU is idle, disables container networking,
mounts the model read-only, and refuses to overwrite an existing result. Raw
checked-in output is in
[`results/future-scheduler-smoke/`](../results/future-scheduler-smoke/).

The controlled-rollout executor can instead consume a known-future workload or
an already emitted plan. Each run is still isolated in a fresh output directory:

```bash
bash scripts/container_future_smoke.sh \
  2 /path/to/model results/future-planning/future-hints \
  --workload benchmarks/workloads/known-future-rollout.json \
  --planning-policy future --future-policy deadline

bash scripts/container_future_smoke.sh \
  2 /path/to/model results/future-planning/fifo-no-hints \
  --workload benchmarks/workloads/known-future-rollout.json \
  --planning-policy fifo --future-policy off --no-hints
```

`results.json` separates planned and actual submit/finish times, records deadline
hit/miss against the original absolute deadline, and reports throughput over the
full makespan including deliberate idle as well as over the serving window. The
executor rejects malformed plans before sending any request and fails loudly on
late dispatch, incomplete output, or conflicting TP residency telemetry.

## Limits

- `release_after_ms` is currently a soft ordering preference. It is not a hard
  admission gate; safely skipping a waiting request requires a scheduler-level
  change that preserves every native `PrefillAdder` check.
- The hook computes remaining slack from SGLang's scheduler receive time and
  never crosses the engine's native priority tier.
- SGLang 0.5.2 and 0.5.17 use the guarded compatibility transport described in
  [the compatibility boundary](KV_HINTS_COMPAT.md). The V5 eager and CUDA Graph
  integration smokes both exercised VMM backing and future queue planning in
  the same serving process. Their checked-in summaries are in
  `results/integration-smoke/v5-future-052-eager-r3/` and
  `results/integration-smoke/v5-future-052-graph-r1/`; these validate the
  integration and scheduling path, not a throughput improvement.
- The four-A100 MiniMax run additionally combined native HiSparse,
  request-level residency, VMM backing, automatic joint demotion and future
  ordering. See [the integration report](MINIMAX_FUTURE_INTEGRATION.md).
- Non-streaming requests were used, so this smoke says nothing about TTFT, TPOT,
  throughput, KV residency migration or end-to-end HiSparse performance.
