# Performance, Graph compatibility, and application value

The original results showed physical memory savings, but did not establish
a serving-capacity benefit. This follow-up separates three questions.

## Allocator overhead

Compare baseline, V4 and V5 under the same eager configuration on 0.6B.
V5 replaces whole-capacity free-slot scans with ordered free intervals and
a scalar free count. Its idle-only policy tracks live slots without doing
per-chunk reference updates at every decode step. Existing slot IDs,
attention kernels and physical backing policy remain comparable.

## CUDA Graph compatibility

Compare baseline, V4 and V5 with exact-batch Graphs on both models. Capture
batch sizes 1 through 8; larger batches fall back to eager execution.
The four original workloads, three repeats and twelve fixed correctness
probes are unchanged. Actual replay counts and physical mapping/reclaim
counts must be nonzero; setting a Graph flag alone is insufficient evidence.

Next test padded buckets separately. A permanently reserved request-table
row points only at the permanent KV sink. Before metadata construction,
padded request-index lanes are overwritten with this row. Batch shrinkage
must not leave a stale request row pointing to unmapped KV. Physical remaps
occur outside capture/replay, before a token slot is exposed to the model.
Reclamation waits for completed GPU work. Prefix retention, overlap,
speculation and distributed inference remain unsupported.

## An idle co-tenant and a long-document burst

Keep the 0.6B model (A) resident and idle while 4B (B) receives forty
concurrent requests, each with 6,144 synthetic input IDs and 512 output
tokens. Both servers initially request 327,680 logical KV slots, context
8,192, prefill chunk 2,048, at most 64 running requests and memory fraction
0.95. SGLang's existing startup profiler remains enabled and may reduce
the second fixed pool's actual capacity. Both requested and actual capacity
must be reported. This is a deployment/admission comparison, not an
equal-actual-capacity allocator microbenchmark.

Compare three deployments on an otherwise idle A100:

- Fixed pools with both requested capacities large.
- V5 with both requested capacities large, backing only live demand.
- A manually tuned fixed-pool control with A capped at 32,768 slots and B
  requesting 327,680. This control knows which model will be idle and gives
  a useful upper bound on what static tuning can achieve in this episode.

One first burst is recorded separately, followed by three measured bursts.
Record completed responses, throughput, TTFT, end-to-end latency and sampled
whole-device memory. Show TTFT threshold sensitivity at 5, 10, 20 and 30
seconds, plus an illustrative 60-second completion threshold. These are
example service targets, not measured user requirements. Do not choose a
threshold after looking at the results or claim production tail guarantees.

The hypothesized benefit is less queuing when an idle model's reserved
capacity would otherwise constrain a busy peer. Physical savings cannot
automatically be translated into a percentage cost saving: that additionally
requires placement, traffic traces, service targets and a smaller GPU count
or a cheaper device that actually meets them. This experiment has no global
broker, does not reclaim live KV and does not guarantee the simultaneous
worst-case demand of both large virtual pools.

## Reproduction commands

Run from the experiment root on an idle A100 with Docker, the built native
library and both pinned Hugging Face snapshots. Set `GPU` to the selected
device index and `HUB` to the absolute model hub directory on the host.
The wrappers mount the model hub read-only and use the locally available
`training-sandbox:torch280-cu128` image by default.

```sh
OUT=results/phase2
bash scripts/run_phase2.sh "$GPU" "$HUB" "$OUT"
bash scripts/run_padding_cases.sh "$GPU" "$HUB" "$OUT"

bash scripts/container_capacity.sh "$GPU" "$HUB" baseline "$OUT/capacity/baseline"
bash scripts/container_capacity.sh "$GPU" "$HUB" v5 "$OUT/capacity/v5"
bash scripts/container_capacity.sh "$GPU" "$HUB" baseline "$OUT/capacity/baseline-tuned" \
  --capacity-a 32768

bash scripts/container_checks.sh "$GPU" "$OUT/checks"
python3 scripts/run_activation_cases.py "$GPU" "$HUB" "$OUT"
```

The first matrix wrapper runs nine arms and the padding wrapper adds six.
The padding wrapper sets `EKV_EXECUTION=graph-pad` and calls
`scripts/container_case.sh GPU HUB SNAPSHOT_REL MODEL_LABEL MODE OUTPUT 3`.
To run one keep8 arm manually, use the same call with `MODE=v5`,
`EKV_RETAIN_CHUNKS=8` and output `$OUT/<model>/graph-pad/v5-keep8`.
Use a new output path for a fresh run. `EKV_RESUME=1` makes the two matrix
wrappers skip arms they validate as complete.

The checks wrapper runs the 13-test suite and a separate synthetic CUDA
pressure case for each of `off` and `both`. Activation calibration runs
Qwen3-4B padded Graph with V5 keep8 in four cases. It measures telemetry
with prefill chunks 512, 2048 and 8192, then runs the combined trim and
recycle policy with chunk 8192. The calibrated headroom rounds the measured
peak growth for chunk 8192 up to a multiple of 64 MiB and adds 128 MiB.
The calibration script supports `--resume` and verifies the recorded
configuration and matching correctness probes.

The separate real pressure harness uses GPU ballast and matched prefill
traffic. Choose target free memory, prefill headroom and a ballast limit in
MiB before the comparison. Run each policy on an idle GPU with the same
parameters and distinct output directories.

```sh
for policy in telemetry both; do
  bash scripts/container_prefill_pressure.sh "$GPU" "$HUB" \
    "$OUT/prefill-pressure/$policy" "$policy" \
    "$TARGET_FREE_MIB" "$HEADROOM_MIB" "$BALLAST_LIMIT_MIB"
  sleep 3
done
```

The published pressure scenario uses target 512 MiB, headroom 768 MiB and
ballast limit 73,728 MiB. It warms a 7,168-input/16-output request, then
measures three 128-input/64-output requests and checks a separate full-token
probe. Ballast is created only after startup and warmup. This checks reclamation
under pressure, not realistic co-tenant compute contention or OOM prevention.

Summarize on CPU after the corresponding arms finish. The exact-only
summary can be generated by omitting both inclusion flags and writing to
`$OUT/summary-exact`. Plotting needs Matplotlib and writes PNG and SVG.

```sh
python3 benchmarks/summarize_phase2.py "$OUT" --include-padding --include-retention \
  --output-dir "$OUT/summary"
python3 benchmarks/summarize_capacity.py --input-dir "$OUT/capacity" \
  --output-dir "$OUT/capacity"
python3 benchmarks/summarize_activation.py "$OUT/activation" \
  --output "$OUT/activation/summary.json"
python3 benchmarks/summarize_prefill_pressure.py "$OUT/prefill-pressure" \
  --output "$OUT/prefill-pressure/summary.json"
python3 benchmarks/plot_phase2.py --results-dir "$OUT" \
  --summary "$OUT/summary/summary.json" --output-dir docs/figures
```

## Source and evidence provenance

Use each arm's `source_manifest.json` to identify the configuration and
source hashes recorded when the arm ran. Exact, padding, capacity and
activation experiments may use different source snapshots. The current
checkout supports reproduction, while the per-arm manifest identifies
the original run. Preserve the raw result, correctness and stats files
alongside the manifest when recomputing summaries.

Archived source directories preserve original line-ending bytes so the recorded
SHA-256 hashes remain verifiable. Run `python3 scripts/verify_phase2_sources.py`
to check that each recorded source hash has a matching current or archived file.

The original exact-Graph baseline stats were last published at capture.
For those runs, the summary accepts the per-arm `graph_evidence.json`
record of periodic SGLang decode log lines with Graph enabled and the
SHA-256 digest of the source log. The recorded 132 lines are evidence
that Graph decode occurred. They do not count all replays. The summary
preserves the raw adapter stats and stores log evidence separately, without
filling in a replay counter.
