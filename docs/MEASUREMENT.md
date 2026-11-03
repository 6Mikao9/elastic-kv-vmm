# Measurement protocol

The benchmark runs real Qwen3 inference through SGLang's HTTP API on one
NVIDIA A100 80 GB. The setup uses SGLang 0.5.2, PyTorch 2.8 with CUDA
12.8, Triton attention, page size 1, a pool cap of 32,768 token slots,
context length 8,192 and chunked prefill size 2,048. Model weights are
loaded read-only. CUDA graphs, schedule overlap and radix caching are
disabled across the comparison. The results apply to this eager
configuration. Production SGLang performance depends on the omitted
features and the serving workload.

The client sends deterministic synthetic input token IDs. These runs measure
serving behavior and check token agreement on fixed probes. They provide no
measure of language quality.

Each workload has three measured repeats, each preceded by warmup. The client
uses normal streaming requests during the performance phase. Throughput is
the number of generated tokens divided by the elapsed wall time for a phase.
The summary reports the median and range across repeats. TTFT runs from
client request start to the first token-bearing stream event. TPOT divides
the remaining time by the number of remaining tokens. The median and
linearly interpolated p95 pool request observations across repeats. With
small samples and three repeats, p95 describes this experiment and should
not be treated as a production tail estimate. GPU clocks were not locked.

TPOT is a client-side streaming estimate, not a measurement of each GPU
decode step. An SSE event may contain several generated tokens. Dividing
the time after the first token-bearing event by total output tokens minus
one can therefore understate the average spacing of the tokens delivered
after that event. The same definition is used across all arms.

The correctness phase runs separately, after performance measurement, using
12 fixed serial greedy requests. It crosses three prompt lengths, two output
budgets and two prompt seeds. The comparison joins on request ID, checks
prompt hashes and compares complete output token sequences to the same
model's fixed-pool baseline. The earlier baseline's correctness phase
failed under the original protocol. Comparing greedy outputs across
different batch shapes can expose numerical differences in inference.
The earlier failure motivated the controlled serial check and does not
establish that VMM corrupted the cache.

NVML provides device-wide used bytes, including weights and runtime
allocations. The adapter provides KV backed bytes from its sampled
mapping state. Peak means the maximum sample during the measured active
window. Idle means the median sample during the post-request idle window.
The 50 ms sampling interval can miss short peaks. Baseline KV backed
bytes and their reductions are blank when the fixed pool provides no
mapping telemetry. NVML reductions use the corresponding same-model
baseline workload and can be negative.

Select runs explicitly so that pilot directories stay out of the comparison.
For example, from the project root with the completed cases available:

```sh
python benchmarks/summarize.py \
  --case Qwen3-0.6B-Base:baseline:results/qwen06/baseline-final \
  --case Qwen3-0.6B-Base:v0:results/qwen06/v0 \
  --case Qwen3-0.6B-Base:v1:results/qwen06/v1 \
  --output-dir results/summary
```

An incomplete or invalid input is rejected. A token or prompt mismatch
is recorded in both summaries, with request-level details in JSON, and
returns exit status 1.

The separate dual-instance experiment uses `scripts/run_dual.py`. The
single-instance runs do not demonstrate global memory arbitration or
actual out-of-memory recovery. The dual runner has no global memory broker.

The recorded dual runs use `active_sample_count`, a snapshot of the sample
list at request completion. Their legacy `peak_device_used_bytes` field
also includes post-request samples; the published summarizer ignores it.
The current runner records timestamp bounds in `active_window_s` and
computes the peak from that window. This reporting-only correction does
not alter the recorded experimental samples or serving implementation.

Source manifests record the exact deployed file bytes for each captured
stage. Git normalizes text line endings to LF; the deployed client had
mixed line endings, so its raw-byte hash differs after normalization.
The final V4 serving files and client otherwise match the captured sources.
Earlier stage manifests predate the addition of later allocator variants.
