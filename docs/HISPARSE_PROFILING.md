# Profile a real MiniMax HiSparse serving phase

Run the diagnostic with the same pinned SGLang image, model, four idle A100s and treatment flags as the arm under investigation. Choose a fresh output directory and pass one existing workload with `--profile-workload`. The runner uses the pinned SGLang manual Torch profiler with stack and shape collection disabled. It first warms the selected workload, then captures one batch with the same request IDs, prompt token IDs, concurrency and capped output lengths. Each workload consists of four requests. `--profile-output-token-cap` defaults to 16 and accepts values from 1 to 32. The cap applies to the diagnostic warmup and capture. The summary records each resulting request shape. [Experiment settings](HISPARSE_REPRODUCE.md)

```bash
bash scripts/container_hisparse_smoke.sh "$GPUS" "$MODEL" native \
  results/hisparse-profile-native-mixed \
  --sm80-bf16-projections --logical-capacity 65536 --host-ratio 2 \
  --profile-workload mixed
```

Use the static or pressure treatment flags from the reproduction guide when capturing those arms. The diagnostic runs separately from performance repeats, semantic probes, future schedules and the other trace diagnostics. It writes its workload marker into the source manifest and outer result, so the performance summary and plots exclude it.

`profile_traces/` contains the compressed Chrome traces. The runner calls SGLang's start endpoint without a step limit after warmup, waits for the captured requests to finish, and then calls stop to flush the traces. `profile_summary.json` requires one trace for each TP rank from 0 through 3 and positive-duration CPU operations and CUDA kernels on every rank.

Each rank reports both the union of GPU intervals and the sum of their durations. The union assumes one CUDA device per rank and checks that assumption when events identify the device. The sum counts concurrent work more than once. GPU intervals include kernels and explicit CUDA copy and memset activity. Each kernel and runtime or driver wait API name has a count, sum and maximum duration. Wait API timings measure the CPU calls. A stream wait call can enqueue a dependency and its CPU duration does not measure GPU stall time. Copy bytes are taken from explicit GPU memcpy events when present. HiSparse kernels can read mapped host memory without issuing a CUDA memcpy, so the reported bytes cover the observed copy events only. Traces can include idle time and profiler overhead around the request batch. Use the separate, unprofiled measurements for throughput and latency comparisons.

The offline reader also retains every observed CUDA Driver and Runtime API in
`cuda_api_cpu_durations_by_category_and_name`, grouped by category and API name.
This makes VMM calls such as `cuMemCreate`, `cuMemMap` and `cuMemUnmap` visible
when the raw trace contains them. These are CPU call durations and do not enter
the GPU activity union. An empty category does not establish that no VMM work
occurred. Native swap kernels also serve zero-length resident rows, while
promotion and demotion share generic copy kernels with ordinary serving;
kernel counts alone cannot establish migration counts or host traffic.
Per-action attribution still needs explicit ranges and launch correlations.
An existing frozen run can be re-read offline into a new analysis artifact;
do not overwrite its original trace or equate re-analysis with a new capture.

## Optional phase attribution

New diagnostic workers enable `EKV_PROFILE_RANGES=1` when the runner receives
`--profile-workload`. Regular performance runs explicitly clear an inherited
flag. The manifest records the setting. Workers bind the wrappers at startup;
enabling profiling later in an already-running unmarked worker does not install
them. With the flag off, the decorator returns the original callable, so the
normal per-layer path gains no profiler wrapper or context.

The Torch trace uses `ekv/` CPU annotations for native mapping/swap, mixed-mode
mapping/attention swap, the Triton resident resolver, native sparse swap, and
demotion/promotion prepare and commit. The Torch resident resolver remains
inline and is covered by its outer attention-swap scope. These labels identify
which operation enqueued work; they do not establish its critical-path cost.
`ekv_cpu_scopes_by_name` reports nested CPU wall intervals separately from GPU
activity. Summing these scopes can double-count nested work and does not measure
GPU stall or transfer volume. Use launch correlations in the raw trace for
further attribution.

The scope helper, runner isolation, trace accounting and relevant migration
paths passed 46 targeted tests in the A100 development container. A separate
real Torch CPU capture checked nesting and original exception propagation.
These checks do not validate a marked full-model GPU capture. The r6/r7/r8
frozen comparison source predates these additions and remains unchanged.

The [pinned-class smoke](../tests/smoke_profile_ranges_pinned.py) subsequently
passed inside the actual MiniMax image (Torch 2.15.0.dev20261005+cu134), with
GPU visibility disabled. It verifies the coordinator source hash, method
signatures, idempotent installation and real bound calls. Deliberately invalid
CPU inputs fail before any kernel; the original exception type and message
survive instrumentation, and the exported Chrome trace contains exactly one
positive `user_annotation` event for each native scope. CUDA remains
uninitialized. [Evidence and raw artifact hashes](../results/profiling/scope-pinned-bound-20261109.json)
cover this CPU entry-path check only. Successful native KV operations, mixed
dispatch, and GPU launch attribution still require the real GPU runs.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -B tests/smoke_profile_ranges_pinned.py \
  --output-dir /reports/scope-pinned-bound
```

## Offline timeline analysis

The [timeline reader](../scripts/analyze_hisparse_timeline.py) can inspect the
existing compressed traces without changing the capture or its frozen source:

```bash
python -m scripts.analyze_hisparse_timeline \
  /path/to/completed-run/profile_traces --expected-tp-size 4
```

It writes JSON to stdout. Save that output in a new analysis artifact rather
than overwriting the original capture. A single trace or partial directory can
also be inspected by omitting the expected TP size; that does not establish
that the full capture completed.

For each rank, the reader merges positive-duration GPU intervals and reports
the first-to-last GPU envelope, activity coverage and interior gaps, plus
separate category and stream unions. Zero-duration events cannot extend the
envelope. A gap means no activity was recorded in this trace: it does not prove
that the entire device was idle or that those cycles can be eliminated.
Category and stream totals may overlap. Recorded CPU-op overlap is a union of
wall-time scopes, which can include sleeping or waiting; it is not CPU usage.

Host synchronization calls and stream-wait enqueue calls are reported
separately. Direct kernel launches are linked only through unique explicit
correlation IDs, with unmatched and ambiguous events counted. The signed
kernel-start minus launch-end interval can be negative because execution can
begin before the launch call returns. It is not a measurement of GPU stall or
critical-path cost. The reader does not infer dependencies from nearby events.

The current r8 profiles use the same `mixed` workload for native and static
arms, but `mixed12k_unequal` for the adaptive group-2 arm. The latter supports
diagnosis within that run, not a matched speed comparison against those two
profiles. The default 16-token output cap also truncates both long requests to
16 tokens, removing the original 128-versus-64 output-length asymmetry. Use
the recorded request shapes, not the workload name, when interpreting results.
These captures run with CUDA Graph, scheduler overlap and radix caching
disabled. They cannot establish the benefit of enabling those features.

## Nsight Systems installation, 2026-11-09

Nsight Systems CLI **2026.5.1.161** is now extracted into the persistent container
workspace. The archive passed its official repository SHA-256 and size checks;
the executable returned its expected version under the unprivileged project UID.
CPU-only NVTX collection and SQLite export subsequently passed in both the
development container and the pinned MiniMax image with GPU visibility disabled.
The exported database contains the intended positive-duration NVTX range in
each case. [CPU-only validation](../results/profiling/nsys-cpu-nvtx-20261109.json)

CUDA and NVTX tracing also passed on an A100 with host driver `580.178.04` in
the pinned MiniMax image `sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87`.
The model-free smoke replayed the existing CUDA Graph correctly after a VMM
remap. The SQLite export contains 15 positive-duration CUDA kernel events,
one GPU graph trace, ten NVTX ranges, and four calls each to `cuMemCreate`,
`cuMemMap`, `cuMemUnmap`, and `cuMemRelease`. Those APIs appear in
`CUPTI_ACTIVITY_KIND_RUNTIME` in this export. Their durations measure CPU API
calls. The roughly 206 ms `vmm_handoff/map` NVTX region includes cold-start and
profiler overhead, so it cannot estimate typical map cost. Full-model tracing
remains pending in the new r7 workflow. This smoke validates compatibility
for model-free capture only. [GPU validation](../results/profiling/nsys-vmm-gpu-20261109.json)

The development environment is Torch 2.8.0+cu128, but the pinned MiniMax image
contains Torch 2.15.0.dev20261005+cu134. The profiler version was selected for the
latter. NVIDIA lists CUDA 13.4 support in [Nsight Systems 2026.5.1](https://developer.nvidia.com/nsight-systems/get-started).
The [official amd64 CLI package](https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb)
is 225,594,886 bytes, with SHA-256
`61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08`,
as recorded in the [NVIDIA package index](https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/Packages).
It was unpacked with `dpkg-deb -x`; neither system packages nor the frozen
inference image were modified. The complete extracted target directory is needed.
Sourcing `scripts/container_session.sh` exposes this installed CLI as `nsys` and
`$NSYS` for that shell, when the sibling tool directory exists.

```bash
NSYS=/workspace/elastic-kv-vmm/tools/nsys-2026.5.1.161/root/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
"$NSYS" --version
"$NSYS" status --environment
```

For a separate model-free diagnostic, run from the project root inside the
pinned MiniMax image. Set `GPU_UUID` to one explicitly admitted idle A100 and
use fresh report destinations. The default Torch backend exports
`trace.json` alongside the handoff correctness result.
Bind the project and a writable `/reports` directory into that container.
For Nsight, bind the entire extracted `target-linux-x64` directory read-only
at `/nsys/target-linux-x64`, then set `NSYS=/nsys/target-linux-x64/nsys`.
Preserve that directory name: mounting its contents directly at `/nsys`
was rejected by the CLI in the pinned-image CPU smoke. Neither tool mounting
nor the `--ack-idle-gpu` flag establishes GPU exclusivity.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" python -m scripts.profile_vmm_handoff \
  --ack-idle-gpu --output-dir /reports/vmm-handoff-torch
```

For external CUDA and NVTX tracing, select the NVTX backend so Torch does
not start its own profiler session. The script runs the same VMM and CUDA Graph
checks and emits `vmm_handoff/` ranges. The Nsight report is written separately
by `nsys` and must be checked there. `result.json` records the backend and
`external_trace_verified: false` in this mode.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o /reports/vmm-handoff-nsys python -m scripts.profile_vmm_handoff \
  --ack-idle-gpu --profiler-backend nvtx --output-dir /reports/vmm-handoff-nvtx
```

The observed host has `perf_event_paranoid=4`, and the unprivileged container's
environment check reports that CPU sampling is unavailable. No host security
setting was changed. The GPU smoke used CUDA/NVTX tracing with CPU sampling disabled.
[NVIDIA documents that tracing can remain available when the CPU sampling check fails](https://docs.nvidia.com/nsight-systems/InstallationGuide/#optional-setting-up-the-cli).
Keep Nsight collection separate from both the Torch profiler and formal timing.
Nsight Compute remains uninstalled: the current CUDA 13.4-era release lists a
newer Linux driver requirement than this host, so it is not yet a validated
diagnostic option here. [Nsight Compute requirements](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2026_3)
