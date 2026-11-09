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

## Nsight Systems installation, 2026-11-09

Nsight Systems CLI **2026.5.1.161** is now extracted into the persistent container
workspace. The archive passed its official repository SHA-256 and size checks;
the executable returned its expected version under the unprivileged project UID.
This establishes installation, **not successful GPU tracing**. A CUDA/NVTX smoke
in the pinned MiniMax image is still required when a GPU is available.

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
# Run only in a separately admitted diagnostic, with a fresh report destination:
"$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o /reports/smoke python cuda_nvtx_smoke.py
```

The observed host has `perf_event_paranoid=4`, and the unprivileged container's
environment check reports that CPU sampling is unavailable. No host security
setting was changed. Start with CUDA/NVTX tracing and CPU sampling disabled;
[NVIDIA documents that tracing can remain available when the CPU sampling check fails](https://docs.nvidia.com/nsight-systems/InstallationGuide/#optional-setting-up-the-cli).
Keep Nsight collection separate from both the Torch profiler and formal timing.
The current host driver is 580.178.04; CUDA 13.4 tracing compatibility must be
verified against the user-mode driver actually loaded by the pinned image.
Nsight Compute remains uninstalled: the current CUDA 13.4-era release lists a
newer Linux driver requirement than this host, so it is not yet a validated
diagnostic option here. [Nsight Compute requirements](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2026_3)
