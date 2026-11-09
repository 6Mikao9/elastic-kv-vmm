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
