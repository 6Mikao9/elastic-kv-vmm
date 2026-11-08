# Elastic SGLang server

`elastic-sglang` is the supported launch entry point. It validates the
experimental memory policy before importing SGLang or touching CUDA, then
installs the pinned SGLang 0.5.2 adapter and passes all remaining arguments to
SGLang unchanged.

`--elastic-mode baseline` deliberately skips the VMM monkeypatch. This keeps a
native baseline available. The separate future-ordering hook uses native
`kv_hints` when present and installs the guarded 0.5.2 transport compatibility
otherwise, so it can be combined with the dense-MHA VMM modes.

```bash
python -m elastic_kv --elastic-dry-run \
  --elastic-mode v5 --elastic-retain-chunks 8

python -m elastic_kv \
  --elastic-mode v5 \
  --elastic-retain-chunks 8 \
  --model-path /models/Qwen3-0.6B-Base \
  --attention-backend triton \
  --page-size 1 \
  --disable-overlap-schedule \
  --disable-radix-cache
```

The equivalent installed command is `elastic-sglang`. Environment variables
remain supported for existing experiment scripts. Explicit elastic command-line
options take precedence over `EKV_*` values for the current process tree.

SGLang uses multiprocessing on worker startup. The launcher marks the validated
configuration and installs the same adapter when Python re-imports the launch
module in a spawned worker; a dry run or ordinary import does not import
SGLang or touch CUDA.

| Option | Default | Constraint |
|---|---:|---|
| `--elastic-mode` | `baseline` | `baseline`, `v0` through `v5` |
| `--elastic-chunk-mib` | `2` | positive integer |
| `--elastic-retain-chunks` | `1` | at least the sink chunk |
| `--elastic-activation-policy` | `off` | requires `v5` unless `off` |
| `--elastic-prefill-headroom-mib` | `0` | nonnegative integer |
| `--elastic-future-policy` | `off` | `deadline`; native `kv_hints` or the built-in 0.5.2 compatibility path |
| `--elastic-stats-path` | unset | atomic JSON telemetry destination |

The CLI currently launches the guarded dense-MHA scope documented in
[integration](INTEGRATION.md). Unsupported SGLang configurations fail at
startup rather than silently running with an incorrect backing layout.

Set `EKV_FUTURE_STATS_PATH` to atomically export scheduler hook counters from
`adaptive_sparse.sglang_future_hook`; these are separate from the VMM adapter's
`EKV_STATS_PATH` file.

The MiniMax eager scheduler can additionally aggregate known-future waiting
requests into a bounded prefill reservation.  Set
`EKV_FUTURE_PREFILL_MODE=dry_run` to export demand without changing admission,
or `enforce` with a positive `EKV_ADAPTIVE_BUDGET_BYTES` to trigger the existing
prefill-pressure migration path. `EKV_FUTURE_PREFILL_HORIZON_MS`,
`EKV_FUTURE_PREFILL_MAX_BYTES`,
`EKV_FUTURE_PREFILL_ACTIVATION_BYTES_PER_TOKEN`,
`EKV_FUTURE_PREFILL_KIND_ACTIVATION_JSON`, and
`EKV_FUTURE_PREFILL_AUTO_RESIDENT_TOKENS` bound the estimate.  This reserves
headroom for native allocation; it does not create future KV or activation
buffers eagerly. See [the future planner design](FUTURE_PLANNER_DESIGN.md#53-已知未来请求的-prefill-预留).

Set `EKV_FUTURE_ORDERING_HORIZON_MS` to make deadline reordering fall back to
the native queue when any request is still farther than that horizon from its
release time. Deadline hints without `expected_output_tokens` also preserve the
native order instead of using an optimistic one-token decode estimate.
