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
  --elastic-admission-policy physical \
  --elastic-prefill-headroom-mib 1024 \
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
| `--elastic-admission-policy` | `logical` | `physical` requires `v5` |
| `--elastic-prefill-headroom-mib` | `0` | nonnegative integer |
| `--elastic-precommit-chunks` | `0` | bounded chunk lookahead from 0 to 4 |
| `--elastic-precommit-max-chunks` | `1` | must cover the lookahead |
| `--elastic-future-policy` | `off` | `deadline`; native `kv_hints` or the built-in 0.5.2 compatibility path |
| `--elastic-future-ordering-horizon-ms` | `0` | nonnegative; zero keeps native horizon behavior |
| `--elastic-stats-path` | unset | atomic JSON telemetry destination |
| `--elastic-future-stats-path` | unset | separate future-hook telemetry destination |

The CLI currently launches the guarded dense-MHA scope documented in
[integration](INTEGRATION.md). Unsupported SGLang configurations fail at
startup rather than silently running with an incorrect backing layout.

With `--elastic-admission-policy physical`, V5's allocator reports a
conservative physical token budget through SGLang's native `available_size()`
call. The budget charges the configured prefill headroom and the bytes needed
to commit a new logical KV chunk before SGLang builds an extend/decode batch.
It also counts only precommitted and explicitly retained free chunks as already
backed. Use `--elastic-activation-policy both` to trim unreferenced VMM chunks
and reclaim PyTorch cache under the same pressure gate. This reduces admission
before VMM commit; it is not yet a cross-process reservation or a guarantee
against a concurrent CUDA allocator race.

Set `--elastic-future-stats-path` (or `EKV_FUTURE_STATS_PATH`) to atomically
export scheduler hook counters from `adaptive_sparse.sglang_future_hook`; these
are separate from the VMM adapter's `--elastic-stats-path` file. The launcher
also exposes the existing bounded VMM precommit controls, so spawned workers
receive the exact same validated settings.

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

Set `--elastic-future-ordering-horizon-ms` (or `EKV_FUTURE_ORDERING_HORIZON_MS`)
to make deadline reordering fall back to
the native queue when any request is still farther than that horizon from its
release time. Deadline hints without `expected_output_tokens` also preserve the
native order instead of using an optimistic one-token decode estimate.
