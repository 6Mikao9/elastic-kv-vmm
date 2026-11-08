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
  --elastic-prefill-headroom-scoped \
  --elastic-prefill-workspace-bytes-per-token 1536 \
  --elastic-prefill-workspace-max-mib 1024 \
  --elastic-global-broker-path /run/ekv/gpu0.sqlite \
  --elastic-global-broker-mib 6144 \
  --elastic-global-broker-prefill-mode observe \
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
| `--elastic-prefill-headroom-scoped` | `false` | apply fallback headroom only during prefill scheduling |
| `--elastic-prefill-workspace-bytes-per-token` | `0` | requires V5 physical admission |
| `--elastic-prefill-workspace-max-mib` | `256` | positive bound on the waiting-prompt estimate |
| `--elastic-precommit-chunks` | `0` | bounded chunk lookahead from 0 to 4 |
| `--elastic-precommit-max-chunks` | `1` | must cover the lookahead |
| `--elastic-future-policy` | `off` | `deadline`; native `kv_hints` or the built-in 0.5.2 compatibility path |
| `--elastic-future-ordering-horizon-ms` | `0` | nonnegative; zero keeps native horizon behavior |
| `--elastic-stats-path` | unset | atomic JSON telemetry destination |
| `--elastic-future-stats-path` | unset | separate future-hook telemetry destination |
| `--elastic-global-broker-path` | unset | shared SQLite state for one physical GPU |
| `--elastic-global-broker-mib` | `0` | aggregate VMM backing capacity; must be positive with a path |
| `--elastic-global-broker-stale-seconds` | `120` | minimum age before a demonstrably dead owner's leases are recovered |
| `--elastic-global-broker-fair-mode` | `off` | `off`, `observe`, or cooperative `enforce` weighted fairness |
| `--elastic-global-broker-tenant` | unset | stable fairness identity; required by observe/enforce |
| `--elastic-global-broker-tenant-weight` | `1` | positive weight shared by every owner of the tenant |
| `--elastic-global-broker-demand-ttl-seconds` | `10` | lifetime of a denied request's starvation signal |
| `--elastic-global-broker-wait-ms` | `0` | bounded lease retry from 0 to 5000 ms |
| `--elastic-global-broker-prefill-mode` | `off` | `off`, `observe`, or `enforce`; requires V5 physical admission, broker capacity, and a positive workspace estimate |

The CLI currently launches the guarded dense-MHA scope documented in
[integration](INTEGRATION.md). Unsupported SGLang configurations fail at
startup rather than silently running with an incorrect backing layout.

With `--elastic-admission-policy physical`, V5's allocator reports a
conservative physical token budget through SGLang's native `available_size()`
call. The budget charges the configured prefill headroom and the bytes needed
to commit a new logical KV chunk before SGLang builds an extend/decode batch.
Already mapped free slots and precommitted chunks consume no new backing and
remain schedulable even when free device memory is tight. Use
`--elastic-activation-policy both` to trim unreferenced VMM chunks and reclaim
PyTorch cache under the same pressure gate. This reduces admission before VMM
commit. When a global broker is configured, the same gate also observes
broker-wide remaining bytes.

For multiple SGLang processes on one GPU, launch every broker-aware process with
the same `--elastic-global-broker-path` and the same aggregate
`--elastic-global-broker-mib`. For multi-GPU launches, use a directory path or
a `{gpu}` template such as `/run/ekv/gpu-{gpu}.sqlite`. The SQLite transaction is
the cross-process admission point, and a lease is kept until the corresponding
chunk is unambiguously unmapped. This coordinates VMM backing only. Weights,
PyTorch caches and activations are outside the mapped capacity. The default remains
first-come. `observe` reports weighted shares and would-yield decisions;
`enforce` adds cooperative soft fairness, and another process's live KV is never
forcibly preempted.
Stale leases are removed only when the owner process is provably gone or the
machine boot ID changed; failed mapping or teardown therefore fails safe by
retaining capacity until a later successful close or crash recovery.

The dynamic path is opt-in. At `Scheduler.get_new_batch_prefill`, the adapter
sums remaining prompt tokens in the waiting queue and the chunked request,
multiplies them by `--elastic-prefill-workspace-bytes-per-token`, and caps the
result with `--elastic-prefill-workspace-max-mib`. That estimate and the static
fallback are combined with `max`, because both model the same transient
workspace. `--elastic-prefill-headroom-scoped` removes the fallback from decode
admission and applies it only while native prefill admission is running. The
rate is still a calibrated model-specific estimate; it does not own or resize
PyTorch's internal activation allocations. `--elastic-global-broker-prefill-mode
observe` reports whether the estimate would fit the shared broker without
inserting a claim. `enforce` inserts one temporary claim before
`get_new_batch_prefill`, then atomically shrinks that claim to the selected
batch's actual token estimate before the forward begins. It is released only
after the corresponding `ModelRunner.forward` succeeds. A denied claim returns
no prefill batch. A
scheduler or forward exception deliberately retains the claim until process or
backend teardown because activation and in-flight access quiescence cannot be
proved. This mode coordinates admission against concurrent VMM growth; it does
not make the activation allocation itself VMM-backed.

Set `--elastic-future-stats-path` (or `EKV_FUTURE_STATS_PATH`) to atomically
export scheduler hook counters from `adaptive_sparse.sglang_future_hook`; these
are separate from the VMM adapter's `--elastic-stats-path` file. The launcher
also exposes the existing bounded VMM precommit controls, so spawned workers
receive the exact same validated settings.

The MiniMax eager scheduler can additionally aggregate known-future waiting
requests into a bounded prefill reservation.  Set
`EKV_FUTURE_PREFILL_MODE=dry_run` to export demand without changing admission,
or `enforce` with a positive `EKV_ADAPTIVE_BUDGET_BYTES` to trigger the existing
pressure migration path. Prefill refreshes the plan on every boundary; decode
refreshes it every `EKV_ADAPTIVE_OBSERVE_STEPS` so an imminent prefill can
reclaim backing before the physical high-water mark is crossed. Fixed prefill
scratch is applied only to prefill pressure. `EKV_FUTURE_PREFILL_HORIZON_MS`,
`EKV_FUTURE_PREFILL_MAX_BYTES`,
`EKV_FUTURE_PREFILL_ACTIVATION_BYTES_PER_TOKEN`,
`EKV_FUTURE_PREFILL_KIND_ACTIVATION_JSON`, and
`EKV_FUTURE_PREFILL_AUTO_RESIDENT_TOKENS` bound the estimate.  This reserves
headroom for native allocation; it does not create future KV or activation
buffers eagerly. See [the future planner design](FUTURE_PLANNER_DESIGN.md#53-已知未来请求的-prefill-预留).

For active MiniMax KV demotion, `EKV_ADAPTIVE_DEADLINE_GUARD_MS` (default
`500`) protects requests whose hinted deadline is imminent. Within the same
guard tier, demotion candidate efficiency is adjusted by hint weight,
remaining decode work and shared-prefix reuse potential. Missing hints remain
neutral. The option changes only candidate ordering after the existing
adaptive budget and migration safety checks have already triggered.

To make MiniMax request residency react to shared-GPU VMM pressure, set both
`EKV_ADAPTIVE_GLOBAL_BROKER_HEADROOM_BYTES` and
`EKV_ADAPTIVE_GLOBAL_BROKER_LOW_BYTES`. The first is the aggregate free-byte
headroom that triggers the normal demotion path; the second is the higher
free-byte threshold that must remain stable before promotion is allowed. The
low watermark must exceed the trigger. Every TP rank must use a broker-enabled
backing, and all participants must reach the scheduler hook. This is a
cooperative multi-instance policy over VMM KV leases; it does not budget model
weights or activation allocators and does not force eviction inside an
uncooperative process.

For a fair two-model deployment on one GPU, give every process of each model
the same explicit tenant and weight. The broker records denied demand for a
short TTL; an over-share model is then denied further growth while the starved
model exists. Its local scheduler must notice `fairness_pressure` and demote
safe requests before retrying. Fairness is active only with `enforce`; use
`observe` for a no-effect measurement arm. The state file binds the mode and
weights, so do not change them while live owners exist.

Set `--elastic-future-ordering-horizon-ms` (or `EKV_FUTURE_ORDERING_HORIZON_MS`)
to make deadline reordering fall back to
the native queue when any request is still farther than that horizon from its
release time. Deadline hints without `expected_output_tokens` also preserve the
native order instead of using an optimistic one-token decode estimate.
