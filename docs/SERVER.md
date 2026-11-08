# Elastic SGLang server

`elastic-sglang` is the supported launch entry point. It validates the
experimental memory policy before importing SGLang or touching CUDA, then
installs the pinned SGLang 0.5.2 adapter and passes all remaining arguments to
SGLang unchanged.

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
| `--elastic-stats-path` | unset | atomic JSON telemetry destination |

The CLI currently launches the guarded dense-MHA scope documented in
[integration](INTEGRATION.md). Unsupported SGLang configurations fail at
startup rather than silently running with an incorrect backing layout.
