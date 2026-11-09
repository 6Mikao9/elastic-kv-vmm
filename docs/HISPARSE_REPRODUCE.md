# Reproduce the MiniMax HiSparse experiments

Run from the project root with four idle A100 GPUs, Docker with GPU support,
and a local MiniMax-M3.1-Flash checkpoint. The wrapper checks the exact
image ID, checks that the selected GPUs are idle, mounts the model
read-only and runs without network access. It starts an isolated server
with the project adapters. The original checkpoint and installed SGLang
source are not edited. All arms use the same SM80 projection compatibility
path, which converts the serialized MXFP8 projections to BF16 at load and
keeps the NVFP4 Marlin experts.
[Container wrapper](../scripts/container_hisparse_smoke.sh) ·
[SM80 adapter](../adaptive_sparse/minimax_sm80.py)

The runner starts through the shared `elastic-sglang` entry point with
`--elastic-runtime-profile minimax-hisparse`. This keeps dense V5 disabled and
installs only the enabled MiniMax VMM/residency/scheduler hooks, including in
spawned SGLang workers. The legacy SM80 launcher remains as a compatibility
wrapper for older manifests.

```bash
IMAGE=lmsysorg/sglang:dev-minimax-m31-flash
# Recorded local image ID used by the wrapper. The registry RepoDigest was not
# recorded and must not be inferred from this image ID.
test "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = \
  sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87

# Build inside an environment with the CUDA headers and libcuda.
g++ -std=c++17 -O2 -Wall -Wextra -shared -fPIC \
  -I/usr/local/cuda/include native/elastic_vmm.cpp -lcuda \
  -o native/libelastic_vmm.so

GPUS=0,1,2,3  # Replace with four idle device indices.
MODEL=/absolute/path/to/MiniMax-M3.1-Flash
OUT=results/hisparse-reproduce
RUN=scripts/container_hisparse_smoke.sh
COMMON=(--sm80-bf16-projections --logical-capacity 65536 --host-ratio 2)
BENCH=(--benchmark-repeats 3 --warmup-once --semantic-probes)
MIXED=(--vmm-backing --vmm-fast-decode --vmm-idle-grace-ms 200
       --resident-limit 2 --resident-resolver triton --resident-full-batch-swap)
```

The native and mixed arms use 32,768 physical slots and a 65,536-slot
logical capacity. With HiSparse disabled, the full-resident arm uses
65,536 physical and logical slots. The runner verifies the actual
startup capacity. Use a fresh output directory for each arm and keep
compiler-cache treatment consistent when comparing measured phases.
[Runner](../scripts/run_hisparse_smoke.py)

## Native, static mixed and full-resident controls

Each invocation runs the five standard workloads with three repeats,
a single workload warmup round and semantic probes. The fixed hot-buffer
setting and workload definitions come from the pinned runner and client.
[Workloads](../benchmarks/hisparse_client.py)

```bash
WORKLOADS=(short_serial short_concurrent long_serial long_concurrent mixed)

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/native" \
  "${COMMON[@]}" "${BENCH[@]}" --workloads "${WORKLOADS[@]}"

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/static" \
  "${COMMON[@]}" "${BENCH[@]}" "${MIXED[@]}" --resident-free-reserve 8192 \
  --workloads "${WORKLOADS[@]}"

bash "$RUN" "$GPUS" "$MODEL" resident "$OUT/resident" \
  "${COMMON[@]}" "${BENCH[@]}" --workloads "${WORKLOADS[@]}"
```

## Pressure comparison

Use `mixed12k_unequal`, which gives the two long requests different
decode lengths. The watermark and migration cap values below are MiB
per rank. Both adaptive arms use identical settings except for the maximum
number of requests in a demotion action. For a matched static pressure
control, use the same workload, backing, reserve and warmup, and omit the
adaptive flags and chunk-aware retention.

```bash
PRESSURE=(--resident-free-reserve 4096 --adaptive-budget-mib 800
          --adaptive-low-mib 720 --adaptive-observe-steps 1
          --adaptive-dwell-seconds .1 --adaptive-low-stable-seconds .2
          --adaptive-migration-cap-mib 1024 --chunk-aware-retention
          --adaptive-trim-low-water)

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/pressure-native" \
  "${COMMON[@]}" "${BENCH[@]}" --workloads mixed12k_unequal

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/pressure-static" \
  "${COMMON[@]}" "${BENCH[@]}" "${MIXED[@]}" --resident-free-reserve 4096 \
  --workloads mixed12k_unequal

for group in 1 2; do
  bash "$RUN" "$GPUS" "$MODEL" native "$OUT/pressure-group-$group" \
    "${COMMON[@]}" "${BENCH[@]}" "${MIXED[@]}" "${PRESSURE[@]}" \
    --adaptive-max-group "$group" --workloads mixed12k_unequal
done

python3 scripts/summarize_hisparse.py --format markdown \
  "$OUT/native" "$OUT/static" "$OUT/resident" \
  "$OUT/pressure-native" "$OUT/pressure-static" \
  "$OUT/pressure-group-1" "$OUT/pressure-group-2"
```

## Future ordering with adaptive residency

The combined integration matrix adds `--future-policy deadline`. In this
path, every benchmark request carries an `elastic.future` ordering hint while
VMM, native HiSparse and request residency remain active. The ordering hook
runs after SGLang's native priority policy; it does not replace admission or
memory checks. Use the same pressure settings and a fresh directory for every
run:

```bash
MIXED12K=(--workloads mixed12k_unequal)
BENCH1=(--benchmark-repeats 1 --warmup-once --semantic-probes)

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/future-off" \
  "${COMMON[@]}" "${BENCH1[@]}" "${MIXED[@]}" --future-policy off \
  "${MIXED12K[@]}"

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/future-deadline" \
  "${COMMON[@]}" "${BENCH1[@]}" "${MIXED[@]}" --future-policy deadline \
  "${MIXED12K[@]}"

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/adaptive-off" \
  "${COMMON[@]}" "${BENCH1[@]}" "${MIXED[@]}" "${PRESSURE[@]}" \
  --future-policy off "${MIXED12K[@]}"

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/adaptive-deadline" \
  "${COMMON[@]}" "${BENCH1[@]}" "${MIXED[@]}" "${PRESSURE[@]}" \
  --future-policy deadline "${MIXED12K[@]}"

# Optional request-level residency policy. With the 128-token threshold, the
# 12K/128 long request may claim a resident slot while the 12K/64 long request
# stays on the native HiSparse hot path. The policy must be enabled together
# with both future hints and request residency.
bash "$RUN" "$GPUS" "$MODEL" native "$OUT/adaptive-deadline-residency" \
  "${COMMON[@]}" "${BENCH1[@]}" "${MIXED[@]}" "${PRESSURE[@]}" \
  --future-policy deadline --resident-hint-policy adaptive \
  --resident-min-hint-output-tokens 128 "${MIXED12K[@]}"

# Optional future-growth admission accounting. Run dry_run first and compare
# its would-decline reasons with the actual `insufficient-physical-reserve`
# fallbacks before allowing the broker to change placement.
bash "$RUN" "$GPUS" "$MODEL" native "$OUT/adaptive-broker-dry-run" \
  "${COMMON[@]}" "${BENCH1[@]}" "${MIXED[@]}" "${PRESSURE[@]}" \
  --future-policy deadline --resident-hint-policy adaptive \
  --resident-min-hint-output-tokens 128 --resident-broker dry_run \
  "${MIXED12K[@]}"

bash "$RUN" "$GPUS" "$MODEL" native "$OUT/adaptive-broker-enforce" \
  "${COMMON[@]}" "${BENCH1[@]}" "${MIXED[@]}" "${PRESSURE[@]}" \
  --future-policy deadline --resident-hint-policy adaptive \
  --resident-min-hint-output-tokens 128 --resident-broker enforce \
  "${MIXED12K[@]}"
```

Before treating an arm as performance evidence, require a complete outer
`results.json`, nonzero future telemetry when enabled, zero fallback exception,
invalid-hint and telemetry-error counters, matching source manifests, and a
verified idle GPU window. Sequential order and warm state can dominate small
differences; do not establish a speed claim from a single off/deadline pair.
For the residency-policy arm also require nonzero `resident_layer_rows`, inspect
the per-rank `future_decision_reasons`, and verify that the configured threshold
rejected the intended short decoded requests before interpreting timing.
For broker arms additionally inspect `request_residency.resident_broker`,
confirm that dry-run assessments are present, and verify in enforce mode that
every broker decline fell back to native HiSparse without changing correctness.

The summarizer requires a completed outer `results.json`, the matching
`source_manifest.json`, consistent runtime capacity and timing without
intrusive diagnostics. Plotting also checks common inference arguments,
the SM80 compatibility source hash, HiSparse settings when both arms use
them, and measured prompt identities and shapes. It permits the different
physical pool sizes used to match logical capacity with HiSparse disabled.
Keep the same checkpoint and pinned image across arms; these checks do not
replace that experimental control.

## Correctness run

Run the shadow separately from performance measurement. The automatic
round-trip configuration uses the same high and low watermarks but no
forced steps. Confirm the run records real demotions and a promotion
with reason `low_water_stable`, and that every TP rank reports nonzero
checked bytes with zero invalid slots and mismatches.

```bash
bash "$RUN" "$GPUS" "$MODEL" native "$OUT/auto-roundtrip-shadow" \
  "${COMMON[@]}" "${MIXED[@]}" "${PRESSURE[@]}" --adaptive-max-group 2 \
  --adaptive-plan-capture --logical-shadow --validation-rounds 2 \
  --validation-long-tokens 12288 --validation-new-tokens 64 \
  --validation-tail-new-tokens 128
```

The decode oracle checks selected sparse main K/V against the independent
logical-slot values. The migration oracle checks the complete committed
sparse main history at the source or destination boundary. The checks
exclude indexer and dense K/V, prefill attention reads and model-level
numerical determinism. Preserve each arm's source manifest, result files
and source snapshot when attributing evidence to a specific version.
[Validation scope](HISPARSE_VALIDATION.md)
