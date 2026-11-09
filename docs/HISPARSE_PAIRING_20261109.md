# MiniMax HiSparse benchmark pairing audit, 2026-11-09

The historical reference is the completed native-greedy run in [final-native-greedy](../results/final-native-greedy). It contains 15 measured phases and five warmup phases. The old [static attempt](../results/final-static-greedy) stopped before timing. Use the completed reference for the five standard workloads and retain its original artifacts. Use a separate native control for any 12K pressure comparison requiring one. The historical run has no 12K phases.

## Common settings and source checks

- Use the pinned image `sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87`, the same MiniMax-M3.1-Flash checkpoint, TP4, SM80 BF16 projections, BF16 model and KV, Triton attention, Triton multimodal attention, Marlin experts, seed 29, and eager inference with overlap and radix caching disabled. The historical manifest also sets page size 128, context 16,384, prefill chunks of 2,048, four running requests, and static memory fraction 0.92. Check effective capacities in the run logs. Native HiSparse and static use 32,768 physical and 65,536 logical slots with top-k 2,048, device buffer 4,096 and host ratio 2. Full-resident uses 65,536 physical and logical slots with HiSparse disabled.
- The historical snapshot and current Windows workspace have byte-identical SM80 projection, allocator telemetry, semantic probe and container-wrapper source. The projection file's raw SHA-256 is `207df72e8294982f5153a236043871f38a26b37338c543d9c8322fd56d0c2bf0`. Its LF-normalized hash is `314e6f18a73e696b7d35054033486a45baeb350521f08cd01a5c9fcb2e6274ab`. Preserve the audited bytes on the run host so the existing plot guard can compare raw hashes. The current launcher routes through the MiniMax profile and installs the same projection adapter. Its MiniMax branch bypasses the dense adapter. The pinned image identifies the stock kernel implementation. Resident and VMM arms intentionally change the runtime path used to serve requests.
- Pass `short_serial short_concurrent long_serial long_concurrent mixed` explicitly and in that order. The historical client's defaults included only these five workloads. The current client's defaults start with `mixed12k_unequal`. The five common workloads retain their request IDs, prompt generation, input and output lengths and offered concurrency. Independently regenerated SHAKE256 prompt IDs matched all 80 historical warmup and timed prompt hashes. Use three repeats, one warmup pass across all selected workloads, greedy sampling, semantic probes, and future hints off. The HTTP generation and synchronized NVML measurement protocol matches, including 50 ms sampling and 300 ms idle pauses. The shared client gained optional future metadata and first-token callbacks, which the core benchmark leaves inactive.
- Current static and pressure arms deliberately test the current residency, VMM and migration code. Even with future hints, shared-prefix reuse, and both brokers off, residency adds a CPU TP intent collective before staging and decision records, plus a CPU row-list conversion on resident decode batches. VMM adds ownership telemetry at idle and uses updated backing and map handling. These are part of the current treatment. Pin these sources across the new pressure arms. Record the run image, source hashes, effective capacities, compiler-cache treatment and rank-to-GPU topology. The historical run used GPUs 0,2,3,4 and the planned current runs use 0,1,3,4. Treat differences against the historical native run as workload-matched cross-run comparisons, with the GPU set and run window as limitations.

## Summary and plot guards

[The summary script](../scripts/summarize_hisparse.py) validates completed outer and performance results, capacities, inference options, matching declared warmup and replay settings, full per-arm phase coverage, request counts and NVML samples. Its CLI groups arms by sampling mode and accepts separate workload sets. [The plot script](../scripts/plot_hisparse.py) additionally requires matched inference signatures, the same raw SM80 source hash, matching phase sets and concurrency, and equal request IDs, prompt hashes and token counts per phase. Plot the five-workload and 12K-only cohorts separately. A passing plot guard verifies workload pairing and the recorded inference signature. It does not verify GPU topology, identical timing windows, or all source hashes.

## Independent acceptance before reporting

The runner can finish while recording `semantic_smoke_passed: false`. A complete
outer result alone is therefore insufficient. Run the independent
[artifact audit](../scripts/audit_hisparse_results.py) before summarizing a formal
cohort. Its default is three repeats; use separate invocations for the five
standard workloads and the 12K-only controls.

```bash
python -B scripts/audit_hisparse_results.py \
  results/final-native-greedy results/matrix-20261109-r6/static \
  results/matrix-20261109-r6/resident

python -B scripts/audit_hisparse_results.py \
  results/matrix-20261109-r6/pressure-native \
  results/matrix-20261109-r6/pressure-static \
  results/matrix-20261109-r6/pressure-group-1 \
  results/matrix-20261109-r6/pressure-group-2
```

The audit requires four distinct passing semantic cases, the three complete
token-ID probes with recomputed hashes, one warmup per workload, and complete
measured phases. It checks warmup, measured and correctness-probe input identities
across arms; it does not require identical generated tokens. It also checks
request latency arithmetic, NVML sample ordering and window duration, and
recomputes both per-rank peaks and the synchronized total peak. VMM-enabled arms
must provide all four rank files and idle snapshots rather than silently losing
their backing evidence.

The historical native artifacts pass this audit. Its report explicitly leaves
model/image/source provenance and external GPU runtime evidence unverified.
Review those against the frozen source, container identity, controller and
per-arm GPU observations. Token hashes validate stored output records, not KV
byte correctness; the separate migration/decode shadow remains necessary.
For pressure treatments, check that the intended demotions and physical releases
actually happened. A treatment with zero actions is not evidence for migration
performance. VMM lifetime backing peaks include probes and warmup, whereas the
NVML phase peaks use only each measured active window.
