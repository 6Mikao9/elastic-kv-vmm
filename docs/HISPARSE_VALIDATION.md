# MiniMax HiSparse validation on A100

This record separates selected KV transport, numerical localization, and
resident lookup timing. The model runs used the MiniMax M3.1 Flash checkpoint,
the SM80 BF16 projection compatibility path, TP4, and eager execution.
Run metadata and source hashes are in each result directory's
`source_manifest.json`.

## Native HiSparse logical shadow

`results/native-logical-shadow/results.json` reports a completed native
HiSparse run with VMM disabled. All four TP ranks published at scheduler idle.

| Metric | Per rank | Four-rank sum |
|---|---:|---:|
| Checked sparse attention calls | 2,220 | 8,880 |
| Checked selected token instances | 1,948,740 | 7,794,960 |
| Checked K/V bytes | 997,754,880 | 3,991,019,520 |
| Uninitialized logical tokens | 0 | 0 |
| Invalid physical tokens | 0 | 0 |
| Bitwise K/V mismatches | 0 | 0 |
| Malformed Top-K blocks | 0 | 0 |

A token instance may appear in multiple calls or layers. The byte sum is
validation traffic across ranks. Each rank's logical shadow has 65,664 slots
including padding. The server reported 32,768 physical slots and a 2:1
host-to-device capacity ratio.

The shadow records unit-scale BF16 main K/V at the producer's logical slot.
It compares those values to the explicit physical slots selected for sparse
decode, independently of physical remapping and host transport. The oracle
shares the request-to-token table and Top-K selection with the runtime. It
cannot independently validate either, the producer's numerical accuracy,
unselected cache contents, or complete model output.

## Mixed VMM HiSparse logical shadow

`results/vmm-mixed-shadow/results.json` completed at TP4 with VMM backing
and resident limit 1. It ran two four-request mixed rounds, each with two
128-token and two 8,192-token prompts generating 16 tokens per request.
Every rank recorded 15 mixed decode batches, 30 resident decode rows and
67 native decode rows over the run.

| Metric | Per rank | Four-rank sum |
|---|---:|---:|
| Checked sparse attention calls | 4,140 | 16,560 |
| Checked selected token instances | 9,379,140 | 37,516,560 |
| Checked K/V bytes | 4,802,119,680 | 19,208,478,720 |
| Uninitialized logical tokens | 0 | 0 |
| Invalid physical tokens | 0 | 0 |
| Bitwise K/V mismatches | 0 | 0 |
| Malformed Top-K blocks | 0 | 0 |

Each rank's VMM stats show 720 MiB peak backed, 240 MiB idle backed,
1,200 MiB reserved, zero idle owned pages and zero idle reclaimable bytes.
Each recorded four ensure calls and two reclaim calls, mapping 480 chunks
and reclaiming 360. These are sparse main K/V backing statistics. They do
not include weights, dense/index pools or total device memory. This test
validates the exercised static mixed routes under the same oracle scope.

## Forced VMM migration round trip

`results/vmm-roundtrip-shadow/results.json` completed at TP4 with one
forced demotion and one forced promotion per rank. Each rank released
and later mapped 251,658,240 bytes (240 MiB), transferring 503,347,200
bytes over the two actions. The cumulative migration time ranged from
0.13144 to 0.13190 seconds across ranks. These are diagnostic measurements
for the forced sequence.

Each rank checked 3,180 sparse attention calls, 5,663,940 selected token
instances, and 2,899,937,280 K/V bytes. All ranks recorded zero uninitialized
logical tokens, invalid physical tokens, bitwise K/V mismatches, and
malformed Top-K blocks. This verifies selected K/V across the exercised
round trip. The following automatic test checks the pressure and
low-water control path separately from model output quality.

## Automatic pressure and low-water round trip

The [automatic round-trip run](../results/vmm-auto-roundtrip-shadow/results.json)
completed at TP4. Each rank recorded three demoted requests across two
pressure actions, one of them a two-request group. Each rank also recorded
one automatic promotion with reason `low_water_stable`. The low-water
controller reclaimed unused backing before evaluating promotion.

| Measurement | Per TP rank |
|---|---:|
| Demotion release | 480 MiB |
| Promotion growth | 480 MiB |
| Separate low-water slack trim | 480 MiB |
| Peak / idle VMM sparse main backing | 960 / 240 MiB |
| Full migration source/destination K/V checked | 1,511,946,240 bytes |
| Selected sparse decode K/V checked | 25,272,944,640 bytes |

All four ranks reported zero shadow mismatches, zero invalid migration or
selected physical positions, zero uninitialized selected logical positions
and zero malformed Top-K blocks. The migration count includes checks
across layers and transitions. The byte counts refer to checked K/V,
rather than total device memory. This instrumented controlled episode
validates the automatic policy and data movement on the tested path.
Performance, output determinism and generalization to other pressure
patterns remain separate questions. See the
[source manifest](../results/vmm-auto-roundtrip-shadow/source_manifest.json).

## Numerical localization

The same-prompt repeats already diverge without VMM. Each row lists the
first differing output token against the first of four captures, with
zero-based output positions.

| Run | 128 input tokens | 8,192 input tokens |
|---|---|---|
| [native-repeat](../results/numerical-diagnostics/native-repeat/determinism.json), native HiSparse | 2, 2, 2 | 2, 2, 2 |
| [resident-repeat](../results/numerical-diagnostics/resident-repeat/determinism.json), HiSparse disabled | 2, 4, 3 | 1, 1, 1 |
| [resident-layer-trace-r3](../results/numerical-diagnostics/resident-layer-trace-r3/determinism.json) | 0, 8, 0 | 4, 2, 2 |
| [resident-component-trace](../results/numerical-diagnostics/resident-component-trace/determinism.json) | 14, 13, 0 | 1, 2, 1 |
| [resident-attention-capture](../results/numerical-diagnostics/resident-attention-capture/determinism.json) | 14, 14, 14 | 3, 2, 2 |

The linked `determinism.json` files provide the token repeat evidence.
Each run directory includes its source manifest with machine-local paths
redacted and the original manifest digest preserved. The layer, component
and attention observations below were summarized from internal diagnostic
captures. The public repeat records cannot reproduce those scalar
observations without regenerating the captures. The layer trace
localized matching-input decode differences to the local contribution at
zero-based layer 4, with a comparison first differing at layer 5. The
component trace at sequence length 8,193 found variation in layer 4
attention output in all 12 matching comparisons before the MoE inputs and
routing diverged.

The attention capture contained 80 records across four ranks, layers 4
and 5, and sequence lengths 129 and 8,193. Each group had five observations
and ten pairs. Top-K sets and their order agreed in every group. At length
8,193, all layer 4 pairs had bitwise-identical Q and current-token K/V.
Historical selected prefill K/V already differed, with maximum relative
L2 differences of 1.39e-4 for K and 1.11e-4 for V across the four ranks.
Comparison uses valid logical positions and excludes padded rows.

| TP rank | Layer 4 inner output max relative L2 at length 8,193 | Max BF16 ULP |
|---:|---:|---:|
| 0 | 3.55e-4 | 69 |
| 1 | 4.31e-4 | 57 |
| 2 | 3.19e-4 | 64 |
| 3 | 2.47e-4 | 2 |

The capture found no pair with identical captured Q, selected K/V, Top-K,
sink and scales but different inner attention output. This locates an
upstream difference already present in prefill history. The first source
of that variation and its contribution to later token divergence remain
open. The instrumented captures alter scheduling and memory traffic.

## Resident lookup microbenchmark

`results/lookup-micro/results.json` compares the existing Torch indexing
sequence and one fused Triton lookup on A100-SXM4-80GB, sequence length
8,192 and 16 selected 128-token blocks. Both paths matched an independent
reference for every batch size.

| Batch | Torch p50 / p95 (us) | Triton p50 / p95 (us) | p50 ratio |
|---:|---:|---:|---:|
| 1 | 213.80 / 323.60 | 30.54 / 47.66 | 7.00x |
| 2 | 211.06 / 212.25 | 30.00 / 30.62 | 7.03x |
| 4 | 211.88 / 213.77 | 30.08 / 30.61 | 7.04x |

Each path ran 100 warmup calls, then 40 interleaved rounds of 100 calls.
Percentiles describe round-average CUDA event time per call. The intervals
include launch pacing and temporary allocation effects. Other GPUs on
the same host were running correctness workloads, so shared CPU load may
affect pacing. These measurements characterize the isolated lookup path.
Pure kernel time and end-to-end serving performance require separate
measurements.
