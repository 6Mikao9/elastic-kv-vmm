# Request-level VMM and HiSparse integration plan

The initial hook audit used local SGLang revision `9f6678a`. The
implemented MiniMax path runs in the separately pinned image
`lmsysorg/sglang:dev-minimax-m31-flash`, image ID
`sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87`.
The image has no Git metadata. Use its audited source hashes and each
experiment's manifest to identify the runtime. The earlier SGLang 0.5.2
dense Qwen setup is separate. See the [architecture](HISPARSE_ARCHITECTURE.md)
and [reproduction commands](HISPARSE_REPRODUCE.md).

## Validated runtime and scope

The MiniMax-M3.1-Flash snapshot has 48 complete shards (232.94 GiB),
MXFP8 projections and NVFP4 experts. Stock-image TP4 startup on A100
failed at the SM90 quantization gate. The opt-in SM80 adapter preserves
the serialized projection loading layout, converts the projection
weights to BF16 and disables native MXFP8 QKV/index fusion for those
layers. It retains NVFP4 Marlin experts. The adapter passed synthetic
and actual checkpoint projection checks and loaded the complete model.
All compared SM80 arms use the same adapter.

HTTP pilots and controlled TP4 migration runs have executed on this
runtime. The automatic round trip passed exact checks of the committed
migration history and selected decode K/V. Separately, identical greedy
probes can diverge in both native HiSparse and full-resident mode under
the SM80 compatibility setup. Traces found differences already present
in selected prefill history. Deterministic flags and serialized CUDA
launches did not remove the discrepancy. This limits cross-run token-ID
parity claims. Byte-level residency checks and matched, uninstrumented
performance measurements answer different questions. See
[validation](HISPARSE_VALIDATION.md) for the observed results and limits.

Capacity matching uses 32,768 physical sparse slots and host ratio 2 for
HiSparse, versus 65,536 slots for full residency. Both have 65,536 logical
slots, four maximum running requests, and 16,384 context length. The runner
asserts the actual profiled pool capacity via `/server_info`; flags alone are
not capacity evidence. Each long host-backed request requires 4,224 physical
slots (4,096 plus a 128-token reserve page), with prefill staging additional.

The [HiSparse guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hisparse_guide.mdx)
documents PD decode. The pinned source also has a local staging path in
`managers/scheduler_components/batch_result_processor.py:395` following
normal prefill. The MiniMax experiments exercise this colocated path.
The documented PD path is a separate runtime configuration.

For the DSA hook points below, candidates include DeepSeek-V3.2 and
GLM-5 family checkpoints with native DSA. Select a checkpoint that fits the
available GPUs with KV, workspace and transfer headroom. The
[DeepSeek-V3.2 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx)
lists model and architecture-specific kernels. Verify the permitted
dtype/backend pair in `arg_groups/hisparse_hook.py` against the selected
image. The original A100 Qwen setup does not validate any of these pairs.
DeepSeek V4 and MiniMax M3 are separate HiSparse implementations with
different cache components and routing. A compact test fixture can
exercise correctness, while a real native sparse checkpoint is required
for capacity or latency claims.

## Current MiniMax implementation and original audit

### MiniMax target

The current image has architecture-specific hooks. Its line numbers differ
from the initial local audit below. MiniMax uses independent K/V storage
and selects 16 blocks of 128 tokens; it is not DeepSeek's MLA cache layout.

* `batch_result_processor.py:396`: preserve the native staging readiness
  queue for resident requests with a CUDA event. Resident requests keep
  their physical prefill mapping and skip host copy and side-buffer
  allocation.
* `schedule_batch.py:3743`: apply newest-token remapping and backup only to
  host-backed rows. Resident rows keep their growing physical mappings.
* `minimax_sparse_backend.py:367`, `_hisparse_swap_in_blocks`: route eager
  rows by residency. The full-batch variant passes zero lengths for
  resident rows to native swap-in, then fills their selected locations
  in the original batch order. The resident resolver checks sequence
  bounds and resolves through the request-token and device-location
  mappings.
* `hisparse_coordinator.py:838`, `admit_request_into_staging`: this is a
  prefill admission routine using `extend_end` and staging ownership, not a
  ready-made mid-decode demotion API. The adapter migrates at a quiescent
  decode boundary, fences copies and publishes the new routing state
  around the source release. Reverse migration restores history from
  host and, when needed, copies the latest token from the GPU reserve.

Every resulting device location must refer to the same attention-visible
K/V planes. Merely enlarging a logical hot-buffer limit does not create
mixed residency or reclaim physical backing. The first implementation is
eager; Graph routing needs fixed-shape device metadata in a later step.

### Original DSA hook audit

The following historical DSA proposal refers to `python/sglang/srt` in the local audit tree. It is not the implemented MiniMax path.
Keep native sparse top-k and logical IDs for both modes. The difference is
physical residency. Start with eager execution, one supported DSA backend,
no overlap, radix cache or speculation. If the checkpoint requires TP,
apply each transition coherently across ranks and budget for the least
available rank.

| Hook | Change |
|---|---|
| `mem_cache/allocator/hisparse.py:160,208,257` | Track actual device allocations, extensions and frees against chunk occupancy. Wrap the compatible KV backing with stable VA. Reclaim only a wholly dead chunk, counting bytes across all planes. |
| `managers/scheduler_components/batch_result_processor.py:395` | Gate the existing staging call behind request residency. Full-resident requests enter decode with their native physical mappings. |
| `managers/schedule_batch.py:3817` | Keep resident append slots on the normal physical mapping. Call the coordinator newest-token mapping and host backup only for host-backed rows. |
| `layers/attention/dsa_backend.py:2335,2624` | Resolve resident top-k selections directly to device locations. Route host-backed rows through native swap-in. Validate selected locations and then scatter them into original batch order, or start with homogeneous decode sub-batches. |
| `managers/hisparse_coordinator.py:838,937,1100,1168,1265,1402` | Reuse staging, hot buffer allocation, newest-token mapping, backup events and cleanup where their assumptions hold. Hold source slots until copy completion and publication of the new routing state. |

The JIT `HOT_BUFFER_SIZE` in
`python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh:418` is a compile-time
parameter. Use a fixed hot pool size for host-backed rows and a separate
resident lookup path. Allow space for the newest-token page, alignment,
index state and any prefetch/copy scratch in the budget.

Use an opt-in adaptive mode so the native HiSparse control remains intact.
Start with pressure-triggered demotion at safe decode boundaries. Reserve
host and hot-buffer space before starting transfer. Commit routing after
the copy is complete, then release only unreferenced backing. Leave the
source usable on failed migration. Favor actual reclaimable chunks and
remaining request lifetime over sequence length alone. Use distinct high
and low watermarks, a cooldown and a byte cap per interval. The implementation stages demotion first, then permits reverse migration
when the low watermark has held for the configured stable interval and
the destination fits while its source remains live.

## Prefill workspace, optional for the first experiment

1. **Telemetry:** bracket prefill or extend, record allocator allocated and
   reserved bytes, allocation peak growth, global free memory and VMM
   backing. Calibrate against prompt length, batch size and chunk size.
2. **Admission:** reserve estimated workspace before admitting a chunk.
   Coordinate existing PyTorch allocations, reserved reclaimable cache,
   native workspaces and Graph pools without double counting. If a grant
   cannot be met, defer work or use a smaller prefill chunk.
3. **Safe live demotion for an activation grant:** complete enough
   migrations before launching prefill and confirm physical reclaim.
   Account for temporary source/destination overlap and in-flight work.
   PyTorch caching blocks and Graph private pools have different lifetimes
   from VMM backing. An allocator flush alone cannot guarantee the grant.
   Keep this outside the initial request-level experiment until admission
   and demotion are stable.

## Validation

Match native HiSparse, full GPU KV with native sparse attention and
adaptive VMM + HiSparse on the same model, backend and workload. Test
resident and host-backed rows in the same traffic episode, forced demotion
mid-generation, physical growth and reclaim, abort and idle cleanup, and
oscillation around both watermarks. Compare deterministic token IDs and
selected KV locations in targeted checks. Report measured TTFT/TPOT,
throughput, physical memory, pinned host memory, migrations, transfer
bytes and miss costs. Current runtime evidence includes forced migration and a controlled
automatic round trip at TP4, with exact logical shadow checks. See
[validation](HISPARSE_VALIDATION.md) for measured coverage and limits.
Repeat the matched performance comparison separately without the shadow.
