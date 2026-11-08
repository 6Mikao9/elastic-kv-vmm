# Integration map

The opt-in SGLang adapter installs at process startup and keeps the attention
interface and logical token slot IDs. It supports the measured SGLang 0.5.2
surface and the newer `KVCacheConfigurator` / `mem_cache` surface, provided the
latter exposes the same MHA pool and token allocator contracts. The supported
path remains dense MHA with Triton, page size 1 and TP/PP/DP size 1, with
overlap, radix and hierarchical caching, speculative decoding and
prefill/decode disaggregation disabled.

The newer path replaces the pool and allocator at every import site used by the
configurator, wraps `KVCacheConfigurator.configure()` for the same startup
guards, and handles the allocator API's `None`/list free-group representation.
Its automated checks use representative newer API modules; a real GPU serving
run on that revision remains necessary. SGLang's built-in post-capture sizing
must remain off because it owns a different backing lifecycle.

Non-baseline startup is fail-closed. MiniMax sparse/HiSparse, compressed-SWA
and every other hybrid family are rejected before pool construction; after a
modern `configure()` call the returned token pool and allocator must still be
the elastic wrapper and observed allocator. This prevents a newer SGLang KV
pool family from silently allocating its stock backing while the launcher
reports that elastic mode is active. The pinned MiniMax request-residency path
uses its own audited adapter and does not pass through this generic MHA guard.

| Layer | Source | Integration |
|---|---|---|
| Physical memory | [elastic_vmm.cpp](../native/elastic_vmm.cpp), [backing.py](../elastic_kv/backing.py) | Reserve stable virtual addresses, map and zero newly committed CUDA chunks, then unmap and release unused physical handles. CuPy/DLPack exposes those addresses as PyTorch tensors. |
| Pool construction | [adapter.py](../elastic_kv/adapter.py) | On 0.5.2, hook `ModelRunner.init_memory_pool`; on the newer API, replace the MHA class and allocator at their `mem_cache` import sites and wrap `KVCacheConfigurator.configure()`. Both routes replace `MHATokenToKVPool._create_buffers` to provide demand-backed K/V planes. |
| Slot allocation | [adapter.py](../elastic_kv/adapter.py) | V5 tracks free ID ranges on the host, favors low IDs and admits ranges before writes. Contiguous allocations use slices of a persistent GPU ID tensor. Fragmented allocations return the selected IDs through the existing allocator interface. Physical admission mode bounds `available_size()` before SGLang builds a batch by charging prefill headroom only when the low-ID free prefix first touches an unmapped VMM chunk. |
| Residency | [adapter.py](../elastic_kv/adapter.py) | V5 defers ordinary reclaim until the pool drains, synchronizes before unmapping and can retain a configurable low-ID chunk reserve with `EKV_RETAIN_CHUNKS`. `EKV_PRECOMMIT_CHUNKS` optionally precommits the next logical chunk and moves its zeroing to a private nonblocking CUDA stream; the first real `ensure` remains the readiness barrier. A global `EKV_PRECOMMIT_MAX_CHUNKS` cap bounds speculative physical backing. Logical capacity remains fixed. |
| Startup | [launch.py](../elastic_kv/launch.py), [adapter.py](../elastic_kv/adapter.py) | Install hooks in the server process tree and reject unsupported configurations. |

## Decode CUDA Graph

The measured SGLang 0.5.2 baseline, V4 and V5 paths support decode CUDA Graph in the
guarded configuration. Graphs keep the same virtual K/V addresses through
mapping and reclaim cycles. Slot zero remains backed. Capture starts from
an idle pool, with request indices pointing at an extra request table row
outside the allocator's free list. Triton uses a capture sequence length of
one, so the first entry of that row points to slot zero. For padded replay,
the adapter resets the request-index tail to that row before the upstream
metadata update. The upstream runner resets padded write positions to slot
zero and copies real request indices and write positions into the prefix.
This prevents a smaller batch from inheriting stale indices into a reclaimed
range. Newer SGLang graph runners use slot zero for padded request and write
indices, which is already the permanently backed sink, so the legacy extra-row
hook is skipped. See [graph_support.py](../elastic_kv/graph_support.py).

## Prefill activation pressure

The optional V5 hooks mark `ScheduleBatch.prepare_for_extend` as prefill
context and instrument `ModelRunner.forward_extend` (or the newer unified
`ModelRunner.forward`) to sample CUDA memory
around prefill. Before committing new KV chunks, the adapter compares free
device memory with the pending backing bytes plus a configured prefill
headroom target. `EKV_ACTIVATION_POLICY` selects `off`, `telemetry`,
`trim`, `recycle` or `both`. The headroom target is set with
`EKV_PREFILL_HEADROOM_MIB`. See [activation.py](../elastic_kv/activation.py).

Under prefill pressure, trim can synchronize and release mapped chunks that
contain neither live slots nor slots in the pending allocation. Recycle can
ask PyTorch to release unused caching-allocator memory when the target still
exceeds free memory. A shortfall is recorded and allocation proceeds through
the usual path instead of changing the scheduler contract. The target is a
heuristic in this mode. Physical admission separately exposes a conservative
budget through the native `available_size()` query, allowing `PrefillAdder` to
reduce a batch before VMM commit. Already mapped free ranges and precommitted
chunks remain available under pressure instead of being lost to a coarse
chunks-to-tokens calculation. The prefill headroom remains an estimate, and
transactional allocation failure, exact activation ownership and cross-process
memory grants remain separate work.

## vLLM

The vLLM 0.29.0 adapter selects the legacy GPU model runner in the V1 engine.
Its scope is TP1 eager with prefix caching, async scheduling, speculation
and KV connectors disabled. It requires one homogeneous full-attention cache
group and matching manager and kernel block sizes. vLLM CUDA Graph is outside
this integration. See [vLLM usage](../vllm_integration/README.md).

The worker's `initialize_kv_cache_tensors` path replaces the backing
allocation while reusing vLLM's native `create_kv_cache_views`. Before
`_update_states`, the adapter maps blocks for new or resumed requests,
appended blocks for cached requests and all blocks the worker will zero.
After the last worker request finishes, it synchronizes and releases unused
backing while keeping the sink resident. vLLM retains its software block
table and logical paging. See [vLLM adapter](../vllm_integration/adapter.py)
and [launcher](../vllm_integration/launch.py).

Mixed Full/SWA/Mamba pools, live KV eviction and a global memory broker
remain separate integration targets. Removing attention's software page
lookup requires changes to the kernel and indexing path in addition to VMM.
