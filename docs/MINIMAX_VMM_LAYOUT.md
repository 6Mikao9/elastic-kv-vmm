# MiniMax M3 HiSparse VMM layout audit

Read-only inspection of local image `lmsysorg/sglang:dev-minimax-m31-flash`,
under `/sgl-workspace/sglang/python/sglang`. This is an integration design.

| Source | SHA-256 |
|---|---|
| `srt/mem_cache/memory_pool.py` | `e3d57e41beb4d5f77c2c91e6bf9bb392ef1b5ad9e853c4521b3dfcf1e12a20e1` |
| `srt/mem_cache/hisparse_memory_pool.py` | `74b46cc95e8336407348e7251ad08c730f1ab6a676f9a44632007d9ae3b72480` |
| `srt/mem_cache/allocator/hisparse.py` | `f3f15a599bcb09131606bd5bf93511a4bf39f4acec5835e73975967b5697f732` |
| `srt/mem_cache/minimax_m3_nvfp4_pool.py` | `e83f3ec57d7f8214cd71e47e91a2d590fbd6b8f9a12132f272d37943d236526d` |

## Physical storage

- `MiniMaxSparseKVPool.__init__` hardcodes `HiSparseMHAMainPool` for sparse
  main K/V with HiSparse active. Its `main_pool_cls` argument reaches the
  dense sub-pool only in this configuration.
- The sparse main pool has `size + page_size` token slots per layer. For
  unquantized normal layout, the stock K and V tensors have shape
  `[slots, local_heads, head_dim]` and `[slots, local_heads, v_head_dim]`,
  respectively. Derive plane strides from each actual store dtype and width.
  TP changes local heads and physical bytes per rank.
- Dense layers and the index K-only and K/V pools have logical
  `size * host_to_device_ratio` capacity. Index tensors use one head with
  `idx_head_dim` and may use a distinct dtype. Their direct logical addressing
  serves scoring even when sparse main KV is offloaded. Leave these pools
  resident in the first adapter.
- `HiSparseMHAMainPool.set_kv_buffer` translates incoming logical IDs through
  the shared physical mapping. The wrapper's `set_fused_kv_index_buffer`
  uses separate stores when HiSparse is enabled. Keep the stock dispatch
  and layer IDs.

## First adapter boundary

Replace only `HiSparseMHAMainPool._create_buffers_normal` for validated normal,
unquantized layout. Use `elastic_kv.backing.Backing` and a stable virtual range,
with per-plane chunk-aligned offsets. Keep stock `_create_buffers` to build
descriptors, pointer arrays and strides from the resulting tensor views. Keep
virtual addresses stable through map and unmap. The padded page at zero must
be mapped before any initialization or attention kernel can use it.

Check the actual pool class, store dtype and layout at startup. Quantized
`_create_quantized_buffers` and `MiniMaxM3Nvfp4KVPool` own packed data, scales
and different host copy paths. They need independent layout audits. A raw
contiguous pointer export spans virtual capacity, so transport paths that may
read uncommitted bytes must stay disabled until verified.

## Physical page ownership

`HiSparseTokenToKVPoolAllocator` uses separate paged allocators for logical
IDs and physical slots. Track chunk occupancy on the physical allocator,
including fresh prefill staging slots from `alloc_extend`, device buffers and
reserve pages from `alloc_device_buffer`, and all physical frees. An extend
can reuse a tail page. Count the page at ownership transitions rather than
once per token. Preserve the mapping sentinels, zero for unmapped and trailing
minus one for a missing predecessor. Stock `alloc_decode` reserves logical
IDs only. Resident decode needs a physical page mapping before its store.

`alloc_device_buffer` clears incoming logical mappings, reuses physical prefill
slots, frees surplus, and allocates extra slots if needed. Reused pages stay
owned after mappings are cleared. Count device-buffer ownership independently
of the mapping table and roll back failed transitions before making any release
available to the budget. Project page ranges into each K and V plane, including
chunk crossings and nonidentical widths.

A chunk is reclaimable when no physical page owner covers it and outstanding
attention, staging, backup, and swap-in work that can reference it is complete.
Use an eager, quiescent boundary for initial testing. Track and confirm actual
unique bytes reclaimed per rank. Logical capacity, occupied pages, and
physically backed bytes are separate quantities. Enlarging virtual capacity
also requires explicit admission constraints for dense/index storage and a
coordinated TP budget.
