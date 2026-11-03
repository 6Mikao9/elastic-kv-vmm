# Integration map

This experiment installs an opt-in adapter into SGLang 0.5.2 at process
startup. It does not edit the installed package or model weights. The
prototype preserves the attention interface and logical token slot IDs.

| Layer | Project file | Change |
|---|---|---|
| Physical backing | `native/elastic_vmm.cpp` | Reserve VA, create/map/access independent physical chunks, zero only newly mapped chunks, unmap and release unused handles |
| Tensor view | `elastic_kv/backing.py` | Expose stable GPU addresses to PyTorch through CuPy/DLPack and track committed chunks |
| Pool construction | `elastic_kv/adapter.py`, `ElasticPool._create_buffers` | Replace fixed physical KV tensors with views into demand-backed virtual ranges |
| Residency | `Residency.admit/release` | Count live slot references per physical chunk; finish mapping before admitting writes; synchronize before reclaim |
| Token allocation | `ObservedAllocator.alloc/free` | Retain the existing slot interface, reuse low IDs, and in V4 return persistent GPU ID slices when contiguous |
| Startup | `elastic_kv/launch.py`, `guarded_init` | Install in spawned workers and reject unsupported execution paths |

The upstream entry points are `ModelRunner.init_memory_pool`,
`MHATokenToKVPool._create_buffers`, and `TokenToKVPoolAllocator`.
V0–V4 vary layout and allocation policy within these hooks. All have the
same configured logical token capacity. Released physical bytes become
available to other CUDA allocations; increasing another server's logical
capacity still requires an admission and capacity policy.

## Extending to a memory management system

1. **Physical budget and admission:** track weights, activation demand,
   committed KV, growth headroom and outstanding grants. Map before the
   scheduler admits work; handle allocation failure transactionally.
2. **Mixed state pools:** give Full KV, SWA and Mamba independent virtual
   ranges, with local slot allocators and one shared byte budget. Preserve
   the Full/SWA ID relationship and make joint allocations atomic. This
   removes neighboring-range collisions but does not itself solve sparse
   live-data fragmentation inside a physical chunk.
3. **Runtime lifetimes:** track retained prefixes, in-flight kernels,
   transfers and Graph replay references. The current eager-only device
   synchronization is a correctness boundary, not the intended final
   performance design.
4. **Cross-instance coordination:** reclaim only unused or explicitly
   evictable backing, acknowledge release before granting growth to a
   peer, and define fairness, minimum capacity and process-failure cleanup.
   A global broker and live-KV preemption are absent from this experiment.
5. **Prefill headroom:** estimate the next prefill demand and coordinate
   reclaim with PyTorch's caching allocator and Graph pools. Compare with
   a tuned chunked-prefill baseline. Activation-budget lending is not
   implemented here.

The newer UnifiedKVPool Mamba/SWA/Full topology is a separate integration
target. The measured dense-MHA adapter does not validate that topology.
Removing attention's software page-table lookup is also a separate kernel
and indexing change; CUDA VMM alone does not do it.
