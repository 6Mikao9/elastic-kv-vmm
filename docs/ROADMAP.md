# Research scope and order

## Current delivery

Improve and validate the elastic physical-backing prototype before changing
the attention addressing model:

1. Reduce allocator overhead and document memory/latency tradeoffs.
2. Support CUDA Graph capture and replay, including safe padding lanes and
   physical reclaim/recommit between requests.
3. Evaluate application value with an idle co-tenant and bursty long-context
   demand, including a manually tuned static-pool control.
4. Coordinate prefill activation workspace with reclaimable KV backing and
   PyTorch cached memory. Separate safe reclamation from unimplemented
   admission guarantees; never reclaim live KV to manufacture headroom.
5. Add a vLLM implementation using existing local model snapshots read-only,
   and validate correctness, memory and serving efficiency against vLLM's
   own baseline.
6. Publish tested code, reproduction commands, raw measurements and limitations.

## Deferred research

After the current delivery is tested, documented and pushed:

- Investigate attention addressing without PagedAttention's software KV
  block table, using stable per-request virtual ranges. CUDA/hardware
  mapping granularity remains; this is not the removal of hardware paging.
- Decouple RadixCache's logical token-prefix boundaries from physical
  reclamation granularity. Investigate references, partial-block sharing,
  fork isolation, eviction and Graph lifetimes. Token-level matching by
  itself is already possible with SGLang page size one.
- Target discrete Top-K KV access. Separate software address-translation
  cost from GPU memory transactions, gather/packing, locality and offload.
  Benchmark matched support sets before claiming a sparse-attention gain.
  The existing `sparse_vattention_v1` workspace experiment is a reference
  for stable virtual addresses and capacity management; its dense-prefix
  reads are not evidence of discrete Top-K performance.

The current SGLang and initial vLLM adapters retain their engines' logical
slot/block indexing. They should be described as elastic physical-backing
prototypes, not as completion of the deferred software-page-table removal.
