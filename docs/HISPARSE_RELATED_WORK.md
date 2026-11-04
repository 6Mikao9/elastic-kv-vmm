# Related work and claim boundaries

## Addressing and physical memory

- **vLLM extensible KV cache (current main, inspected 2026-11-04)** already
  uses VMM-backed extensible tensors. It initially commits the null block,
  grows backing for warmup, then sizes the final KV capacity from measured
  headroom after warmup and Graph capture. Virtual addresses remain stable;
  software KV blocks remain. Sleep and cache-discardable weight updates
  also have release/remap paths. Distinguish startup sizing from a policy
  that migrates active requests and reclaims their backing during serving.
  Claims about fixed preallocated pools apply to the project's pinned
  baseline paths, not every present-day engine configuration.
  [Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/extensible_kv_cache.py)
  · [Configuration](https://docs.vllm.ai/en/latest/cli/run-batch/#--enable-extensible-kv-cache)
- **vAttention** keeps each request's KV tensor virtually contiguous and
  maps physical pages as needed. Standard attention kernels can address
  tokens directly. The runtime still manages page lifetimes. The public
  implementation integrates with Sarathi-Serve. Its 2 MiB mode uses the
  stock driver, while the documented smaller-page modes use a modified
  NVIDIA UVM driver. Background allocation and retaining completed
  requests' backing for reuse are established techniques.
  [Paper](https://arxiv.org/abs/2405.04437)
  · [Implementation](https://github.com/microsoft/vattention)
- **vTensor** maps discontiguous physical storage into extensible virtual
  tensor views and describes shared-prefix mappings. This provides a
  precedent for tensor growth and prefix sharing at the address layer.
  The paper and repository establish those mechanisms. Returning memory
  to another process requires separate evidence about the physical
  allocation lifecycle.
  [Paper](https://arxiv.org/abs/2407.15309)
  · [Implementation directory](https://github.com/antgroup/glake/tree/main/vTensor)
- **eLLM** uses VMM-backed elastic tensors to coordinate KV and activation
  memory. Its implementation integrates a C++ KV library into vLLM 0.5.5
  and modifies the PyTorch caching allocator for activations. This is a
  precedent for a shared budget across allocations. The paper describes
  the implementation. A public repository was not verified.
  [Paper, §3](https://arxiv.org/html/2506.15155v2)
- **Elastic KV** keeps the vLLM block table and attention kernel and changes
  the backing of a contiguous virtual range per layer. A scheduler
  controller drains and unmaps an elastic tail before prefill, then
  recommits it when capacity is available. The authors report Graph and
  prefix caching compatibility on vLLM 0.23.0. In their measured regime,
  reducing the prefill chunk size recovered more KV capacity at similar
  latency. The paper describes a pluggable allocator and harnesses. A
  public repository was not verified.
  [Paper, §3–6](https://arxiv.org/html/2608.23658)

## Sparse retrieval and residency

- **Original SGLang HiSparse**, as used by the audited MiniMax baseline,
  stages full history in pinned host memory and bounds device residency
  with a per-request hot buffer. Short sequences can fit in that buffer.
  The baseline's coordinator does not implement the physical-backing
  pressure policy used here to move an active request between full GPU
  history and a bounded hot buffer.
  [Guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hisparse_guide.mdx)
  · [Coordinator](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/hisparse_coordinator.py)
- **SGLang HiSparse V2** proposes HiCache and prefix-tree integration for
  DSA. In the PR implementation, a selected logical position with a
  resident device slot is read directly. Other positions use the hot
  buffer and can be fetched from pinned host memory. Indexer storage
  expands to keep the index for host-backed history on the GPU. This
  page-level mixed residency is a separate path from the original
  MiniMax baseline.
  [PR #32314](https://github.com/sgl-project/sglang/pull/32314)
  · [V2 cache implementation](https://github.com/bingps/sglang-fork/blob/pr/hisparse-v2/python/sglang/srt/managers/hisparse_v2_cache.py)
- **vLLM Hybrid HiSparse** changes residency while requests continue
  decoding. Resident and hot pages share the HMA pool and KV tensor.
  The published evaluation checkout allocates the backing at
  initialization and returns freed blocks to the pool queue, so those
  slots can serve other requests. Physical decommit is a separate
  operation from that block release.
  [Design and evaluation](https://vllm-project.github.io/2026/09/08/glm53-part1-hybrid-sparse-offloading.html)
  · [Backing allocation](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1/vllm/v1/worker/gpu/attn_utils.py#L254-L269)
  · [Block release](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1/vllm/v1/core/block_pool.py#L663-L685)
- **Adaptive DHiSparse** retains optional imported history while device
  capacity is available. It can reclaim the optional history of an
  already admitted request, serve misses through a fixed hot cache, and
  incrementally repopulate its history as capacity returns. Generated
  rows and the mutable tail follow separate rules. Pressure-driven
  residency and recovery during active decoding are established by this
  design.
  [Paper, §3.1–3.2](https://arxiv.org/html/2610.13708#S3.SS1)
- **LouisKV** organizes prompt KV by clusters of similar Key vectors
  and generated KV by temporal segments. A change in successive query
  vectors can trigger another retrieval. The system scores centroids to
  choose variable-size groups and transfers the selected KV rows from
  CPU to GPU. This changes what is selected and how much is transferred.
  The VMM systems above change how storage is addressed and backed.
  [Paper, §4.1–4.3](https://arxiv.org/html/2510.11292v1#S4)

VMM can back an extensible storage layout while cluster metadata still
maps cluster IDs to member rows and the resolver gathers those rows.
Scattered members can touch many physical chunks, keeping them resident
even when only a few rows per chunk are live. Mapping each cluster
separately can also increase padding and mapping work. Compare original
token order and cluster-packed layouts with the same selection trace.
Measure useful bytes, touched chunks, physical backing and relocation
cost alongside attention time.

**MorphServe** is a precedent for pressure-aware KV resizing with
quantized weight swapping. **Llumnix** is a precedent for moving active
requests between serving instances. The prototype here changes residency
within the same serving instance and TP group.
[MorphServe](https://arxiv.org/abs/2506.02006)
· [Llumnix](https://arxiv.org/abs/2406.03243)

## Matched comparisons

The empirical question is whether joint planning can release actual
physical chunks that are shared by independent request pages while
preserving native sparse attention. Keep the model, logical capacity, hot
buffer, workload and pressure thresholds matched. Compare native
HiSparse, static mixed residency, and adaptive single-request and joint
planning. Use the same VMM backing in the static and adaptive arms to
isolate policy. To isolate physical release, compare the same residency
strategy under a fixed backing and a decommittable backing.

Report software slots, transfer bytes, actual mapped bytes and device
memory separately, alongside migration stalls, TTFT and TPOT. Check
committed history with the logical shadow in separate correctness runs.
Repeat pressure episodes with different placement histories before
generalizing a joint release. For a future activation budget experiment,
include a tuned prefill chunk baseline and report observed headroom
separately from the amount of KV backing returned.
