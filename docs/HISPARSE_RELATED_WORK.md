# Related work and claim boundaries

- **SGLang HiSparse** documents an instance-level decode switch with pinned
  host history, a per-request device buffer and sparse swap-in. A short
  sequence can fit entirely in that buffer. The audited coordinator
  provisions host-backed requests and grows short buffers up to a
  configured limit. The guide and coordinator do not expose a scheduler
  policy that moves active requests between unrestricted GPU history and
  the fixed hot buffer in response to measured physical backing.
  [Guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hisparse_guide.mdx)
  · [Coordinator](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/hisparse_coordinator.py)
- **vAttention** already separates virtual and physical KV allocation.
  Section 6.2 describes mapping upcoming decode pages on a background
  thread, proactively mapping for inactive request IDs, and retaining
  completed requests' backing for reuse. Its reclamation policy uses
  physical page availability. These are precedents for asynchronous
  allocation and deferred reclaim. Its demonstrated mapping scheme
  addresses contiguous request tensors, while this prototype retains
  SGLang's software page mapping.
  [Paper, §6.2](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/vattention_arxiv24.pdf)
- **MorphServe** resizes KV cache capacity under memory pressure and swaps
  quantized model layers. It is a direct precedent for pressure-aware
  cache budgeting. Its published mechanism does not establish the
  MiniMax HiSparse full-history to fixed-hot-buffer transition or
  shared VMM chunk release studied here.
  [Paper](https://arxiv.org/abs/2506.02006)
- **Llumnix** migrates running requests and their state between serving
  instances to address load imbalance, fragmentation and service goals.
  This is a precedent for live request migration. The mechanism studied
  here changes residency within the same serving instance and TP group.
  [Paper](https://arxiv.org/abs/2406.03243)

The empirical question here is whether a scheduler can identify and
release **actual shared physical chunks** by choosing migrations jointly
while preserving native sparse attention. Logical slots freed and bytes
copied do not measure that outcome. Attribute release with native HiSparse,
static mixed residency, single-request planning and pair planning under
the same model, logical capacity, device buffer, workload and pressure
thresholds. Include the same VMM backing in the static and adaptive arms
to isolate the policy. Report mapped bytes and device-wide memory
separately, alongside transfer volume, migration stalls, TTFT and TPOT.
Check the committed history with the logical shadow in separate correctness
runs. Use repeated pressure episodes with different allocation layouts
before generalizing an observed pair benefit.
