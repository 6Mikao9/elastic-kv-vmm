# Direct shared-prefix reads for MiniMax HiSparse

`EKV_SHARED_PREFIX_REUSE=direct` enables an experimental data-plane path for
requests carrying a valid `elastic.future` shared-prefix claim. It is disabled
by default and requires `EKV_RESIDENT_HINT_POLICY=adaptive`. Full-batch swap is
not compatible because that path launches native host loading before the
runtime can replace selected rows.

The smoke runner exposes the same switch and validates the complete dependency
chain before starting a server:

```bash
bash scripts/container_hisparse_smoke.sh \
  0,2,3,4 /path/to/MiniMax-M3.1-Flash native \
  results/hisparse-prefix/direct \
  --vmm-backing --resident-limit 2 --resident-free-reserve 4096 \
  --resident-hint-policy adaptive --future-policy deadline \
  --future-workload benchmarks/workloads/known-future-rollout.json \
  --shared-prefix-reuse direct
```

Use the same command with `--shared-prefix-reuse off` as the paired control.

The source is a live full-resident request. A host-backed request resolves each
Top-K block that lies wholly inside the declared shared prefix through the
source request's logical-to-physical mapping. Those output slots point at the
source request's device KV pages, so the native host-copy kernel is skipped for
that subset. Blocks outside the prefix, partial blocks, and requests without a
valid source continue through the unchanged native HiSparse path.

Sharing requires equality of the declared group and token length plus a
sha256 identity over the complete prefix token ids, `extra_key`, cache salt,
and available LoRA/adapter identifiers. Input-embedding and multimodal
requests are excluded because equal token ids do not establish equal prefill
state. MiniMax Top-K operates on complete sparse blocks, so only blocks with
`block_start >= 0` and `block_end <= shared_prefix_tokens` participate.

The source remains owned by its resident request. It is removed from the
registry before completion, abort, or demotion, and consumers then fall back
to host staging on their next decode step. This deliberately avoids adding a
second physical owner or changing VMM chunk reference counts. A later version
may pin pages across source completion, but that requires an explicit lifetime
and budget contract.

Promotion restores the metadata source after the complete GPU mapping has been
published. The scheduler's VMM reclaim path remains authoritative: direct reuse
never extends a source lease, and a Top-K lookup from the same layer remains
valid only while that source request is still resident.

The opt-in path is intended first for correctness and hit-rate evaluation.
Compare it with native HiSparse under identical Top-K and residency settings,
and report `shared_prefix_direct_blocks`, `shared_prefix_native_blocks`,
`shared_prefix_native_requests`, `shared_prefix_partial_requests`, TPOT, host
transfer bytes, and source-removal events. Planning-only prefix credit remains a
separate capacity model until this data path has passed real MiniMax GPU
validation.
