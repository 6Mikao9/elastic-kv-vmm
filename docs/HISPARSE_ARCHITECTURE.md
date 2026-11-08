# MiniMax HiSparse and VMM architecture

This prototype gives each MiniMax request one of two KV residency modes.
A full-resident request keeps its sparse-layer history on GPU. A host-backed
request keeps that history in pinned CPU memory and uses the native HiSparse
device buffer during decode. Both modes use the same MiniMax sparse
attention, indexer and top-k selection. The current implementation targets
the pinned `lmsysorg/sglang:dev-minimax-m31-flash` image and checks source
hashes before installing hooks.

## Common SM80 runtime

All matched SM80 arms use the same opt-in MiniMax M3.1 compatibility path.
It preserves SGLang's sharded MXFP8 parameter loading, converts projections
to BF16 after loading, and uses BF16 linear operations. The NVFP4 MoE
experts retain the image's Marlin W4A16 path. Native MXFP8 QKV/index fusion
is disabled for those BF16 projections. This compatibility change is
separate from residency and applies equally to controls.
[SM80 adapter](../adaptive_sparse/minimax_sm80.py)

## Logical slots and physical memory

The native `HiSparseTokenToKVPoolAllocator` has separate logical and device
paged allocators. Prefill `alloc_extend` assigns both and fills the
logical-to-device mapping. Native `alloc_decode` assigns a logical slot.
The coordinator supplies the corresponding device location for host-backed
requests, while the resident path extends its own physical allocation and
updates the same mapping.

Only the sparse main K/V pool uses elastic physical backing. The VMM
adapter replaces its BF16 NHD planes with views into stable virtual address
ranges, tracks physical page ownership and maps chunks before a page can be
used. A chunk can be released only when all page references are gone.
Trimming waits for GPU work before unmapping. The optional decode fast path
uses CPU sequence lengths to bypass VMM bookkeeping when an append stays in
an already backed page.
[VMM pool and allocator hooks](../adaptive_sparse/minimax_vmm.py)

The indexer, dense-layer KV, weights and activations use their existing
allocators. The request table, software paging and logical-to-physical
translation remain part of the attention path. VMM changes when physical
bytes are committed. It does not remove the software page lookup.

## Mixed decode

The request residency hook wraps the MiniMax HiSparse coordinator and
`MiniMaxSparseAttnBackend._hisparse_swap_in_blocks`. A selected full-resident
request keeps the prefill physical mapping. It enters the original staging
queue with a readiness event, preserving scheduler order and TP readiness,
and skips the initial host copy and side-buffer allocation. Other requests
use the native staging path.

The coordinator's `map_last_loc_to_buffer` runs on the host-backed rows
only, where it handles previous-token backup and the latest-token slot.
At attention, a full-batch variant gives resident rows zero sequence
length in the native swap call. The guarded CUDA kernel emits padding
slots for those rows. The resident resolver then fills their valid
positions by expanding the original selected blocks through the request
table and logical-to-device map. This keeps native top-k semantics and
batch order. The optional Triton resolver combines expansion, bounds
checking and lookup in one kernel.
[Request residency](../adaptive_sparse/request_residency.py) ·
[Resident lookup](../adaptive_sparse/resident_lookup.py)

Host-backed rows use a fixed hot-buffer capacity `T` plus an extra
allocator page for the newest token. The native top-k budget specifies
selected blocks. The hot-buffer capacity also has to accommodate those
selections and the coordinator's layout constraints.

## Known-future hint consumer

The opt-in future path can carry a `residence` hint with each request.
At HiSparse staging admission, `host_backed` yields the complete-GPU slot,
`resident` requests it when capacity exists, and `auto` compares predicted
output length with a configured threshold. This decision runs after the
request and its prompt are known, so it can use the observed allocated KV
length. All TP ranks apply the same policy and the existing physical-reserve
vote remains authoritative. Missing hints preserve the legacy long-request
default; the feature is inert unless `EKV_RESIDENT_HINT_POLICY=adaptive`.
[Hint residency policy](../adaptive_sparse/future_residency.py) ·
[Residency growth broker](../adaptive_sparse/residency_broker.py) ·
[Future scheduler hook](../adaptive_sparse/sglang_future_hook.py)

## Budget and migration

The scheduler evaluates committed main-KV backing at safe eager
boundaries. A high watermark can trigger demotion. A lower watermark,
stable interval and minimum dwell time govern promotion eligibility. The
policy also bounds migration traffic and checks whether promotion can
allocate a complete destination while the source remains live. Forced
decode steps support controlled tests of both directions.
When a request has `kv_hints`, demotion ranking divides the existing
byte-efficiency term by a bounded retention value derived from business
weight, remaining output, deadline distance and shared-prefix reuse potential.
An imminent deadline protects an individual or joint candidate from ordinary
pressure; SGLang's hard priority and native admission remain unchanged.
[Scheduler budget](../adaptive_sparse/scheduler_budget.py) ·
[Runtime value](../adaptive_sparse/runtime_value.py)

The planner counts releasable physical chunks using the page ownership
tracker. A long request can have few exclusive chunks if pages from
several requests share the same backing. The optional retention planner
concentrates the fixed hot pages into shared or already protected chunks,
and the scheduler can consider a pair of requests when neither alone
releases enough memory. The copy into host memory follows original
logical order. Hot-buffer locations are selected separately after
backup, and the native kernel accepts a per-slot location list.
[Migration page planning](../adaptive_sparse/migration.py)

Migration separates prepare from commit. Each TP rank prepares its copy
and votes. After a rejected prepare, successful ranks release prepared
destination resources. A commit failure is fatal to the group. Demotion
backs up all committed sparse main K/V, transfers retained pages to hot
ownership, clears the old mapping, then releases the remaining pages.
Promotion allocates fresh full-resident pages and restores history with
the native per-layer host loader. Immediately after staging or demotion,
host holds the latest committed token. After ordinary decode, that token
may still be in the reserved device slot, so promotion copies it from
there before publishing. The source host and hot state remain live until
the destination is ready.
[Migration transactions](../adaptive_sparse/migration.py)

The optional logical shadow records BF16 producer values by logical
slot before translation. It checks selected sparse decode reads and can
check the complete committed history at migration boundaries. It adds
memory and work and is intended for correctness runs.
[Logical KV shadow](../adaptive_sparse/logical_shadow.py)

The prefill headroom term is an input to the residency budget heuristic.
It does not reserve PyTorch activation or workspace memory, guarantee
admission or govern Graph private pools. PyTorch allocator telemetry,
VMM backing and whole-device memory are separate measurements.
[Allocator telemetry](../adaptive_sparse/memory_telemetry.py)

## CUDA Graph boundary

This MiniMax integration is eager. Routing, row selection, page
commitment and migration use host decisions and dynamic metadata.
The separate dense SGLang VMM Graph integration does not validate this
MiniMax path.

A Graph extension would use stable-address, fixed-capacity inputs for
request rows, modes, lengths, top-k slots and output slot tables. A
resident row would be masked out of the swap path using persistent
device metadata, with safe sink entries for padding. Keep transitions
and mapping changes outside capture/replay, and publish new routing
only after transfer completion and before replay. Reclaim only after
the previous users are fenced. Graph capture, replay across migration
and padding correctness would need independent validation.
