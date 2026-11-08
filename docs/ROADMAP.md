# Research scope and order

## Published dense backing phase

The dense SGLang and vLLM experiments cover the following work before
changing the attention addressing model:

1. Reduce allocator overhead and document memory/latency tradeoffs.
2. Support CUDA Graph capture and replay, including safe padding lanes and
   physical reclaim/recommit between requests.
3. Evaluate application value with an idle co-tenant and bursty long-context
   demand, including a manually tuned static-pool control.
4. Coordinate prefill activation workspace with reclaimable KV backing and
   PyTorch cached memory. Separate safe reclamation from unimplemented
   admission guarantees; never reclaim live KV to manufacture headroom. The
   V5 physical-admission slice now reports a conservative combined KV-plus-
   prefill token budget through SGLang's native `available_size()` call. Exact
   activation ownership, transactional failure and cross-process grants remain
   open.
5. Add a vLLM implementation using existing local model snapshots read-only,
   and validate correctness, memory and serving efficiency against vLLM's
   own baseline.
6. Publish tested code, reproduction commands, raw measurements and limitations.

## Request-level HiSparse phase

The current MiniMax prototype keeps the native sparse attention algorithm
and adds full GPU residency alongside the fixed native HiSparse hot pool.
It implements safe eager transitions, physical-chunk-aware candidate
selection, joint demotion, low-water promotion and anti-thrashing controls.
Full selected-KV and migration-history byte checks have passed in the
recorded scenarios. Compare native HiSparse, static mixed residency,
fully resident execution and matched single/joint pressure policies;
publish repeated timing, actual backing, whole-device memory and limits.

Prefill activation allocation and MiniMax CUDA Graph support remain
separate extensions. A configured prefill headroom term is not ownership
of the activation allocator. See the [research report](HISPARSE_REPORT_ZH.md).

Known-future rollout hints now have three connected consumers: the scheduler can
stabilize the waiting order after SGLang's native policy, an opt-in
request-residency policy can decide whether a long request claims a complete-GPU
KV slot or stays on the native bounded HiSparse path, and a growth ledger can
prevent multiple claims from spending the same predicted decode headroom. The
residency path is enabled with `--resident-hint-policy adaptive`; it remains
disabled by default and still passes every native capacity check. This is an
integrated mechanism, not yet a measured end-to-end speed result.

The implemented admission ledger converts predicted
decode tokens into KV bytes and subtracts both the fixed free reserve and growth
already promised to other resident requests before accepting another complete
GPU history. A dry-run mode records the decision without changing placement;
an enforce mode falls back to native HiSparse. Prediction and consumption follow
the allocator page geometry, and scheduler demotions/promotions update the same
ledger. This closes a concrete accounting gap between request-count capacity and
future physical KV demand. Its transition cost and serving benefit still require
GPU measurement.

Known-future prefill reservation now aggregates the bounded footprint of
requests inside a configurable horizon.  Request-level full or hot-set KV and
an optional measured activation term are checked against one byte cap; TP
ranks atomically fall back to zero reservation when their plans disagree.
`dry_run` exports the demand, while `enforce` feeds it into the existing
prefill-pressure migration path.  The planner never allocates for future
requests and SGLang's `PrefillAdder` remains authoritative.  The activation
coefficient defaults to zero until calibrated because no source-level estimate
has yet been validated on the target MiniMax workload.

Controlled-rollout planning now emits an executable submission schedule from
the offline policy.  Each request carries its planned arrival, type, expected
output length, relative deadline and selected resident/host-backed mode, so a
rollout controller can feed the existing SGLang hint transport without
bypassing native admission.  The planner may deliberately idle a short interval
when starting a long prefill now would make an imminent hard-deadline request
miss. The live executor now consumes this schedule, adjusts relative deadlines
against the original absolute deadline, rejects inconsistent plans before
dispatch, and records actual submit/finish, deadline outcomes, TTFT, TPOT, full
makespan throughput and per-TP residency evidence. A matching `--no-hints`
control arm makes paired real-serving measurement possible. This behavior is
validated in deterministic CPU tests; it remains to measure whether
control-plane idle time and prediction error are worthwhile on real serving
workloads.

The planner and request hint now carry shared-prefix identity. Deterministic
capacity accounting charges a shared immutable prefix once when multiple
requests in the same `prefix_group` are simultaneously resident, and imminent
future reservations apply the same credit. HiSparse requests remain fully
charged because their Top-K hot set is not guaranteed to retain the prefix.
This is planning and transport evidence; the current radix-disabled SGLang
runtime does not yet reference another request's resident KV or resolve a
Top-K hit from it.

The controller now also tracks the real in-flight set.  A bounded concurrency
mode turns planned timestamps into earliest admission times, deadline ordering
selects among already-eligible requests, and an aggregate byte ledger demotes
only newly admitted long requests from full residency to the native HiSparse hot
set when the live set crosses the planner budget.  Requests that cannot fit even
their hot set wait for an actual completion unless a hard deadline requires an
explicitly recorded overcommit.  The next evaluation should compare timeline,
deadline ordering, static residency and live elastic residency using the same
requests and report capacity delay, demotions, deadline misses, TTFT/TPOT and
whole-device memory separately.

### Performance targets and next optimizations (2026-11-04)

The dynamic policy has two reference targets: approach fully GPU-resident
performance at low load, and approach native HiSparse serving performance
at high load. Keep the same native sparse attention semantics and matched
workloads. At low load compare TTFT, TPOT and tail latency; at high load
compare throughput under a common latency objective, capacity and failure
rate. Measure transition stalls and recovery time separately. These are
acceptance goals, not demonstrated results or an assumed speedup threshold.

Physical chunk co-occupancy is an implementation issue to optimize, not
prefix sharing: the 240 MiB/rank joint-reclaim example used independent
request pages and disabled radix caching. The scheduler now maintains
incremental request-to-page/chunk and chunk-to-owner indexes for pressure
candidates. Shared-chunk owner closures seed a bounded two-to-four-request
search; overlapping-page cases retain the exact page-union fallback. Candidate
ranking includes whole-chunk release benefit and transfer cost, followed by the
existing TP agreement and exact pre-commit checks. The [initial CPU
microbenchmark](GROUP_INDEX_MICROBENCH.md) reduced a 32-request synthetic
search from 256 examined combinations to 0–16 owner closures. The next step is
to measure index maintenance on real allocator callbacks and run a paired
serving trace against exhaustive search. Request-affine placement without
excessive tail waste remains open.

Separately, explore reuse of identical prefix KV that another request still
holds on the GPU. A HiSparse Top-K lookup could first resolve a compatible
shared GPU copy and avoid host-to-device fetch when available. Begin with
immutable complete prefix blocks and device-to-device fill of the existing
hot buffer; evaluate direct shared reads only after correctness. Identity
must include model/weight version, full prefix, position/RoPE, adapter and
other input-dependent conditions. Protect shared and in-flight references,
use copy-on-write for mutable tails, and version mappings across eviction
or migration. Pinned shared chunks can reduce reclamation, so their budget
cost must be included. Benchmark against equivalent native prefix-cache
settings; this potential benefit is not automatic from VMM and has not yet
been measured in the current radix-disabled MiniMax experiments.

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

The dense, MiniMax and initial vLLM adapters retain their engines' logical
slot/block indexing. They should be described as elastic physical-backing
prototypes, not as completion of the deferred software-page-table removal.
