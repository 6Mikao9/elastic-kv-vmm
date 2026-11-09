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
   prefill token budget through SGLang's native `available_size()` call. A
   per-GPU SQLite broker now provides cross-process VMM leases, dense/MiniMax
   pressure signals, and opt-in weighted soft fairness between explicit tenant
   ids. A temporary prefill capacity claim now shares the same atomic broker
   budget as VMM leases before batch selection and is released at the successful
   forward boundary; observe mode records would-admit without mutation. Exact
   hard partitions, true activation ownership and transactional OOM recovery
   remain open.
   SGLang schedule overlap is now an explicit V5-only experiment. The
   implementation keeps a device-wide fence before physical unmap and rejects
   two-batch overlap; correctness, memory and serving latency still need a
   paired GPU comparison before it becomes a default.
5. Add a vLLM implementation using existing local model snapshots read-only,
   and validate correctness, memory and serving efficiency against vLLM's
   own baseline.
6. Publish tested code, reproduction commands, raw measurements and limitations.

The tenant fairness slice is opt-in and defaults off. Denied growth writes a
bounded demand record; Dense V5 stops admitting a starved tenant's competitor,
and MiniMax uses the same signal both at new request-residency admission and
for cooperative demotion. A request that would claim a complete-GPU history
under live fairness pressure falls back to native HiSparse before allocating
that history; all TP ranks use the conservative common result. Ownership remains
with the original process: a lease is transferred only after successful
`cuMemUnmap` and release. The dense dual-instance runner now exposes `off`,
`observe` and `enforce` with per-model tenants, samples broker and NVML state,
executes `a-hot`, `b-hot`, staged `a-to-b`/`b-to-a` transitions and
`both-active`, synchronizes dispatch, and reverses hot-instance order across
repeats. Transition schedules keep the incumbent active while the incoming
instance arrives after a configured gap. They record per-instance start/finish
times, observed overlap and overlap-token throughput, and fail closed if a
supposed transition is actually serial. It writes raw request/sample files
plus a runtime/source-bound summary and an untimed non-streaming correctness
matrix. Request count and prompt/output lengths are explicit, and results record
the model-derived logical KV demand against broker capacity. The next evidence is a
paired GPU run, including an unfair long-lived burst; MiniMax still needs its
request-level pair driver. Until that run, the implemented weighted targets,
anti-starvation, request-admission fallback and bounded retry are CPU and
multi-process SQLite evidence, not an end-to-end throughput or latency result.

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
prefill-pressure migration path. During decode, the waiting queue is refreshed
every configured observation interval so a known imminent prefill can cause
reclaim before backing already crosses the high-water mark. Fixed prefill
scratch is included only at prefill boundaries; decode pressure uses predicted
future demand without permanently inflating its watermark. The planner never
allocates for future requests and SGLang's `PrefillAdder` remains authoritative.
The activation coefficient defaults to zero until calibrated because no
source-level estimate has yet been validated on the target MiniMax workload.

V5 can now turn that same bounded plan into optional physical prewarming. It
selects only chunks backing currently free logical slots, starts the existing
asynchronous VMM mapping/zeroing path, and records TTL hit and expiry outcomes.
The default is off; `dry_run` observes candidate chunks and `enforce` maps them.
Because speculative mappings consume physical and global-broker capacity, they
remain bounded by TTL and a chunk cap, never claim logical slots, and leave the
native `PrefillAdder` authoritative. Real paired GPU runs are still required to
establish whether avoided mapping latency exceeds the false-prewarm cost.

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
Active-prefix prefill reuse now has a fail-closed execution guard: a rollout that
turns on the timing model must explicitly prove that the launched SGLang
instance has radix caching enabled. This prevents the existing radix-disabled
adapter and smoke runners from consuming planner-only prefix credit as if it
were a real hit. The guard does not implement shared VMM backing or relax the
adapter's current radix-cache rejection.

Known-future timing can now consume a measured aggregate decode-throughput
profile at exact co-batch sizes. The planner divides aggregate throughput by the
batch size, records the estimate used by each admission, and emits the matching
per-stream rate in the rollout hint. Missing measurements retain the ordinary
linear model and are never interpolated. Because an already-admitted request is
not rewritten when a later request joins its batch, the next step is to collect
real SGLang co-batch traces and compare predicted with observed TTFT, TPOT and
makespan before treating the profile as a serving performance claim.
The known-future `batched` policy adds exact-size aggregate prefill profiles and
jointly placed resident/HiSparse cohorts. It waits only when a measured profile
improves both weighted completion and makespan without turning a satisfiable
deadline into a miss; missing measurements use serial prefill and linear decode.
Cohort IDs, prefill/decode batch sizes and latest-admission timing are carried
through the executable plan and validated before dispatch. This remains a
planner/controller mechanism. Future work must capture real SGLang prefill and
decode profiles for each target model, verify that native `PrefillAdder` forms
the intended cohort, and report predicted versus observed TTFT, TPOT, makespan,
prefill-workspace peak and whole-device memory.

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

Closed-loop service-time feedback now separates decode observations by the
request's actual `resident` or `host_backed` placement. Runtime demotion no
longer leaves a host-backed request carrying a resident-only EWMA, and an
immature mode falls back to the planner's placement-specific rate instead of a
mixed estimate. Per-request evidence records which feedback placement was used,
while the summary exports both mode-specific rates and sample counts. This is a
CPU-validated policy correction; real serving validation remains required.

Low-water promotion now ranks the TP-common candidates by the same immutable
request value used for demotion: imminent deadlines and high-weight work come
before ordinary requests, transfer cost breaks ties, and every rank uses the
worst-rank cost. The previous behavior selected the lexicographically smallest
request id. Promotion telemetry records candidate count, hinted/invalid values,
deadline-guarded selections and the chosen retention evidence. Prefix reuse can
raise a donor's priority using verified active consumers and the bounded future
demand forecast.

Future-aware admission now reserves interval peaks instead of summing every item
inside the horizon, skips deadline protection when positive post-current slack
remains, and recomputes service duration for backlogged requests. TP prefill-plan
signatures also distinguish KV from activation bytes. These are CPU-validated
accounting fixes. The rollout executor now emits action version 2 with a
host-local `perf_counter` absolute deadline captured at HTTP dispatch; the
scheduler prefers it and retains version-1 relative hints as a compatibility
fallback. This removes network, tokenization and queue time from deadline slack
within one host. Cross-host rollout remains future work and needs calibrated
Unix-clock offsets rather than assuming a shared monotonic reference.

Active residency now consumes the same immutable hint at demotion time.  A
conservative retention value combines business weight, remaining decode work,
deadline distance, and potential shared-prefix reuse; imminent deadlines and
joint request groups are ranked in a protected tier.  Missing hints remain
neutral, so legacy ordering is unchanged.  This is a tested CPU policy path;
serving gains still require the pending exclusive-GPU comparison.

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
compares fixed slots, per-request chunk-rounded private slabs and exact-length
packed request ranges. Private slabs remove cross-owner chunk coupling but pay
per-request tail slack; packing admits more mixed/short-heavy requests but
couples reclamation. At matched concurrency all three reach the same physical
chunk union. A large-private/small-packed hybrid cuts the synthetic mixed
peak from four shared chunks to one while retaining the private arm's
active-request capacity. A real trace, GPU mapping cost and serving comparison
remain open.

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
