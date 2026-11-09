# SGLang Dual-Instance Integration Audit

> Status: the original findings below describe the audited baseline commit and
> remain as design history. “Resolutions applied after the audit” at the end of
> this document is authoritative for the current worktree.

Audit date: 2026-11-09

Repository: `D:\ws\01_研究\实验环境\sparse-kv-lab\vmm_elastic_pool\e2e`

Baseline: `51b03686af08dc181d79e723b8d04f83aac81cec`,
`feature/sglang-framework`

## Verdict

The broker and dual-runner plumbing are structurally sound enough for focused
CPU tests, but this revision is **not yet sufficient for a dual-instance GPU
acceptance claim**. Two issues need correction before running the experiment:

1. Weighted fairness can admit a batch that takes a tenant far beyond its
   target when that tenant is only slightly under target before the request.
2. The run manifest does not identify the complete serving implementation or
   runtime, so an unchanged `source_sha256` can describe different experiments.

The runner also implements fewer phases and less telemetry than the accompanying
validation protocol requires. It can be used for a preliminary smoke test, but
not as evidence for the full arm matrix, load-reversal robustness, or a complete
reproducibility claim.

## Findings

| Priority | Finding | Evidence | Impact |
|---|---|---|---|
| P1 | Fair-share enforcement checks the tenant's usage before the batch, not usage plus the requested bytes. | [global_broker.py:390](../elastic_kv/global_broker.py#L390) selects the current usage; [global_broker.py:412](../elastic_kv/global_broker.py#L412) only tests `tenant_usage >= target`; [backing.py:281](../elastic_kv/backing.py#L281) can submit all pending chunks as one broker request. | A large map/precommit batch can overshoot a weighted target by many chunks. The next request is blocked, but the first request can consume the room the under-target tenant needed. This is stronger than the documented one-chunk boundary allowance. |
| P1 | The source manifest omits the central SGLang adapter and configuration files. | [run_dual.py:29](../scripts/run_dual.py#L29) enumerates the hashed files; `elastic_kv/adapter.py` and `elastic_kv/config.py` are absent. | Adapter or launch-config changes can leave `source_sha256` unchanged. The result hash does not identify the behavior under test. |
| P1 | The run does not record enough runtime identity for reproduction. | [run_dual.py:413](../scripts/run_dual.py#L413) records only model labels, ports, PIDs, a partial source hash and broker values. It omits the Git commit and dirty state, local image digest, SGLang/PyTorch/CUDA/driver versions, GPU UUID, model paths/revisions, exact launch command and native library hash. | A run cannot be tied to the actual container, binary, model snapshot or SGLang implementation. This conflicts with [DUAL_INSTANCE_VALIDATION_ZH.md:43](DUAL_INSTANCE_VALIDATION_ZH.md#L43) and [line 212](DUAL_INSTANCE_VALIDATION_ZH.md#L212). |
| P1 | The claimed load-reversal protocol is not implemented by the driver. | [run_dual.py:296](../scripts/run_dual.py#L296) launches four equal requests per active instance at once; [run_dual.py:411](../scripts/run_dual.py#L411) runs `a-hot`, then `b-hot`, then `both-active` in a fixed order. There is no independent process barrier, configurable arrival schedule, steady window, explicit transition, over-demand burst or drain phase. | The output cannot establish synchronized arrival, reversal order robustness, admission rejection under an exactly controlled over-budget demand, or drain latency. It may show basic coexistence, but not the core fairness experiment. |
| P1 | Dual telemetry omits the broker/admission and reclamation counters required by the protocol. | [run_dual.py:250](../scripts/run_dual.py#L250) copies only `backed_bytes` and `live_tokens`, while [DUAL_INSTANCE_VALIDATION_ZH.md:37](DUAL_INSTANCE_VALIDATION_ZH.md#L37) asks for mapping, reclamation and admission counters. | The run cannot distinguish successful allocation, broker denial, wait, retry, unmap and reclamation behavior from aggregate backing totals. |
| P2 | A broker sample is treated as initialized if any active sample is ready. | [run_dual.py:321](../scripts/run_dual.py#L321) accepts one ready sample; malformed, locked or inconsistent samples remain silently `uninitialized`. | Transient failures are hidden and serialized samples do not form a complete audit trail. Record every failure and fail the phase on a configured maximum error rate. |
| P2 | `--broker-mib -1` is accepted when no broker path is supplied. | [run_dual.py:89](../scripts/run_dual.py#L89) compares `bool(path)` with `mib > 0`, so both `False` and `False` agree; validated directly. The value is then recorded in results despite being ignored. | Invalid or misleading configuration can enter a result manifest. Reject any nonzero capacity without a path and reject all nonpositive capacities. |
| P2 | The result schema and atomic-write behavior differ from the validation plan. | [run_dual.py:447](../scripts/run_dual.py#L447) writes one embedded `results.json` only after a phase succeeds; [DUAL_INSTANCE_VALIDATION_ZH.md:76](DUAL_INSTANCE_VALIDATION_ZH.md#L76) specifies separate raw files plus a summarized schema with `status`, `arm`, `configuration`, `violations` and `source_manifest`. | A tool consuming the documented schema will not read these results, and a failed phase leaves no explicit incomplete/violation record. |
| P2 | The new tests validate a hand-built broker schema rather than an end-to-end runner against the production broker owner. | [test_run_dual.py:52](../tests/test_run_dual.py#L52) creates the tables directly; there is no mocked `main()` lifecycle test, container argument-forwarding test, or real `GlobalAdmissionBroker` snapshot test. | Schema drift, startup order, cleanup, and fairness overshoot can pass the focused runner suite. |

## Concrete fairness reproduction

The existing broker code was exercised with a 100-byte capacity, equal weights,
and both tenants active. Tenant B first attempted one byte while A held the
entire capacity, creating B's demand. A then released 60 bytes and requested a
single 60-byte batch:

```text
A usage before request: 40
A target:                50
A requested batch:       60
Decision:                admitted
A usage after request:   60
B remained starved:      true
```

The broker correctly blocked A's next growth request after the overshoot. That
does not repair the loss: the atomic batch already received more than its
remaining five-byte share. With the default 2 MiB chunks, the same behavior can
move a multi-chunk batch past the target. Enforcement should either reject a
batch that would exceed the deficit while another tenant is starved, or split it
at the share boundary while preserving atomic mapping failure semantics.

## What is working

- The dual driver forwards distinct tenant ids and weights to the two server
  processes and rejects an existing broker file before launch.
- The broker snapshot uses a read-only SQLite URI and a transaction, so routine
  sampling does not create broker owners or take a write transaction.
- Enforce mode requires two distinct tenants, and broker capacity must be
  supplied with a path for the normal `v2` path.
- The global broker remains authoritative for aggregate leases and does not
  claim cross-process live-KV preemption.
- The focused dual-runner tests pass, as do the adjacent broker, backing,
  configuration, and single-runner tests.

## Required changes before GPU acceptance

1. Bound fairness at prospective usage. While another tenant is starved, deny
   or deterministically split a batch whose `usage + required` exceeds the
   tenant target. Preserve the all-or-nothing lease decision for each admitted
   native mapping operation.
2. Build a complete provenance manifest before server startup. Include the Git
   commit and dirty diff hash, container image id/digest, SGLang, PyTorch, CUDA,
   driver and library versions, GPU UUID, model revisions, exact server command,
   native library hash, every serving source that can affect behavior, and the
   SQLite WAL/journal state.
3. Implement process-level barriers and a configurable arrival schedule. Add at
   least balanced `a-hot -> b-hot` and `b-hot -> a-hot` orders, a jointly
   over-budget burst, explicit queue/admission recording, and a drain phase that
   waits for lease removal rather than a fixed 300 ms sleep.
4. Preserve the full adapter counters in each dual sample. At minimum retain
   broker reservations, denials, retries and wait time; mapping/ensure and
   reclaim counts and bytes; precommit activity; and per-instance live tokens.
5. Emit separate raw request/sample files and a summarized result with explicit
   `status`, `arm`, `configuration`, `violations` and `source_manifest` fields.
   Write an incomplete result with failure evidence when a phase raises.
6. Count broker snapshot failures. Fail the phase when no owner exists after
   startup, when the snapshot schema is invalid, or when the failure rate exceeds
   a predeclared threshold.
7. Reject every nonpositive or pathless nonzero broker capacity and test that
   rule explicitly.
8. Add tests that instantiate the real broker, exercise the fairness overshoot,
   run the runner against mocked servers for both normal and failed phases, and
   assert container argument forwarding.

## Validation performed

- `unittest`: 6/6 `test_run_dual.py` tests passed.
- Adjacent regression set: 39 tests passed, with one CUDA-only test skipped.
- `py_compile`: `run_dual.py`, `test_run_dual.py`, and `global_broker.py` passed.
- `ruff check`: 11 diagnostics in the two new Python files (3 auto-fixable style
  issues, 7 suggested rewrites, and one nested-`if` simplification). These are
  secondary to the correctness and evidence gaps above.
- No GPU serving run was attempted on this Windows host. The audit establishes
  implementation and evidence-contract gaps; it does not measure SGLang
  throughput, latency, OOM behavior, or actual VMM reclamation.

## Resolutions applied after the audit

- Fair-share enforcement now checks prospective tenant usage
  (`current usage + requested bytes`) against the weighted target. A batch that
  would overshoot the target while another tenant is starved is rejected as one
  atomic operation. A regression test covers the 40 + 60 versus target-50 case.
- Known-future reservations now use concurrent interval peaks, skip
  deadline-only pressure when positive post-current slack remains, retain
  independent higher-weight protection, and retain predicted service time after
  backlog.
- TP prefill signatures include KV and activation bytes separately so ranks
  cannot agree on the same total while computing different effective headroom.
- A deadline hint without a scheduler timestamp now produces a neutral value and
  increments the invalid-hint counter instead of granting implied full slack.
- The dual runner now hashes every serving source, records Git dirty/untracked
  state, model/config identities, container image id/digest, package and torch
  runtime, selected GPU identity, native-library hash, exact launch command and
  SQLite WAL/journal state.
- Request groups pass through an in-driver `threading.Barrier`; repeats reverse
  the hot-instance order while keeping the joint phase last. `a-to-b` and
  `b-to-a` phases dispatch the incumbent first and start the incoming
  instance after a configurable gap, so the test includes an actual overlap
  transition instead of only independent hot phases. Each phase records actual
  per-instance start/finish times, observed overlap and overlap-token
  throughput; a nominal transition with no observed overlap fails closed.
  Warmup and measured KV lifecycles use an explicit drain wait rather than one
  widened sleep.
- Raw request/sample artifacts are written separately. `results.json` now has
  `status`, `arm`, `configuration`, summarized `phases`, `violations` and
  `source_manifest`, and failed phases leave a machine-readable incomplete
  record. Adapter samples retain mapping, reclamation, precommit and broker
  counters; active broker failures above 25% fail the phase.
- A fixed non-streaming correctness matrix runs outside timed phases and keeps
  full output IDs in `correctness.json`; paired baseline/VMM runs can compare
  the same prompt and output hashes. Broker warmup now requires every configured
  tenant owner, not merely one owner, before measured phases begin.
- Request count, prompt length and output length are explicit CLI parameters.
  The runner derives dense TP1 KV bytes/token from each model config and records
  the logical phase-demand/broker-capacity ratio; chunk alignment and observed
  lease peaks remain authoritative. Per-instance arrival delays are applied
  after the common barrier and the measured dispatch skew is retained.
- Server teardown is attempted for every process in reverse launch order; any
  cleanup exception is appended to `violations` and changes the final status to
  `failed` before the result is rewritten.
- `SIGTERM`, `KeyboardInterrupt` and other non-ordinary termination paths now
  record a failed result before teardown, while final process return codes are
  retained in `server_exit_codes`.
- The final SQLite snapshot is taken only after both server process groups have
  exited. `sqlite_final` records WAL/sidecar provenance, while
  `configuration.broker.snapshot_final` queries owners, leases and outstanding
  bytes. A completed broker run fails if the database is not queryable or if
  any owner or lease remains after teardown. A direct `main(argv)` mock test
  exercises the launch-failure, reverse cleanup and post-teardown snapshot
  order without CUDA.
- The snapshot reader is also exercised against a real
  `GlobalAdmissionBroker` database created by the production broker, including
  fair targets and a recorded demand from the second tenant.
- Still open before a GPU acceptance claim: a live-server end-to-end runner
  test and the paired
  isolated/shared/observe/enforce GPU matrix itself.
