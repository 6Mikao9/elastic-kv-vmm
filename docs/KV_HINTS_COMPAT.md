# SGLang 0.5.2/0.5.17 `kv_hints` compatibility

Neither SGLang 0.5.2 nor 0.5.17 carries the `kv_hints` transport used by this
project's future-aware scheduler hook. The compatibility layer crosses the HTTP,
tokenizer and scheduler process boundaries on both releases while keeping the
same baseline and VMM serving build.

`adaptive_sparse/kv_hints_compat.py` provides that transport only:

1. `GenerateReqInput` gains an appended `kv_hints` field before `http_server`
   imports the request class. Single, batch-broadcast, per-item and parallel
   sampling forms are normalized at this boundary.
2. The tokenizer wrapper validates and converts the envelope to a wire-safe
   mapping. On 0.5.17, it appends `kv_hints` to the msgspec
   `TokenizedGenerateReqInput` schema before the scheduler starts; SGLang uses
   that schema for msgpack IPC, so a dynamic Python attribute is insufficient.
   It does not parse or change prompt tokens, sampling parameters or output.
3. The scheduler wrapper attaches the envelope and scheduler receive time to
   `Req` before the native `SchedulePolicy.calc_priority()` call. The future hook
   still wraps that call, and native `PrefillAdder` remains the admission
   authority.

The compatibility installer is idempotent and process-local. It runs from the
same spawn-safe launcher path before SGLang imports the HTTP request class, and
repeat installation in a scheduler worker only sets shared attributes. Native
revisions that already expose `sglang.srt.managers.kv_hints` bypass the adapter.

The wire envelope is versioned and bounded:

```json
{
  "protocol_version": "1",
  "message_id": "example-message",
  "actions": [
    {
      "action_id": "request-42",
      "action_type": "elastic.future",
      "action_version": "2",
      "payload": {
        "deadline_after_ms": 1800,
        "deadline_clock": "perf_counter",
        "deadline_absolute_ms": 9123456800,
        "expected_output_tokens": 1024,
        "residence": "auto"
      }
    }
  ]
}
```

Unknown actions remain opaque so the transport can coexist with other versions.
Malformed required fields fail request normalization. The future hook validates
the action payload it consumes and falls back to native queue order when a
waiting queue lacks complete planning metadata.

Action version 2 carries an optional host-local `perf_counter` absolute deadline
alongside the legacy relative field. The rollout executor stamps it at HTTP
dispatch so scheduler receive time does not erase network and tokenization
latency. Version 1 remains accepted for relative-only hints. The executor and
server must share a host for the current clock-domain guarantee; cross-host use
requires a calibrated wall-clock protocol.

Action version 3 adds optional `prefill_tokens_per_s` and
`decode_tokens_per_s` fields. The rollout executor emits version 2 until the
closed-loop estimator has at least three valid completion samples, then emits
version 3 for later requests. Both rates must be finite and within the bounded
service-rate range. Older action versions remain accepted, but carrying either
rate in a version 1 or 2 action is rejected so old and new semantics cannot be
confused.

`residence` accepts `auto`, `resident` or `host_backed`. It is inert unless the
request-residency adapter is compiled into the run and
`EKV_RESIDENT_HINT_POLICY=adaptive` is set. `resident` requests a complete-GPU
KV slot subject to the configured slot count and per-rank physical reserve;
`host_backed` yields that slot; `auto` compares `expected_output_tokens` with
`EKV_RESIDENT_MIN_HINT_OUTPUT_TOKENS`. Missing or invalid planning metadata
uses the legacy long-request behavior, and no path bypasses SGLang's native
memory admission checks.

The focused tests cover immutable envelope decoding, malformed input, batch and
parallel expansion, pickling, HTTP-to-tokenizer-to-scheduler transport,
receive-time propagation, native `kv_hints` detection and the 0.5.17 msgspec
field extension. A real A100 run also passes the complete VMM-plus-future path
in both eager and CUDA Graph modes; the checked-in evidence is under
`results/integration-smoke/`.

Run the CPU checks from the project root:

```bash
python3 -m unittest discover -s tests -v
```
