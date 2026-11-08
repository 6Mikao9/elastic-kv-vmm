# SGLang 0.5.2 `kv_hints` compatibility

SGLang 0.5.2 carries ordinary request metadata across the HTTP, tokenizer and
scheduler processes, but it predates the native `kv_hints` field used by the
future-aware scheduler hook. The dense VMM adapter is measured on 0.5.2, so the
future policy has to cross those three process-local boundaries without
requiring a different serving build for its baseline and VMM arms.

`adaptive_sparse/kv_hints_compat.py` provides that transport only:

1. `GenerateReqInput` gains an appended `kv_hints` field before `http_server`
   imports the request class. Single, batch-broadcast, per-item and parallel
   sampling forms are normalized at this boundary.
2. The tokenizer wrapper copies the typed envelope onto the tokenized request.
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
      "action_version": "1",
      "payload": {
        "deadline_after_ms": 1800,
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
receive-time propagation and native `kv_hints` detection. A real A100 run also
passes the complete VMM-plus-future path in both eager and CUDA Graph modes;
the checked-in evidence is under `results/integration-smoke/`.

Run the CPU checks from the project root:

```bash
python3 -m unittest discover -s tests -v
```
