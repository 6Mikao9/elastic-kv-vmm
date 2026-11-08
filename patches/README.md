# Native SGLang request-transport patches

`sglang-v0.5.2-kv-hints-transport.patch` applies to SGLang tag `v0.5.2` at
commit `b0d25e72c401f37b55d689ddbf05b8c583afe854`. It adds only the versioned
`kv_hints` request field, strict envelope validation, tokenizer-to-scheduler
transport, and scheduler request propagation. The VMM adapter and future
ordering policy remain in this repository.

From a clean SGLang checkout:

```bash
git apply --check /path/to/sglang-v0.5.2-kv-hints-transport.patch
git apply /path/to/sglang-v0.5.2-kv-hints-transport.patch
```

The patch fixes the transport protocol at version `1`, rejects malformed or
partially specified envelopes, treats only `null`/`None` as the absence of a
hint, and preserves one validated envelope per batch item through parallel
sampling. Unknown actions remain available for consumers that understand them.

Validation on 2026-11-09:

- 22 protocol and normalization tests passed.
- The six modified SGLang modules and the test file passed `py_compile`.
- `git diff --check` and reverse `git apply --check` passed.

The patch has not been submitted upstream. The runtime compatibility installer
in `adaptive_sparse/kv_hints_compat.py` remains the default supported path for
SGLang 0.5.2 and 0.5.17.
