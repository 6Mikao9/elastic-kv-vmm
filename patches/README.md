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

## Cache-request abort lifecycle patch

`sglang-main-cache-request-abort-lifecycle.patch` applies to SGLang `main` at
commit `9f6678aa3d6b8fd4e898fe0fc0b11b932126cf61`. It separates cache-request
abort into three ordered phases:

1. cancel lookup/prefetch work;
2. optionally release KV and let a final STORE retain its source lock;
3. close the backend request session once after pending stores complete.

This fixes two distinct failure modes. FlexKV must cancel before KV
finalization or an asynchronous STORE can lose its source-node lock. LMCache
must keep its session open across an abort-time STORE and suppress duplicate
`finish_request` calls. The patch also makes requests that never opened an
external session close without emitting one.

From a checkout at the recorded base commit:

```bash
git apply --check /path/to/sglang-main-cache-request-abort-lifecycle.patch
git apply /path/to/sglang-main-cache-request-abort-lifecycle.patch
```

The local patch commit is `ac50029ac5c46d8ab05c6515ad3faf0bccdfa80b`;
its author is `Anonymous Researcher <anonymous@example.invalid>`. It has not
been submitted to the official SGLang remote. The changed files passed
`py_compile` and `git diff --check` on Windows. The new CPU regression test
could not collect there because SGLang imports the Unix-only `resource`
module during package initialization, so Linux CI remains required before
publication.
