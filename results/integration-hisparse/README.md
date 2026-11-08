# Combined MiniMax integration evidence

These checked-in files are JSON summaries copied from a real four-A100
MiniMax-M3.1-Flash integration run. The server used TP4, eager execution and
native sparse HiSparse with VMM backing, request residency, and optional
future-aware ordering. Logs, source snapshots, model weights and machine-local
paths are intentionally excluded.

`vmm-future-off-r2`, `vmm-future-deadline-r2` and `vmm-future-off-r3` are
single-repeat controls over three workloads. They are order- and warm-state-
sensitive and do not support a stable throughput claim. `vmm-future-off-
adaptive-r1` and `vmm-future-deadline-adaptive-r1` exercise automatic joint
demotion on `mixed12k_unequal`. The first future deadline attempt is retained
on the experiment host, but excluded here because process-local telemetry
writers raced in that version.

Interpretation, limitations and result tables are in
[`docs/MINIMAX_FUTURE_INTEGRATION.md`](../../docs/MINIMAX_FUTURE_INTEGRATION.md).
