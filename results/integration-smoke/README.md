# Combined SGLang integration smoke evidence

These JSON files are checked-in summaries from real SGLang serving runs. They
exclude server logs, model weights and machine-local source snapshots.

`future-052-compat-r1/` validates the SGLang 0.5.2 request-transport bridge and
future-aware queue ordering without the VMM adapter. `v5-future-052-eager-r3/`
and `v5-future-052-graph-r1/` combine that transport with V5 VMM backing in the
same serving process. The graph arm also records graph capture and replay
telemetry.

Each run required all completion checks to pass: three requests completed, the
hinted request overtook the slower request, scheduler telemetry advanced, and
the fallback and telemetry error counters stayed at zero. The V5 arms also
required physical backing to be below logical capacity. These are integration
smokes; they do not establish a throughput or latency improvement.

The implementation boundary is described in
[`docs/KV_HINTS_COMPAT.md`](../../docs/KV_HINTS_COMPAT.md) and
[`docs/INTEGRATION.md`](../../docs/INTEGRATION.md).
