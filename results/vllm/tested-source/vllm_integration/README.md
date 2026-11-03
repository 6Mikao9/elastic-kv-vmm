# vLLM elastic physical backing prototype

Use the pinned `vllm/vllm-openai:v0.29.0` image. Build the existing
`elastic_kv` native CUDA VMM library as described in the repository, then
run from the repository root in the container. Use the same model snapshot,
token limit, batch limits and block count in both arms.

```sh
EKV_VLLM_MODE=baseline python -m vllm_integration.launch /models/qwen   --served-model-name qwen --block-size 16
EKV_VLLM_MODE=vmm EKV_VLLM_STATS_PATH=/results/adapter_stats.json   python -m vllm_integration.launch /models/qwen   --served-model-name qwen --block-size 16
```

The server is limited to the legacy runner in the vLLM V1 engine, TP1 and
eager execution, with prefix caching, async scheduling, speculative decoding,
sleep mode and KV connectors disabled. The prototype supports one homogeneous
full-attention cache group and requires equal manager and kernel block sizes.

The adapter preserves vLLM's logical blocks, block table and native tensor
layout. It replaces physical backing only. Before request state update it
commits all new or resumed blocks and any blocks vLLM will zero. It conservatively
keeps all touched chunks until the worker has no requests, then synchronizes
before reclaiming them. Block zero and the first physical chunk stay resident.
Eager dummy runs use invalid slot mappings and omit attention. The adapter
rejects dummy runs that explicitly request attention after allocating the cache.

The VMM arm writes stats at startup and worker idle, including logical and
physical bytes, peak bytes, and mapping/reclaim counts. Measure device-wide
memory for both arms separately. For deterministic output checks, send
`return_token_ids: true` and `temperature: 0` to `/v1/completions`, then
compare `choices[0].token_ids` across matched requests.
