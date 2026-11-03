# vLLM elastic physical backing prototype

Run the pinned vLLM 0.29.0 benchmark from the repository root. The wrapper uses the local image, checks that the selected GPU is idle, mounts the model hub read-only and disables container networking. Each invocation needs a fresh output directory relative to the repository.

First build `native/libelastic_vmm.so` with the [root build command](../README.md#reproduce).
The measured Python/package versions are in [environment.json](../results/vllm/environment.json).
The existing image was used; a fresh environment rebuild was not tested.

```sh
# Set GPU to an idle GPU index and HUB to the absolute model hub path.
MODEL_REL=models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
bash vllm_integration/container_case.sh "$GPU" "$HUB" "$MODEL_REL" Qwen3-0.6B-Base baseline results/repro-vllm/06-baseline 3
bash vllm_integration/container_case.sh "$GPU" "$HUB" "$MODEL_REL" Qwen3-0.6B-Base vmm results/repro-vllm/06-vmm 3
```

For Qwen3-4B, use the label Qwen3-4B and snapshot relative path
`models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`.
For a smoke run, pass `1 short_serial` as the trailing arguments.

Both arms use 2,048 GPU blocks of 16 tokens, an 8,192-token context limit, eight concurrent sequences, and a 2,048-token batch limit. The server uses the legacy runner in the vLLM V1 engine with TP1 and eager execution. Prefix caching and async scheduling are disabled. The tested scope excludes speculative decoding, sleep mode and KV connectors. The adapter requires one homogeneous full-attention cache group and matching manager and kernel block sizes.

The adapter preserves vLLM's logical blocks, block table and native KV tensor layout. It commits 2 MiB physical chunks before the worker accesses newly assigned, resumed or zeroed blocks. It retains touched chunks until the worker is idle, then synchronizes and reclaims them. The chunks covering block zero and the first physical chunk stay resident.

The runner writes streaming performance, fixed nonstreaming token-ID probes and source manifests. Compare peak and idle whole-device NVML readings against the matched vLLM baseline. Adapter backing counters are available for the VMM arm. See the [measured results](../docs/RESULTS_PHASE2.md#vllm-0290), [summary CSV](../results/vllm/summary/summary.csv), [summary JSON](../results/vllm/summary/summary.json), and [tested source snapshot](../results/vllm/tested-source). The archived source and manifests preserve the README revision present at measurement time. Subsequent README edits document reproduction and results.

Run both models and both arms with `bash vllm_integration/run_matrix.sh "$GPU" "$HUB" results/my-vllm`.
Recompute the published summary with:

```sh
python3 benchmarks/summarize.py \
  --case Qwen3-0.6B-Base:baseline:results/vllm/qwen06/baseline \
  --case Qwen3-0.6B-Base:vmm:results/vllm/qwen06/vmm \
  --case Qwen3-4B:baseline:results/vllm/qwen4/baseline \
  --case Qwen3-4B:vmm:results/vllm/qwen4/vmm \
  --output-dir results/vllm/summary
```
