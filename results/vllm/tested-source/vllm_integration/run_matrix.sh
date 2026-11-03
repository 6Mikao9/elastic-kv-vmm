#!/usr/bin/env bash
set -euo pipefail
# Usage: run_matrix.sh GPU MODEL_HUB OUTPUT_PREFIX
gpu=$1; hub=$2; prefix=$3
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
for model in qwen06 qwen4; do
    if [[ $model == qwen06 ]]; then
        label=Qwen3-0.6B-Base
        relative=models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
    else
        label=Qwen3-4B
        relative=models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
    fi
    for mode in baseline vmm; do
        echo "START vLLM $model $mode"
        bash vllm_integration/container_case.sh "$gpu" "$hub" "$relative" \
            "$label" "$mode" "$prefix/$model/$mode" 3
        echo "DONE vLLM $model $mode"
        sleep 3
    done
done
