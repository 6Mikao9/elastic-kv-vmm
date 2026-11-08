#!/usr/bin/env bash
set -euo pipefail
# Usage: container_future_smoke.sh GPU MODEL_PATH OUTPUT_DIRECTORY [RUNNER_FLAGS ...]
if (( $# < 3 )); then
    echo "Usage: $0 GPU MODEL_PATH OUTPUT_DIRECTORY [RUNNER_FLAGS ...]" >&2
    exit 2
fi
gpu=$1; model=$2; output=$3
shift 3
root=$(cd "$(dirname "$0")/.." && pwd)
image=${FUTURE_SMOKE_IMAGE:-lmsysorg/sglang:dev-minimax-m31-flash}
test -r "$model/config.json"
IFS=, read -r gpu_uuid gpu_memory gpu_util < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ ${gpu_memory// /} != 0 || ${gpu_util// /} != 0 ]] || grep -Fxq "${gpu_uuid// /}" <<< "$apps"; then
    echo "Selected GPU is busy; refusing to start." >&2
    exit 1
fi
mkdir -p "$root/cache/e2e/home"
docker run --rm --pull=never --gpus "device=$gpu" --network none --shm-size 2g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$model,dst=/model,readonly" \
    --mount "type=bind,src=$root,dst=/experiment" --workdir /experiment \
    -e HOME=/experiment/cache/e2e/home \
    -e HF_HOME=/experiment/cache/e2e/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e XDG_CACHE_HOME=/experiment/cache/e2e/xdg -e TORCH_HOME=/experiment/cache/e2e/torch \
    -e TRITON_CACHE_DIR=/experiment/cache/e2e/triton \
    -e FLASHINFER_WORKSPACE_BASE=/experiment/cache/e2e/flashinfer \
    -e SGLANG_JIT_CACHE_DIR=/experiment/cache/e2e/sglang-jit \
    -e CUDA_CACHE_PATH=/experiment/cache/e2e/cuda -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "$image" scripts/run_future_smoke.py \
    --model-path /model --output-dir "/experiment/$output" "$@"
