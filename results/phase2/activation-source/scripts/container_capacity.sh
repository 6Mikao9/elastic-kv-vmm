#!/usr/bin/env bash
set -euo pipefail
# Usage: container_capacity.sh GPU MODEL_HUB MODE OUTPUT
gpu=$1; hub=$2; mode=$3; output=$4; shift 4
root=$(cd "$(dirname "$0")/.." && pwd)
image=${SGLANG_IMAGE:-training-sandbox:torch280-cu128}
model_a=models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
model_b=models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
test -r "$hub/$model_a/config.json"
test -r "$hub/$model_b/config.json"
IFS=, read -r gpu_uuid gpu_memory gpu_util < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ ${gpu_memory// /} != 0 || ${gpu_util// /} != 0 ]] || grep -Fxq "${gpu_uuid// /}" <<< "$apps"; then
    echo "Selected GPU is busy; refusing to start." >&2; exit 1
fi
mkdir -p "$root/cache/capacity/home"
docker run --rm --pull=never --gpus "device=$gpu" --network none --shm-size 1g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$hub,dst=/hf/hub,readonly" \
    --mount "type=bind,src=$root,dst=/experiment" --workdir /experiment \
    -e "BENCH_GPU_UUID=${gpu_uuid// /}" -e HOME=/experiment/cache/capacity/home \
    -e HF_HOME=/experiment/cache/capacity/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e XDG_CACHE_HOME=/experiment/cache/capacity/xdg -e TORCH_HOME=/experiment/cache/capacity/torch \
    -e TRITON_CACHE_DIR=/experiment/cache/capacity/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/experiment/cache/capacity/torchinductor \
    -e FLASHINFER_WORKSPACE_BASE=/experiment/cache/capacity/flashinfer \
    -e SGLANG_JIT_CACHE_DIR=/experiment/cache/capacity/sglang-jit \
    -e CUDA_CACHE_PATH=/experiment/cache/capacity/cuda -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "$image" scripts/run_capacity.py \
    --model-a "/hf/hub/$model_a" --model-b "/hf/hub/$model_b" \
    --mode "$mode" --output-dir "/experiment/$output" "$@"
