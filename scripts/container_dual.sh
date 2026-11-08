#!/usr/bin/env bash
set -euo pipefail
# Usage: container_dual.sh GPU MODEL_HUB MODE OUTPUT [run_dual.py flags]
gpu=$1; hub=$2; mode=$3; output=$4
shift 4
root=$(cd "$(dirname "$0")/.." && pwd)
image=${SGLANG_IMAGE:-training-sandbox:torch280-cu128}
image_id=$(docker image inspect --format '{{.Id}}' "$image")
image_digest=$(docker image inspect --format '{{join .RepoDigests ","}}' "$image")
model_a=models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
model_b=models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
test -r "$hub/$model_a/config.json"
test -r "$hub/$model_b/config.json"
IFS=, read -r gpu_uuid gpu_memory gpu_util < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ ${gpu_memory// /} != 0 || ${gpu_util// /} != 0 ]] || grep -Fxq "${gpu_uuid// /}" <<< "$apps"; then
    echo "Selected GPU is busy; refusing to start." >&2; exit 1
fi
mkdir -p "$root/cache/dual/home"
docker run --rm --pull=never --gpus "device=$gpu" --network none --shm-size 1g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$hub,dst=/hf/hub,readonly" \
    --mount "type=bind,src=$root,dst=/experiment" --workdir /experiment \
    -e "BENCH_GPU_UUID=${gpu_uuid// /}" -e HOME=/experiment/cache/dual/home \
    -e "SGLANG_IMAGE=$image" -e "EKV_IMAGE_ID=$image_id" \
    -e "EKV_IMAGE_DIGEST=$image_digest" \
    -e HF_HOME=/experiment/cache/dual/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e XDG_CACHE_HOME=/experiment/cache/dual/xdg -e TORCH_HOME=/experiment/cache/dual/torch \
    -e TRITON_CACHE_DIR=/experiment/cache/dual/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/experiment/cache/dual/torchinductor \
    -e FLASHINFER_WORKSPACE_BASE=/experiment/cache/dual/flashinfer \
    -e SGLANG_JIT_CACHE_DIR=/experiment/cache/dual/sglang-jit \
    -e CUDA_CACHE_PATH=/experiment/cache/dual/cuda -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "$image" scripts/run_dual.py \
    --model-a "/hf/hub/$model_a" --model-b "/hf/hub/$model_b" \
    --mode "$mode" --output-dir "/experiment/$output" "$@"
