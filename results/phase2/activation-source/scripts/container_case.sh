#!/usr/bin/env bash
set -euo pipefail
# Usage: container_case.sh GPU MODEL_HUB MODEL_REL LABEL MODE OUTPUT [REPEATS] [WORKLOAD ...]
gpu=$1; hub=$2; model_rel=$3; label=$4; mode=$5; output=$6
repeats=${7:-3}
if (( $# >= 7 )); then shift 7; else shift 6; fi
root=$(cd "$(dirname "$0")/.." && pwd)
image=${SGLANG_IMAGE:-training-sandbox:torch280-cu128}
test -r "$hub/$model_rel/config.json"
IFS=, read -r gpu_uuid gpu_memory gpu_util < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ ${gpu_memory// /} != 0 || ${gpu_util// /} != 0 ]] || grep -Fxq "${gpu_uuid// /}" <<< "$apps"; then
    echo "Selected GPU is busy; refusing to start." >&2; exit 1
fi
mkdir -p "$root/cache/e2e/home"
workload_args=()
if (( $# )); then workload_args=(--workloads "$@"); fi
docker run --rm --pull=never --gpus "device=$gpu" --network none --shm-size 1g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$hub,dst=/hf/hub,readonly" \
    --mount "type=bind,src=$root,dst=/experiment" --workdir /experiment \
    -e "BENCH_GPU_UUID=${gpu_uuid// /}" -e HOME=/experiment/cache/e2e/home \
    -e EKV_EXECUTION="${EKV_EXECUTION:-eager}" \
    -e EKV_RETAIN_CHUNKS="${EKV_RETAIN_CHUNKS:-1}" \
    -e EKV_ACTIVATION_POLICY="${EKV_ACTIVATION_POLICY:-off}" \
    -e EKV_PREFILL_HEADROOM_MIB="${EKV_PREFILL_HEADROOM_MIB:-0}" \
    -e EKV_PREFILL_CHUNK="${EKV_PREFILL_CHUNK:-2048}" \
    -e HF_HOME=/experiment/cache/e2e/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e XDG_CACHE_HOME=/experiment/cache/e2e/xdg -e TORCH_HOME=/experiment/cache/e2e/torch \
    -e TRITON_CACHE_DIR=/experiment/cache/e2e/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/experiment/cache/e2e/torchinductor \
    -e FLASHINFER_WORKSPACE_BASE=/experiment/cache/e2e/flashinfer \
    -e SGLANG_JIT_CACHE_DIR=/experiment/cache/e2e/sglang-jit \
    -e CUDA_CACHE_PATH=/experiment/cache/e2e/cuda -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "$image" scripts/run_case.py \
    --model-path "/hf/hub/$model_rel" --label "$label" --mode "$mode" \
    --output-dir "/experiment/$output" --repeats "$repeats" "${workload_args[@]}"
