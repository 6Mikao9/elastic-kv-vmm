#!/usr/bin/env bash
set -euo pipefail
# GPU HUB OUTPUT POLICY TARGET_FREE_MIB HEADROOM_MIB BALLAST_LIMIT_MIB
if (( $# != 7 )); then
    echo "Usage: $0 GPU HUB OUTPUT POLICY TARGET_FREE_MIB HEADROOM_MIB BALLAST_LIMIT_MIB" >&2
    exit 2
fi
gpu=$1; hub=$2; output=$3; policy=$4; target=$5; headroom=$6; limit=$7
root=$(cd "$(dirname "$0")/.." && pwd)
image=${SGLANG_IMAGE:-training-sandbox:torch280-cu128}
test -r "$hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/config.json"
[[ "$output" != /* && "/$output/" != */../* ]]
IFS=, read -r uuid memory util < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ ${memory// /} != 0 || ${util// /} != 0 ]] || grep -Fxq "${uuid// /}" <<< "$apps"; then
    echo "Selected GPU is busy; refusing to start." >&2
    exit 1
fi
mkdir -p "$root/cache/e2e/home"
docker run --rm --pull=never --gpus "device=$gpu" --network none --shm-size 1g \
    --user "$(id -u):$(id -g)" --mount "type=bind,src=$hub,dst=/hf/hub,readonly" \
    --mount "type=bind,src=$root,dst=/experiment" --workdir /experiment \
    -e "BENCH_GPU_UUID=${uuid// /}" -e HOME=/experiment/cache/e2e/home \
    -e HF_HOME=/experiment/cache/e2e/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e XDG_CACHE_HOME=/experiment/cache/e2e/xdg -e TORCH_HOME=/experiment/cache/e2e/torch \
    -e TRITON_CACHE_DIR=/experiment/cache/e2e/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/experiment/cache/e2e/torchinductor \
    -e FLASHINFER_WORKSPACE_BASE=/experiment/cache/e2e/flashinfer \
    -e SGLANG_JIT_CACHE_DIR=/experiment/cache/e2e/sglang-jit \
    -e CUDA_CACHE_PATH=/experiment/cache/e2e/cuda -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "$image" scripts/run_prefill_pressure.py \
    --output "/experiment/$output" --policy "$policy" --target-free-mib "$target" \
    --headroom-mib "$headroom" --ballast-limit-mib "$limit"
