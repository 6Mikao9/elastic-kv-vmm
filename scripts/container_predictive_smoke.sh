#!/usr/bin/env bash
set -euo pipefail
# Usage: container_predictive_smoke.sh GPU OUTPUT_DIRECTORY
if (( $# != 2 )); then
    echo "Usage: $0 GPU OUTPUT_DIRECTORY" >&2
    exit 2
fi
gpu=$1
output=$2
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
image=${PREDICTIVE_SMOKE_IMAGE:-lmsysorg/sglang:latest}
test ! -e "$output"

IFS=, read -r gpu_uuid gpu_memory gpu_util < <(
    nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ ${gpu_memory// /} != 0 || ${gpu_util// /} != 0 ]] || \
   grep -Fxq "${gpu_uuid// /}" <<< "$apps"; then
    echo "Selected GPU is busy; refusing to start." >&2
    exit 1
fi

mkdir -p "$output" cache/predictive-smoke/home
docker run --rm --pull=never --gpus "device=$gpu" --network none --shm-size 1g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$root,dst=/experiment" --workdir /experiment \
    -e "BENCH_GPU_UUID=${gpu_uuid// /}" \
    -e HOME=/experiment/cache/predictive-smoke/home \
    -e XDG_CACHE_HOME=/experiment/cache/predictive-smoke/xdg \
    -e CUDA_CACHE_PATH=/experiment/cache/predictive-smoke/cuda \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "$image" tests/smoke_predictive_prewarm.py \
    --output "/experiment/$output/result.json" \
    > "$output/smoke.log" 2>&1
