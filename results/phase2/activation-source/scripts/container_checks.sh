#!/usr/bin/env bash
set -euo pipefail
# Usage: container_checks.sh GPU OUTPUT_DIRECTORY
gpu=$1; output=$2
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
image=${SGLANG_IMAGE:-training-sandbox:torch280-cu128}
test ! -e "$output"
mkdir -p "$output" cache/e2e/home

run_check() {
    IFS=, read -r gpu_uuid gpu_memory gpu_util < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
    if [[ ${gpu_memory// /} != 0 || ${gpu_util// /} != 0 ]] || grep -Fxq "${gpu_uuid// /}" <<< "$apps"; then
        echo "Selected GPU is busy; refusing to start." >&2; return 1
    fi
    docker run --rm --pull=never --gpus "device=$gpu" --network none --shm-size 1g \
        --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=$root,dst=/experiment" --workdir /experiment \
        -e "BENCH_GPU_UUID=${gpu_uuid// /}" -e HOME=/experiment/cache/e2e/home \
        -e XDG_CACHE_HOME=/experiment/cache/e2e/xdg \
        -e CUDA_CACHE_PATH=/experiment/cache/e2e/cuda -e PYTHONDONTWRITEBYTECODE=1 \
        --entrypoint python3 "$image" "$@"
}

run_check -m unittest discover -s tests -v > "$output/unit-tests.txt" 2>&1
for policy in off both; do
    sleep 3
    run_check tests/activation_pressure.py --policy "$policy" \
        --outputpath "$output/pressure-$policy.json" > "$output/pressure-$policy.log" 2>&1
done
