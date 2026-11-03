#!/usr/bin/env bash
set -euo pipefail
# Usage: container_hisparse_smoke.sh GPU_LIST MODEL_DIR MODE OUTPUT [runner flags]
gpus=$1; model=$2; mode=$3; output=$4; shift 4
root=$(cd "$(dirname "$0")/.." && pwd)
image=lmsysorg/sglang:dev-minimax-m31-flash
expected_image=sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87
[[ $(docker image inspect -f '{{.Id}}' "$image") == "$expected_image" ]] || {
    echo "Image differs from the audited runtime." >&2
    exit 1
}
test -r "$model/config.json"
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]
IFS=, read -ra devices <<< "$gpus"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
uuids=()
for gpu in "${devices[@]}"; do
    IFS=, read -r uuid memory util < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if [[ ${memory// /} != 0 || ${util// /} != 0 ]] || grep -Fxq "${uuid// /}" <<< "$apps"; then
        echo "Selected GPU $gpu is busy; refusing to start." >&2
        exit 1
    fi
    uuids+=("${uuid// /}")
done
printf -v gpu_uuids '%s,' "${uuids[@]}"
gpu_uuids=${gpu_uuids%,}
mkdir -p "$root/cache"
cache="$root/cache/hisparse-shared"
cache_preexisted=0
if [[ -d "$cache" ]]; then cache_preexisted=1; fi
mkdir -p "$cache/home"
docker run --rm --pull=never --gpus "\"device=$gpus\"" --network none --shm-size 16g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$model,dst=/model,readonly" \
    --mount "type=bind,src=$root,dst=/experiment" \
    --mount "type=bind,src=$cache,dst=/cache" --workdir /experiment \
    -e HOME=/cache/home -e XDG_CACHE_HOME=/cache/xdg \
    -e HF_HOME=/cache/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e TORCH_HOME=/cache/torch -e TRITON_CACHE_DIR=/cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e SGLANG_JIT_CACHE_DIR=/cache/sglang -e CUDA_CACHE_PATH=/cache/cuda \
    -e PYTHONDONTWRITEBYTECODE=1 -e EKV_BENCH_GPU_UUIDS="$gpu_uuids" \
    -e EKV_COMPILER_CACHE_PREEXISTED="$cache_preexisted" \
    --entrypoint python3 "$image" -m scripts.run_hisparse_smoke \
    --model-path /model --output-dir "/experiment/$output" --mode "$mode" \
    --tp-size "${#devices[@]}" "$@"
