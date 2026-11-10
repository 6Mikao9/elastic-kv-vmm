#!/usr/bin/env bash
set -euo pipefail
# Usage: container_hisparse_locality.sh GPU_LIST MODEL_DIR DATASET_JSONL OUTPUT_ROOT [runner flags]
# OUTPUT_ROOT is private and git-ignored under .container-work/locality-study/runs.
# Use GPU_LIST=none with --prepare-only to use the real tokenizer without CUDA.
umask 077
gpus=$1; model=$(realpath -- "$2"); dataset=$(realpath -- "$3"); output=$4; shift 4
root=$(cd "$(dirname "$0")/.." && pwd)
image=lmsysorg/sglang:dev-minimax-m31-flash
expected_image=sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87
[[ $(docker image inspect -f '{{.Id}}' "$image") == "$expected_image" ]] || {
    echo "Image differs from the audited runtime." >&2
    exit 1
}
test -r "$model/config.json"
test -f "$dataset" && test -r "$dataset"
[[ "$output" =~ ^\.container-work/locality-study/runs(/[a-zA-Z0-9_-]+)*$ ]] || {
    echo "OUTPUT_ROOT must be under .container-work/locality-study/runs." >&2
    exit 1
}
gpu_args=()
gpu_uuids=
if [[ "$gpus" == none ]]; then
    [[ " $* " == *" --prepare-only "* ]] || {
        echo "GPU_LIST=none requires --prepare-only." >&2
        exit 1
    }
else
    [[ "$gpus" =~ ^[0-9]+(,[0-9]+){3}$ ]]
    IFS=, read -ra devices <<< "$gpus"
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
    uuids=()
    for gpu in "${devices[@]}"; do
        status=$(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
        IFS=, read -r uuid memory util <<< "$status"
        uuid=${uuid// /}
        if [[ -z "$uuid" || ${memory// /} != 0 || ${util// /} != 0 ]] || grep -Fxq "$uuid" <<< "$apps"; then
            echo "Selected GPU $gpu is unavailable or busy." >&2
            exit 1
        fi
        for previous_uuid in "${uuids[@]}"; do
            [[ "$previous_uuid" != "$uuid" ]] || { echo "Duplicate GPU $gpu." >&2; exit 1; }
        done
        uuids+=("$uuid")
    done
    printf -v gpu_uuids '%s,' "${uuids[@]}"
    gpu_uuids=${gpu_uuids%,}
    gpu_args=(--gpus "\"device=$gpu_uuids\"")
fi

private_root="$root/.container-work/locality-study/runs"
output_dir=$(realpath -m -- "$root/$output")
[[ "$output_dir" == "$private_root" || "$output_dir" == "$private_root/"* ]] || {
    echo "Output escaped the private runs directory." >&2
    exit 1
}
mkdir -p "$output_dir"
chmod 0700 "$root/.container-work/locality-study" "$private_root" "$output_dir"
free_kib() {
    df -Pk -- "$1" | awk 'NR == 2 {print $4}'
}
require_free() {
    local available
    available=$(free_kib "$1")
    [[ "$available" =~ ^[0-9]+$ && "$available" -ge "$2" ]] || {
        echo "Insufficient free disk space on the filesystem for $1." >&2
        exit 1
    }
}
require_free "$output_dir" 2097152
shared_cache="$root/cache/hisparse-shared"
cache_reused=0
if [[ -d "$shared_cache" ]]; then
    cache=$(realpath -- "$shared_cache")
    [[ "$cache" == "$shared_cache" && -O "$cache" && -w "$cache" ]] || {
        echo "Existing shared cache must be project-local, owned by the caller and writable." >&2
        exit 1
    }
    require_free "$cache" 2097152
    cache_reused=1
else
    require_free "$private_root" 6291456
    cache=$(mktemp -d "$private_root/.cache.XXXXXXXX")
fi
mkdir -p "$cache/home"
container_name="ekv-locality-$(date -u +%Y%m%dT%H%M%S)-$$"
docker run --rm --pull=never --name "$container_name" --label elastic-kv-vmm.locality=1 \
    "${gpu_args[@]}" --network none --shm-size 16g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$model,dst=/model,readonly" \
    --mount "type=bind,src=$dataset,dst=/dataset.jsonl,readonly" \
    --mount "type=bind,src=$root,dst=/experiment,readonly" \
    --mount "type=bind,src=$output_dir,dst=/output" \
    --mount "type=bind,src=$cache,dst=/cache" --workdir /experiment \
    -e HOME=/cache/home -e XDG_CACHE_HOME=/cache/xdg \
    -e HF_HOME=/cache/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e TORCH_HOME=/cache/torch -e TRITON_CACHE_DIR=/cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e SGLANG_JIT_CACHE_DIR=/cache/sglang -e CUDA_CACHE_PATH=/cache/cuda \
    -e PYTHONDONTWRITEBYTECODE=1 -e EKV_BENCH_GPU_UUIDS="$gpu_uuids" \
    -e EKV_LOCALITY_SHARED_CACHE_REUSED="$cache_reused" -e EKV_LOCALITY_CONTAINER_NAME="$container_name" \
    --entrypoint python3 "$image" -m scripts.run_hisparse_locality \
    "$@" --model-path /model --dataset /dataset.jsonl --output-root /output
