#!/usr/bin/env bash
# Source from an existing development container; no packages or models are changed.
# Usage: source scripts/container_session.sh
if [[ ! -f /.dockerenv && ! -f /run/.containerenv ]]; then
    echo 'Run this session inside the development container.' >&2
    return 1 2>/dev/null || exit 1
fi
ekv_project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export EKV_WORK_HOME="${EKV_WORK_HOME:-$ekv_project_root/.container-work}"
mkdir -p "$EKV_WORK_HOME"/{cache,profiles,results,tmp}
export XDG_CACHE_HOME="$EKV_WORK_HOME/cache/xdg"
export TORCH_HOME="$EKV_WORK_HOME/cache/torch"
export TRITON_CACHE_DIR="$EKV_WORK_HOME/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$EKV_WORK_HOME/cache/inductor"
export CUDA_CACHE_PATH="$EKV_WORK_HOME/cache/cuda"
export SGLANG_JIT_CACHE_DIR="$EKV_WORK_HOME/cache/sglang"
export FLASHINFER_WORKSPACE_BASE="$EKV_WORK_HOME/cache/flashinfer"
export HF_HOME="$EKV_WORK_HOME/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$EKV_WORK_HOME/tmp"
export PYTHONPATH="$ekv_project_root${PYTHONPATH:+:$PYTHONPATH}"
# Visibility does not grant a GPU reservation. Check device processes before use.
# Formal paired runs keep the cache policy in container_hisparse_smoke.sh.
cd "$ekv_project_root" || return 1
