#!/usr/bin/env bash
set -euo pipefail
# Usage: run_padding_cases.sh GPU MODEL_HUB OUTPUT_PREFIX
gpu=$1; hub=$2; prefix=$3
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
for model in qwen06 qwen4; do
    if [[ $model == qwen06 ]]; then
        label=Qwen3-0.6B-Base
        relative=models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
    else
        label=Qwen3-4B
        relative=models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
    fi
    for arm in baseline v5 v5-keep8; do
        mode=$arm; retain=1
        if [[ $arm == v5-keep8 ]]; then mode=v5; retain=8; fi
        case_dir="$prefix/$model/graph-pad/$arm"
        if [[ ${EKV_RESUME:-0} == 1 && -f "$case_dir/results.correctness.json" ]]; then
            python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); assert all(json.loads((p/f).read_text())["status"]=="complete" for f in ("results.json","results.correctness.json"))' "$case_dir"
            echo "SKIP completed $model padded $arm"
            continue
        fi
        echo "START $model padded $arm"
        EKV_EXECUTION=graph-pad EKV_RETAIN_CHUNKS=$retain \
            bash scripts/container_case.sh "$gpu" "$hub" "$relative" "$label" \
            "$mode" "$case_dir" 3
        echo "DONE $model padded $arm"
        sleep 3
    done
done
