#!/usr/bin/env bash
set -euo pipefail
# Sequential GPU experiments. Usage: run_phase2.sh GPU MODEL_HUB OUTPUT_PREFIX
gpu=$1; hub=$2; prefix=$3
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
for model in qwen06 qwen4; do
    if [[ $model == qwen06 ]]; then
        label=Qwen3-0.6B-Base
        relative=models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
        executions=(graph eager)
    else
        label=Qwen3-4B
        relative=models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
        executions=(graph)
    fi
    for execution in "${executions[@]}"; do
        for mode in baseline v4 v5; do
            case_dir="$prefix/$model/$execution/$mode"
            if [[ ${EKV_RESUME:-0} == 1 && -f "$case_dir/results.correctness.json" ]]; then
                python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); assert all(json.loads((p/f).read_text())["status"]=="complete" for f in ("results.json","results.correctness.json"))' "$case_dir"
                echo "SKIP completed $model $execution $mode"
                continue
            fi
            echo "START $model $execution $mode"
            EKV_EXECUTION=$execution bash scripts/container_case.sh \
                "$gpu" "$hub" "$relative" "$label" "$mode" \
                "$prefix/$model/$execution/$mode" 3
            echo "DONE $model $execution $mode"
            sleep 3
        done
    done
done
