"""Launch matched vLLM baseline and elastic backing configurations."""
from importlib.metadata import version
import os
from pathlib import Path
import sys


def main():
    if version("vllm") != "0.29.0":
        raise SystemExit("This integration requires vLLM 0.29.0")
    mode = os.environ.get("EKV_VLLM_MODE", "baseline")
    if mode not in {"baseline", "vmm"}:
        raise SystemExit("EKV_VLLM_MODE must be baseline or vmm")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    sys.argv[1:1] = ["serve"]
    sys.argv.extend(["--enforce-eager", "--no-enable-prefix-caching",
                     "--no-async-scheduling", "--tensor-parallel-size", "1"])
    if mode == "vmm":
        stats = os.environ.get("EKV_VLLM_STATS_PATH")
        if not stats:
            raise SystemExit("Set EKV_VLLM_STATS_PATH for VMM")
        Path(stats).parent.mkdir(parents=True, exist_ok=True)
        sys.argv.extend(["--worker-cls", "vllm_integration.adapter.ElasticWorker"])
    from vllm.entrypoints.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
