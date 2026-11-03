"""Start the stock server with the experimental SM80 projection adapter."""
import os
import sys

from adaptive_sparse.minimax_sm80 import install

install()

if os.environ.get("EKV_DISABLE_FUSED_QKNORM") == "1":
    from adaptive_sparse.minimax_diagnostic_fallbacks import install_minimax_diagnostic_fallbacks

    install_minimax_diagnostic_fallbacks()

if os.environ.get("EKV_LAYER_TRACE_DIR"):
    from adaptive_sparse.layer_trace import install_layer_trace

    install_layer_trace(os.environ["EKV_LAYER_TRACE_DIR"])

if os.environ.get("EKV_ATTN_CAPTURE_DIR"):
    from adaptive_sparse.minimax_attention_capture import install_minimax_attention_capture

    install_minimax_attention_capture(os.environ["EKV_ATTN_CAPTURE_DIR"])

if os.environ.get("EKV_VMM_STATS_DIR"):
    from adaptive_sparse.vmm_runtime import install_vmm_runtime

    install_vmm_runtime(os.environ["EKV_VMM_STATS_DIR"])

if int(os.environ.get("EKV_RESIDENT_LIMIT", "0")):
    from adaptive_sparse.request_residency import install_request_residency

    install_request_residency()

if os.environ.get("EKV_ADAPTIVE_STATS_DIR"):
    from adaptive_sparse.scheduler_budget import install_scheduler_budget

    install_scheduler_budget()

if os.environ.get("EKV_LOGICAL_SHADOW_DIR"):
    from adaptive_sparse.logical_shadow import install_logical_shadow

    install_logical_shadow(os.environ["EKV_LOGICAL_SHADOW_DIR"])

if os.environ.get("EKV_ALLOCATOR_STATS_DIR"):
    from adaptive_sparse.memory_telemetry import install_memory_telemetry

    install_memory_telemetry(os.environ["EKV_ALLOCATOR_STATS_DIR"])

if __name__ == "__main__":
    from sglang.launch_server import kill_process_tree, load_plugins, prepare_server_args, run_server

    load_plugins()
    args = prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
