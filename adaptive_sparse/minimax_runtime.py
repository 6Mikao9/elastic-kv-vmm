"""Install opt-in adapters for the audited MiniMax HiSparse server path.

The function is intentionally import-safe: every SGLang-facing module is loaded
only when its corresponding environment feature is enabled.  The legacy SM80
launcher and ``elastic-sglang`` share this sequence so both entry points apply
the same compatibility, residency and scheduler hooks before HTTP startup.
"""

from __future__ import annotations

import os


def install_minimax_runtime() -> tuple[str, ...]:
    """Install enabled MiniMax adapters in the audited order.

    Return component names for launch diagnostics.  Configuration remains
    environment-driven because these adapters predate the dense ``ElasticConfig``
    and must stay compatible with the pinned MiniMax development image.
    """

    installed: list[str] = []

    if os.environ.get("EKV_MINIMAX_SM80_BF16") == "1":
        from adaptive_sparse.minimax_sm80 import install as install_sm80

        install_sm80()
        installed.append("sm80_bf16_projections")

    if os.environ.get("EKV_DISABLE_FUSED_QKNORM") == "1":
        from adaptive_sparse.minimax_diagnostic_fallbacks import (
            install_minimax_diagnostic_fallbacks,
        )

        install_minimax_diagnostic_fallbacks()
        installed.append("diagnostic_fallbacks")

    if trace_dir := os.environ.get("EKV_LAYER_TRACE_DIR"):
        from adaptive_sparse.layer_trace import install_layer_trace

        install_layer_trace(trace_dir)
        installed.append("layer_trace")

    if capture_dir := os.environ.get("EKV_ATTN_CAPTURE_DIR"):
        from adaptive_sparse.minimax_attention_capture import (
            install_minimax_attention_capture,
        )

        install_minimax_attention_capture(capture_dir)
        installed.append("attention_capture")

    if vmm_stats_dir := os.environ.get("EKV_VMM_STATS_DIR"):
        from adaptive_sparse.vmm_runtime import install_vmm_runtime

        install_vmm_runtime(vmm_stats_dir)
        installed.append("vmm_runtime")

    if os.environ.get("EKV_FUTURE_POLICY", "off") != "off":
        from adaptive_sparse.sglang_future_hook import (
            install as install_future_policy,
        )

        install_future_policy()
        installed.append("future_policy")

    if int(os.environ.get("EKV_RESIDENT_LIMIT", "0")):
        from adaptive_sparse.request_residency import install_request_residency

        install_request_residency()
        installed.append("request_residency")

    if any(
        name.startswith("EKV_ADAPTIVE_") or name == "EKV_FUTURE_PREFILL_MODE"
        for name in os.environ
    ):
        from adaptive_sparse.scheduler_budget import install_scheduler_budget

        install_scheduler_budget()
        installed.append("scheduler_budget")

    if shadow_dir := os.environ.get("EKV_LOGICAL_SHADOW_DIR"):
        from adaptive_sparse.logical_shadow import install_logical_shadow

        install_logical_shadow(shadow_dir)
        installed.append("logical_shadow")

    if allocator_stats_dir := os.environ.get("EKV_ALLOCATOR_STATS_DIR"):
        from adaptive_sparse.memory_telemetry import install_memory_telemetry

        install_memory_telemetry(allocator_stats_dir)
        installed.append("memory_telemetry")

    if (os.environ.get("EKV_PROFILE_RANGES") == "1"
            and not int(os.environ.get("EKV_RESIDENT_LIMIT", "0"))):
        from adaptive_sparse.profile_ranges import install_native_profile_ranges

        install_native_profile_ranges()
        installed.append("native_profile_ranges")

    return tuple(installed)
