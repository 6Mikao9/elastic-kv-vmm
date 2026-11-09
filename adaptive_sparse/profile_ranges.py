"""Opt-in Torch profiler ranges for MiniMax HiSparse diagnostics."""

from functools import wraps
import os


def profile_range(name):
    """Wrap a callable only when the worker enables diagnostic ranges."""
    if os.environ.get("EKV_PROFILE_RANGES") != "1":
        return _unchanged

    from torch.profiler import record_function

    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            with record_function(f"ekv/{name}"):
                return function(*args, **kwargs)

        return measured

    return decorate


def _unchanged(function):
    return function


def install_native_profile_ranges():
    """Mark native-only MiniMax calls in a diagnostic worker."""
    if os.environ.get("EKV_PROFILE_RANGES") != "1":
        return

    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    cls = HiSparseCoordinator
    if getattr(cls, "_ekv_native_profile_ranges", False):
        return
    map_latest = profile_range("native/map_latest")(cls.map_last_loc_to_buffer)
    swap = profile_range("native/swap")(cls.swap_in_selected_blocks)
    cls.map_last_loc_to_buffer = map_latest
    cls.swap_in_selected_blocks = swap
    cls._ekv_native_profile_ranges = True
