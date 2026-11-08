"""Exercise predictive VMM prewarming on one real CUDA device.

This is a manual integration smoke, not a serving benchmark. It checks the
physical mapping lifecycle without loading a model or claiming CPU-only speed.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CHUNK = 2 << 20
SLOT = CHUNK
POOL_SIZE = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("predictive prewarm smoke requires a CUDA device")

    os.environ.update(
        {
            "EKV_MODE": "v5",
            "EKV_CHUNK_MIB": str(CHUNK >> 20),
            "EKV_STATS_PATH": "",
            "EKV_RETAIN_CHUNKS": "1",
            "EKV_PRECOMMIT_CHUNKS": "0",
            "EKV_PREDICTIVE_PREWARM_MODE": "enforce",
            "EKV_PREDICTIVE_PREWARM_TTL_MS": "100",
            "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS": "2",
        }
    )

    # Import after setting MODE: the adapter intentionally treats its mode as
    # process configuration rather than mutable request state.
    from elastic_kv import adapter

    torch.cuda.set_device(0)
    head_dim = SLOT // torch.empty((), dtype=torch.float64).element_size()
    pool = SimpleNamespace(
        size=POOL_SIZE,
        page_size=1,
        head_num=1,
        head_dim=head_dim,
        dtype=torch.float64,
        layer_num=1,
        device="cuda:0",
    )
    slots = pool.size + pool.page_size
    projection = adapter.PlaneProjection(slots * SLOT, 2, CHUNK, 0)
    views = []
    try:
        projection.ensure([0])
        ctrl = adapter.Residency(pool, [projection], SLOT, "layer_contiguous_shared_arena")
        ctrl.free_slots = adapter.FreeSlots(pool.size)
        free_before = ctrl.free_slots.count
        initial_backed = projection.backed_bytes

        demand = 4 * SLOT
        selected = ctrl.predictive_prewarm(demand)
        mapped_backed = projection.backed_bytes
        assert selected == 2, selected
        assert ctrl.free_slots.count == free_before
        assert ctrl.live_count == 0
        assert mapped_backed == initial_backed + 2 * CHUNK * projection.planes

        ranges = ctrl.free_slots.peek(2)
        ctrl.free_slots.claim(ranges)
        ctrl.admit_ranges(ranges)
        assert ctrl.live_count == 2
        assert ctrl.counters["predictive_prewarm_hit_chunks"] == 2
        assert ctrl.counters["predictive_prewarm_hit_bytes"] == 2 * CHUNK * projection.planes
        assert ctrl._predictive_precommitted == set()

        views = projection.views(torch.float64, (slots, 1, head_dim))
        views[0][1, 0, 0] = 17.0
        views[1][2, 0, 0] = 29.0
        torch.cuda.synchronize(0)
        assert views[0][1, 0, 0].item() == 17.0
        assert views[1][2, 0, 0].item() == 29.0

        ids = np.array([1, 2], dtype=np.int64)
        ctrl.release_idle_only(ids)
        ctrl.free_slots.release(ids)
        ctrl.reclaim_idle()
        assert ctrl.live_count == 0
        assert projection.backed_bytes == initial_backed

        second = ctrl.predictive_prewarm(demand)
        assert second == 2, second
        expired = ctrl.expire_predictive_prewarm(
            now_s=time.monotonic() + 0.2
        )
        assert expired == 2, expired
        assert ctrl.precommitted == set()
        assert projection.backed_bytes == initial_backed
        assert ctrl.counters["predictive_prewarm_candidate_bytes"] == 4 * CHUNK * projection.planes
        assert ctrl.counters["predictive_prewarm_mapped_bytes"] == 2 * CHUNK * projection.planes
        assert ctrl.counters["predictive_prewarm_hit_bytes"] == 2 * CHUNK * projection.planes
        assert ctrl.counters["predictive_prewarm_expired_bytes"] == 2 * CHUNK * projection.planes

        result = {
            "status": "passed",
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "selected_chunks": selected,
            "requested_demand_bytes": demand,
            "free_slots_before_after_prewarm": [free_before, free_before],
            "initial_backed_bytes": initial_backed,
            "prewarmed_backed_bytes": mapped_backed,
            "final_backed_bytes": projection.backed_bytes,
            "predictive_counters": {
                key: value
                for key, value in ctrl.counters.items()
                if key.startswith("predictive_prewarm_")
            },
            "native_metrics": projection.backing.metrics,
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        del views
        gc.collect()
        projection.backing.close()


if __name__ == "__main__":
    main()
