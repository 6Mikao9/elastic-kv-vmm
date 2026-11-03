"""Elastic physical backing for vLLM 0.29.0's legacy V1 GPU runner."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, create_kv_cache_views
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_worker import Worker

from elastic_kv.backing import Backing


class Residency:
    def __init__(self, config, device, layout, kernel_block_sizes):
        groups = config.kv_cache_groups
        tensors = config.kv_cache_tensors
        if len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ValueError("Elastic KV requires one dense full-attention cache group")
        spec = groups[0].kv_cache_spec
        if kernel_block_sizes and kernel_block_sizes != [spec.block_size]:
            raise ValueError("Elastic KV requires matching manager and kernel block sizes")
        sizes = {tensor.size for tensor in tensors}
        if len(sizes) != 1 or config.num_blocks < 2:
            raise ValueError("Elastic KV requires one shared backing allocation")
        if {name for tensor in tensors for name in tensor.layers} != set(groups[0].layer_names):
            raise ValueError("Elastic KV tensor layout does not match cache group")
        self.num_blocks = config.num_blocks
        self.page_bytes = spec.page_size_bytes
        self.segments = [(tensor.offset + i * tensor.layer_stride, tensor.block_stride)
                         for tensor in tensors for i in range(len(tensor.layers))]
        size = sizes.pop()
        if any(start < 0 or stride <= 0 or
               start + (self.num_blocks - 1) * stride + self.page_bytes > size
               for start, stride in self.segments):
            raise ValueError("Elastic KV tensor segments exceed the allocation")

        self.device = device
        self.backing = Backing(size, device=torch.device(device).index)
        self.sink_chunks = self.chunks([0]) | {0}
        self.backing.ensure(self.sink_chunks)
        raw = self.backing.raw.view(torch.int8)
        self.views = {}
        for tensor in tensors:
            views = create_kv_cache_views(raw, spec, self.num_blocks, layout, tensor,
                                          kernel_block_size=spec.block_size)
            self.views.update(zip(tensor.layers, views))
        self.blocks = {}
        self.peak_bytes = self.backing.backed_bytes
        self.ensure_calls = self.reclaim_calls = 0
        self.path = Path(os.environ["EKV_VLLM_STATS_PATH"])
        self.publish()

    def chunks(self, blocks):
        indices = set()
        chunk = self.backing.chunk_bytes
        for block in blocks:
            if not 0 <= block < self.num_blocks:
                raise ValueError(f"Invalid KV block {block}")
            for start, stride in self.segments:
                offset = start + block * stride
                indices.update(range(offset // chunk, (offset + self.page_bytes - 1) // chunk + 1))
        return indices

    def before_update(self, output):
        if output.kv_cache_block_copies:
            raise RuntimeError("Elastic KV does not support cache block copies")
        for req_id in output.finished_req_ids:
            self.blocks.pop(req_id, None)
        touched = set(output.new_block_ids_to_zero or ())
        for req in output.scheduled_new_reqs:
            self.blocks[req.req_id] = list(req.block_ids[0])
            touched.update(self.blocks[req.req_id])
        cached = output.scheduled_cached_reqs
        for req_id, ids in zip(cached.req_ids, cached.new_block_ids, strict=True):
            if ids is not None:
                if req_id in cached.resumed_req_ids:
                    self.blocks[req_id] = list(ids[0])
                else:
                    self.blocks[req_id].extend(ids[0])
                touched.update(ids[0])
        missing = self.chunks(touched) - self.backing.mapped_chunks
        if missing:
            self.backing.ensure(missing)
            self.ensure_calls += 1
            self.peak_bytes = max(self.peak_bytes, self.backing.backed_bytes)
            self.publish()

    def after_update(self):
        if self.blocks:
            return
        dead = self.backing.mapped_chunks - self.sink_chunks
        if dead:
            with torch.cuda.device(self.device):
                self.backing.reclaim(dead)
            self.reclaim_calls += 1
        self.publish()

    def publish(self):
        stats = {"mode": "vmm", "num_blocks": self.num_blocks,
                 "live_requests": len(self.blocks),
                 "logical_capacity_bytes": self.backing.capacity_bytes,
                 "backed_bytes": self.backing.backed_bytes,
                 "peak_backed_bytes": self.peak_bytes,
                 "ensure_calls": self.ensure_calls, "reclaim_calls": self.reclaim_calls,
                 **self.backing.metrics}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class ElasticModelRunner(gpu_model_runner.GPUModelRunner):
    residency = None

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes,
                                    kv_cache_allocation_context=None):
        def allocate(config, device, layout, block_sizes):
            self.residency = Residency(config, device, layout, block_sizes)
            return self.residency.views

        with patch.object(gpu_model_runner, "allocate_kv_cache", allocate):
            return super().initialize_kv_cache_tensors(
                kv_cache_config, kernel_block_sizes, kv_cache_allocation_context)

    def _update_states(self, scheduler_output):
        self.residency.before_update(scheduler_output)
        result = super()._update_states(scheduler_output)
        self.residency.after_update()
        return result

    def _dummy_run(self, *args, **kwargs):
        if self.residency is not None and (
            kwargs.get("force_attention") or
            getattr(kwargs.get("cudagraph_runtime_mode"), "name", "NONE") != "NONE"
        ):
            raise RuntimeError("Elastic KV requires eager dummy runs without attention")
        return super()._dummy_run(*args, **kwargs)


class ElasticWorker(Worker):
    def init_device(self):
        config = self.vllm_config
        parallel = config.parallel_config
        if (self.use_v2_model_runner or not config.model_config.enforce_eager
                or config.cache_config.enable_prefix_caching
                or config.scheduler_config.async_scheduling
                or config.speculative_config is not None
                or config.kv_transfer_config is not None
                or parallel.world_size != 1 or parallel.data_parallel_size != 1
                or config.model_config.enable_sleep_mode):
            raise ValueError("Unsupported vLLM configuration for elastic KV")
        with patch.object(gpu_model_runner, "GPUModelRunner", ElasticModelRunner):
            return super().init_device()
