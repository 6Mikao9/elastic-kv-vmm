"""Matched gather kernels isolating software slot translation from payload reads.

Experimental only: input positions already represent the sparse support set.
This is neither a Top-K selector nor a model attention implementation.
"""

import triton
import triton.language as tl


@triton.jit
def _gather(raw, positions, request_ids, lengths, request_table, slot_offsets, out,
            width: tl.constexpr, selected: tl.constexpr, table_width: tl.constexpr,
            request_stride: tl.constexpr, sink_elements: tl.constexpr,
            indirect: tl.constexpr, tile: tl.constexpr):
    batch = tl.program_id(0)
    index = tl.program_id(1)
    req = tl.load(request_ids + batch).to(tl.int64)
    pos = tl.load(positions + batch * selected + index).to(tl.int64)
    length = tl.load(lengths + batch)
    valid = (pos >= 0) & (pos < length)
    if indirect:
        logical = tl.load(request_table + req * table_width + pos, mask=valid, other=0)
        offset = tl.load(slot_offsets + logical, mask=valid, other=0).to(tl.int64)
    else:
        offset = sink_elements + req * request_stride + pos * width
    column = tl.arange(0, tile)
    value = tl.load(raw + offset + column, mask=valid & (column < width), other=0)
    tl.store(out + (batch * selected + index) * width + column,
             value, mask=column < width)


def gather_rows(raw, positions, request_ids, lengths, request_table, slot_offsets,
                output, layout, *, indirect):
    """All tensors must be contiguous CUDA tensors on the same device.

    The caller prepares and validates positions and mapped prefixes before
    launching. Table values are ELEMENT offsets into raw, not physical addresses.
    Both modes access identical virtual/physical payload locations in the benchmark.
    """
    batch, selected = positions.shape
    if output.shape != (batch, selected, layout.row_elements):
        raise ValueError("output shape does not match positions and row width")
    if not batch or not selected:
        return
    _gather[(batch, selected)](
        raw, positions, request_ids, lengths, request_table, slot_offsets, output,
        layout.row_elements, selected, layout.max_tokens,
        layout.request_bytes // layout.itemsize, layout.chunk_bytes // layout.itemsize,
        indirect, triton.next_power_of_2(layout.row_elements),
    )
