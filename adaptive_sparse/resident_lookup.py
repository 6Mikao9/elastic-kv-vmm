"""Resolve native MiniMax selected blocks for full-resident decode rows."""
import triton
import triton.language as tl


@triton.jit
def _resolve(topk, table, mapping, rows, lengths, selected, output,
             topk_row: tl.constexpr, topk_col: tl.constexpr,
             table_row: tl.constexpr, table_col: tl.constexpr,
             map_stride: tl.constexpr, row_stride: tl.constexpr,
             length_stride: tl.constexpr, selected_stride: tl.constexpr,
             out_row: tl.constexpr, out_col: tl.constexpr,
             width: tl.constexpr, block_size: tl.constexpr, tile: tl.constexpr):
    row = tl.load(selected + tl.program_id(0) * selected_stride)
    offset = tl.program_id(1) * tile + tl.arange(0, tile)
    active = offset < width
    block = tl.load(topk + row * topk_row + (offset // block_size) * topk_col,
                    mask=active, other=-1).to(tl.int64)
    position = block * block_size + offset % block_size
    length = tl.load(lengths + row * length_stride)
    req = tl.load(rows + row * row_stride).to(tl.int64)
    valid = active & (block >= 0) & (position < length)
    logical = tl.load(table + req * table_row + position * table_col,
                      mask=valid, other=0).to(tl.int64)
    physical = tl.load(mapping + logical * map_stride, mask=valid, other=-1)
    tl.store(output + row * out_row + offset * out_col, physical, mask=active)


def resolve_resident(topk, table, mapping, rows, lengths, selected, output, block_size):
    """Write only selected rows into an existing [1, B, K * block_size] tensor."""
    width = topk.shape[2] * block_size
    if not selected.numel() or not width:
        return
    tile = 256
    _resolve[(selected.numel(), triton.cdiv(width, tile))](
        topk, table, mapping, rows, lengths, selected, output,
        topk.stride(1), topk.stride(2), *table.stride(), mapping.stride(0),
        rows.stride(0), lengths.stride(0), selected.stride(0),
        output.stride(1), output.stride(2), width, block_size, tile,
    )
