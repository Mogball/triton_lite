import triton_lite as tl
from gpu.host import DeviceContext
from math import ceildiv


fn make_4d_causal_mask_kernel[
    XBLOCK: UInt32
](
    in_ptr: tl.Ptr[UInt64],
    out_ptr: tl.Ptr[BFloat16],
    seq_len: UInt32,
    numel: UInt32,
):
    alias blocked = tl.Blocked.one_d(1, 4)

    off_x = tl.program_id[0]() * XBLOCK
    offs_x = off_x + tl.arange[blocked, 0, XBLOCK]()
    mask_x = offs_x < numel

    offs_in = offs_x % seq_len
    offs_out = offs_x // seq_len

    x = tl.load(in_ptr + offs_in, mask=mask_x, other=0).to[tl.float32]()
    x = Float32(1.0) - x
    attn_mask_value = Float32(-3.3895313892515355e38)
    x = tl.where(x != 0.0, attn_mask_value, x)

    final_mask = tl.where(
        offs_in < (offs_out + 1),
        tl.Tile[offs_x.shape, Float32, blocked](0.0),
        attn_mask_value,
    )
    out = tl.where(x != 0.0, attn_mask_value, final_mask).to[tl.bfloat16]()
    tl.store(out_ptr + offs_x, out, mask=mask_x)


@export
fn make_4d_causal_mask_invoke(
    in_ptr: tl.Ptr[UInt64],
    out_ptr: tl.Ptr[Float32],
    stride: UInt32,
    numel: UInt32,
):
    try:
        with DeviceContext() as ctx:
            grid = ceildiv(numel, 128)
            ctx.enqueue_function[make_4d_causal_mask_kernel[128]](
                in_ptr, out_ptr, stride, numel, grid_dim=grid, block_dim=128
            )
            ctx.synchronize()
    except e:
        print(e)
