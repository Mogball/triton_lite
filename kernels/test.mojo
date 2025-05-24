import triton_lite as tl


fn test_kernel[
    BLOCK_X: UInt32
](out_ptr: tl.Ptr[UInt32], stride: UInt32, numel: UInt32,):
    alias blocked = tl.Blocked.one_d(1, 4)
    off_x = tl.program_id[0]() * BLOCK_X
    index = off_x + tl.arange[blocked, 0, BLOCK_X]()

    out_ptrs = out_ptr + index
    tl.store(out_ptrs, index)
