from memory.unsafe_pointer import UnsafePointer

from gpu.id import block_id_in_cluster

from math import ceildiv


fn batch_linear_scaled[
    BLOCK_L: UInt, BLOCK_K: UInt, BLOCK_Q: UInt
](
    owned x_ptr: UnsafePointer[Scalar[DType.bfloat16]],
    owned w_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn]],
    owned w_scale_ptr: UnsafePointer[Scalar[DType.float32]],
    owned y_ptr: UnsafePointer[Scalar[DType.bfloat16]],
    B: UInt,
    L: UInt,
    K: UInt,
    Q: UInt,
    x_stride0: UInt,
    x_stride1: UInt,
    x_stride2: UInt,
    w_stride0: UInt,
    w_stride1: UInt,
    w_scale_stride0: UInt,
    w_scale_stride1: UInt,
    y_stride0: UInt,
    y_stride1: UInt,
    y_stride2: UInt,
):
    batch_id = block_id_in_cluster.x
    pid_l = block_id_in_cluster.y
    pid_q = block_id_in_cluster.z

    x_ptr = x_ptr + (batch_id * x_stride0) + (pid_l * x_stride1 * BLOCK_L)
    w_ptr = w_ptr + (pid_q * w_stride0 * BLOCK_Q)
    w_scale_ptr = w_scale_ptr + (pid_q * w_scale_stride0)
    y_ptr = y_ptr + (
        (batch_id * y_stride0)
        + (pid_l * y_stride1 * BLOCK_L)
        + (pid_q * y_stride2 * BLOCK_Q)
    )

    k_tiles = ceildiv(K, BLOCK_K)
