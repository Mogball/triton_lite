import triton_lite as tl
from gpu.host import DeviceContext
from math import ceildiv


fn div_floor_integer(
    a: UInt32,
    b: UInt32,
) -> UInt32:
    return (a - (a % b)) / b


fn cos_rotate[
    XBLOCK: UInt32
](
    in_ptr0: tl.Ptr[BFloat16],
    in_ptr1: tl.Ptr[UInt64],
    in_ptr2: tl.Ptr[BFloat16],
    in_ptr3: tl.Ptr[BFloat16],
    out_ptr0: tl.Ptr[BFloat16],
    ks0: UInt32,
    ks1: UInt32,
    ks2: UInt32,
    ks3: UInt32,
    ks4: UInt32,
    xnumel: UInt32,
):
    alias blocked = tl.Blocked.one_d(1, 4)
    xoffset = tl.program_id[0]() * XBLOCK
    xindex = xoffset + tl.arange[blocked, 0, XBLOCK]()
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x3 = xindex // ks0
    x2 = xindex // ks2
    x1 = (xindex // ks0) % ks4
    tmp0 = tl.load(
        in_ptr0
        + (
            UInt32(2) * ((x0 % (ks0 // 2)))
            + ks1 * x3
            + (((x0 // (ks0 // 2)) % 2))
        ),
        xmask,
        other=0,
    ).to[tl.float32]()
    tmp1 = tl.load(in_ptr1 + (x2), xmask, other=0).to[tl.uint32]()
    tmp2 = ks3
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    # tl.device_assert(
    #     ((0 <= tmp5) & (tmp5 < ks3)) | ~(xmask),
    #     "index out of bounds: 0 <= tmp5 < ks3",
    # )
    tmp7 = tl.load(in_ptr2 + (x0 + ks0 * tmp5), xmask, other=0).to[tl.float32]()
    tmp8 = tmp0 * tmp7
    tmp9 = x0
    tmp12 = ks0 + (-1) * (
        div_floor_integer((ks0 // 2) * (div_floor_integer(ks0, ks0 // 2)), 2)
    )
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(
        in_ptr0
        + (
            2
            * (
                (
                    (
                        (
                            div_floor_integer(
                                (ks0 // 2) * (div_floor_integer(ks0, ks0 // 2)),
                                2,
                            )
                        )
                        + (x0)
                    )
                    % (ks0 // 2)
                )
            )
            + ks1 * x3
            + (
                (
                    (
                        (
                            (
                                div_floor_integer(
                                    (ks0 // 2)
                                    * (div_floor_integer(ks0, ks0 // 2)),
                                    2,
                                )
                            )
                            + (x0)
                        )
                        // (ks0 // 2)
                    )
                    % 2
                )
            )
        ),
        xmask & tmp13,
        other=0.0,
    ).to[tl.float32]()
    tmp15 = -tmp14
    tmp16 = tl.full[blocked, tmp15.shape, tmp15.T.dtype](0.0)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tmp9 >= tmp12
    tmp21 = tl.load(
        in_ptr0
        + (
            2
            * (
                (
                    (
                        x0
                        + ((-1) * ks0)
                        + (
                            div_floor_integer(
                                (ks0 // 2) * (div_floor_integer(ks0, ks0 // 2)),
                                2,
                            )
                        )
                    )
                    % (ks0 // 2)
                )
            )
            + ks1 * x3
            + (
                (
                    (
                        (
                            x0
                            + ((-1) * ks0)
                            + (
                                div_floor_integer(
                                    (ks0 // 2)
                                    * (div_floor_integer(ks0, ks0 // 2)),
                                    2,
                                )
                            )
                        )
                        // (ks0 // 2)
                    )
                    % 2
                )
            )
        ),
        xmask & tmp18,
        other=0.0,
    ).to[tl.float32]()
    tmp22 = tl.where(tmp13, tmp17, tmp21)
    tmp23 = tl.load(
        in_ptr3 + (x0 + ks0 * tmp5),
        xmask,
        other=0,
    ).to[tl.float32]()
    tmp24 = tmp22 * tmp23
    tmp25 = tmp8 + tmp24
    tl.store(
        out_ptr0
        + (
            x0
            + x2 * (ks0 // 2) * (div_floor_integer(ks0, ks0 // 2))
            + ks3 * x1 * (ks0 // 2) * (div_floor_integer(ks0, ks0 // 2))
        ),
        tmp25.to[tl.bfloat16](),
        xmask,
    )


fn sin_rotate[
    XBLOCK: UInt32
](
    in_ptr0: tl.Ptr[BFloat16],
    in_ptr1: tl.Ptr[UInt64],
    in_ptr2: tl.Ptr[BFloat16],
    in_ptr3: tl.Ptr[BFloat16],
    out_ptr0: tl.Ptr[BFloat16],
    ks0: UInt32,
    ks1: UInt32,
    ks2: UInt32,
    xnumel: UInt32,
):
    alias blocked = tl.Blocked.one_d(1, 4)
    xoffset = tl.program_id[0]() * XBLOCK
    xindex = xoffset + tl.arange[blocked, 0, XBLOCK]()
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = xindex // ks0
    tmp0 = tl.load(
        in_ptr0
        + (2 * ((x0 % (ks0 // 2))) + ks1 * x1 + (((x0 // (ks0 // 2)) % 2))),
        xmask,
        other=0,
    ).to[tl.float32]()
    tmp1 = tl.load(in_ptr1 + (x1), xmask, other=0).to[tl.uint32]()
    tmp2 = ks2
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    # tl.device_assert(
    #     ((0 <= tmp5) & (tmp5 < ks2)) | ~(xmask),
    #     "index out of bounds: 0 <= tmp5 < ks2",
    # )
    tmp7 = tl.load(
        in_ptr2 + (x0 + ks0 * tmp5),
        xmask,
        other=0,
    ).to[tl.float32]()
    tmp8 = tmp0 * tmp7
    tmp9 = x0
    tmp12 = ks0 + (-1) * (
        div_floor_integer((ks0 // 2) * (div_floor_integer(ks0, ks0 // 2)), 2)
    )
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(
        in_ptr0
        + (
            2
            * (
                (
                    (
                        (
                            div_floor_integer(
                                (ks0 // 2) * (div_floor_integer(ks0, ks0 // 2)),
                                2,
                            )
                        )
                        + (x0)
                    )
                    % (ks0 // 2)
                )
            )
            + ks1 * x1
            + (
                (
                    (
                        (
                            (
                                div_floor_integer(
                                    (ks0 // 2)
                                    * (div_floor_integer(ks0, ks0 // 2)),
                                    2,
                                )
                            )
                            + (x0)
                        )
                        // (ks0 // 2)
                    )
                    % 2
                )
            )
        ),
        xmask & tmp13,
        other=0.0,
    ).to[tl.float32]()
    tmp15 = -tmp14
    tmp16 = tl.full[blocked, tmp15.shape, tmp15.T.dtype](0.0)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tmp9 >= tmp12
    tmp21 = tl.load(
        in_ptr0
        + (
            2
            * (
                (
                    (
                        x0
                        + ((-1) * ks0)
                        + (
                            div_floor_integer(
                                (ks0 // 2) * (div_floor_integer(ks0, ks0 // 2)),
                                2,
                            )
                        )
                    )
                    % (ks0 // 2)
                )
            )
            + ks1 * x1
            + (
                (
                    (
                        (
                            x0
                            + ((-1) * ks0)
                            + (
                                div_floor_integer(
                                    (ks0 // 2)
                                    * (div_floor_integer(ks0, ks0 // 2)),
                                    2,
                                )
                            )
                        )
                        // (ks0 // 2)
                    )
                    % 2
                )
            )
        ),
        xmask & tmp18,
        other=0.0,
    ).to[tl.float32]()
    tmp22 = tl.where(tmp13, tmp17, tmp21)
    tmp23 = tl.load(in_ptr3 + (x0 + ks0 * tmp5), xmask, other=0).to[
        tl.float32
    ]()
    tmp24 = tmp22 * tmp23
    tmp25 = tmp8 + tmp24
    tl.store(
        out_ptr0 + (x0 + x1 * (ks0 // 2) * (div_floor_integer(ks0, ks0 // 2))),
        tmp25.to[tl.bfloat16](),
        xmask,
    )


@export
fn apply_rotary_emb(
    s0: UInt32,
    s1: UInt32,
    cos_ptr: tl.Ptr[BFloat16],
    position_ids_ptr: tl.Ptr[UInt64],
    sin_ptr: tl.Ptr[BFloat16],
    s4: UInt32,
    s5: UInt32,
    q_ptr: tl.Ptr[BFloat16],
    s6: UInt32,
    k_ptr: tl.Ptr[BFloat16],
    out0_ptr: tl.Ptr[BFloat16],
    out1_ptr: tl.Ptr[BFloat16],
):
    try:
        with DeviceContext() as ctx:
            ps0 = s1 * s4
            cos_rotate_numel = s0 * s1 * s4
            cos_grid = ceildiv(cos_rotate_numel, 128)
            ctx.enqueue_function[cos_rotate[128]](
                q_ptr,
                position_ids_ptr,
                cos_ptr,
                sin_ptr,
                out0_ptr,
                s1,
                s5,
                ps0,
                s0,
                s4,
                cos_rotate_numel,
                grid_dim=cos_grid,
                block_dim=128,
            )
            sin_rorate_numel = s0 * s1
            sin_grid = ceildiv(sin_rorate_numel, 128)
            ctx.enqueue_function[sin_rotate[128]](
                k_ptr,
                position_ids_ptr,
                cos_ptr,
                sin_ptr,
                out1_ptr,
                s1,
                s6,
                s0,
                sin_rorate_numel,
                grid_dim=sin_grid,
                block_dim=128,
            )
            ctx.synchronize()
    except e:
        print(e)
