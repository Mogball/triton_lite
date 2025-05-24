from sys.intrinsics import block_idx


fn program_id[dim: UInt]() -> UInt:
    @parameter
    if dim == 0:
        return block_idx.x
    elif dim == 1:
        return block_idx.y
    else:
        return block_idx.z
