import triton_lite as tl

from kernels.test import test_kernel
from kernels.make_4d_causal_mask import make_4d_causal_mask_kernel
from kernels.apply_rotary_emb import cos_rotate, sin_rotate

from gpu.host import DeviceContext
from memory import UnsafePointer
from gpu.host._compile import _compile_code, _to_sass


fn main() raises:
    # with DeviceContext() as ctx:
    #    dev_ptr = ctx.enqueue_create_buffer[DType.uint32](512)
    #    ctx.enqueue_function[test_kernel[128]](
    #        dev_ptr, 1, 512, grid_dim=4, block_dim=128
    #    )
    #    host_ptr = UnsafePointer[UInt32].alloc(512)
    #    ctx.enqueue_copy(host_ptr, dev_ptr)
    #    ctx.synchronize()

    #    for i in range(512):
    #        print(host_ptr[i])
    #    host_ptr.free()

    func = _compile_code[make_4d_causal_mask_kernel[128]]()
    print(func.asm)
    # print(_to_sass(func.asm))
