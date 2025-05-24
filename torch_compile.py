import torch
import torch._inductor.codegen.triton as triton_codegen

import mojo_backend

old_init = triton_codegen.TritonKernel.__init__


def get_index_dtype_as_torch_dtype(self):
    import torch

    if self.index_dtype == "tl.int64":
        return torch.int64
    elif self.index_dtype == "tl.int32":
        return torch.int32
    else:
        raise ValueError(f"Unknown dtype: {self.index_dtype}")


def hijacked_init(self, *args, **kwargs):
    self.codegen_kernel = lambda *args, **kwargs: mojo_backend.TritonKernel.codegen_kernel(
        self, *args, **kwargs)
    self.codegen_reduction_numels = lambda *args, **kwargs: mojo_backend.TritonKernel.codegen_reduction_numels(
        self, *args, **kwargs)
    self.init_cooperative_reduction = lambda *args, **kwargs: mojo_backend.TritonKernel.init_cooperative_reduction(
        self, *args, **kwargs)
    self.codegen_range_tree = lambda *args, **kwargs: mojo_backend.TritonKernel.codegen_range_tree(
        self, *args, **kwargs)
    self.init_cooperative_reduction_mask = lambda *args, **kwargs: mojo_backend.TritonKernel.init_cooperative_reduction_mask(
        self, *args, **kwargs)
    self.iteration_ranges_codegen_header = lambda *args, **kwargs: mojo_backend.TritonKernel.iteration_ranges_codegen_header(
        self, *args, **kwargs)
    self.iteration_ranges_ranges_code = lambda *args, **kwargs: mojo_backend.TritonKernel.iteration_ranges_ranges_code(
        self, *args, **kwargs)
    self.codegen_reduction_indices = lambda *args, **kwargs: mojo_backend.TritonKernel.codegen_reduction_indices(
        self, *args, **kwargs)
    self.codegen_reduction_numels = lambda *args, **kwargs: mojo_backend.TritonKernel.codegen_reduction_numels(
        self, *args, **kwargs)
    self.iteration_ranges_scalar_code = lambda *args, **kwargs: mojo_backend.TritonKernel.iteration_ranges_scalar_code(
        self, *args, **kwargs)
    self.iteration_ranges_get_pid = lambda *args, **kwargs: mojo_backend.TritonKernel.iteration_ranges_get_pid(
        self, *args, **kwargs)
    self.load = lambda *args, **kwargs: mojo_backend.TritonKernel.load(
        self, *args, **kwargs)
    self.get_index_dtype_as_torch_dtype = lambda *args, **kwargs: get_index_dtype_as_torch_dtype(
        self, *args, **kwargs)
    self.kexpr = mojo_backend.texpr
    self.overrides = mojo_backend.TritonKernelOverrides
    old_init(self, *args, **kwargs)


triton_codegen.TritonKernel.__init__ = hijacked_init

# save the original entrypoint

#torch._dynamo.reset()
#torch._logging.set_logs(output_code=True)


@torch.compile
def test_hijacking(x, y, z):
    return x + y * z


def rmsnorm(weights, hidden_states, eps):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weights * hidden_states.to(input_dtype)


torch.manual_seed(42)
x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
z = torch.randn(1024, device="cuda")
print(test_hijacking(x, y, z))
print(x + y * z)

# rmsnorm_ref = rmsnorm
# rmsnorm_mojo = torch.compile(rmsnorm)
#
# x = torch.randn((4, 200, 7168), dtype=torch.bfloat16, device='cuda')
# w = torch.randn((7168, ), dtype=torch.bfloat16, device='cuda')
# out = rmsnorm_mojo(w, x, 1e-6)

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask, )

make_4d_causal_mask_mojo = torch.compile(_prepare_4d_causal_attention_mask)

ins = torch.ones((1, 6), dtype=torch.int64).cuda()
x = torch.empty(1, dtype=torch.bfloat16).cuda()
ref = _prepare_4d_causal_attention_mask(ins, (1, 6), x, 0)
out = make_4d_causal_mask_mojo(ins, (1, 6), x, 0)
torch.testing.assert_close(ref, out)
