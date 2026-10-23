import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _situ_kernel(
    x_ptr,  # [M, 2F] = [hidden, gate]
    o_ptr,  # [M, F]
    F: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = offs < F
    hidden = tl.load(x_ptr + pid_m * 2 * F + offs, mask).to(tl.float32)
    gate = tl.load(x_ptr + pid_m * 2 * F + F + offs, mask).to(tl.float32)

    out = (4.0 * _tanh(gate / 4.0) * tl.sigmoid(gate)) * (25.0 * _tanh(hidden / 25.0))
    tl.store(o_ptr + pid_m * F + offs, out, mask)


# soft-capped SwiGLU: (4 tanh(gate/4) sigmoid(gate)) * (25 tanh(hidden/25))
# x: [..., 2F] = [hidden, gate]
def situ(x: Tensor) -> Tensor:
    if torch.is_grad_enabled():
        hidden, gate = x.chunk(2, dim=-1)
        return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))

    assert x.is_contiguous()
    F = x.shape[-1] // 2
    M = x.numel() // (2 * F)
    out = x.new_empty(*x.shape[:-1], F)
    BLOCK_F = 1024
    _situ_kernel[(M, triton.cdiv(F, BLOCK_F))](x, out, F, BLOCK_F)
    return out
