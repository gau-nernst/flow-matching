import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _gated_norm_kernel(
    x_ptr,  # [B, L, D]
    w_ptr,  # [D]
    gate_ptr,  # [B, D]
    add_ptr,  # [B, L, D]
    o_ptr,  # [B, L, D]
    L: tl.constexpr,
    D: tl.constexpr,
    gate_type: tl.constexpr,
    GEMMA_NORM: tl.constexpr,
    eps: float = 1e-5,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)

    BLOCK_DIM: tl.constexpr = triton.next_power_of_2(D)
    offs = tl.arange(0, BLOCK_DIM)
    mask = offs < D
    x = tl.load(x_ptr + (pid_b * L * D + pid_l * D + offs), mask, other=0.0)
    w = tl.load(w_ptr + offs, mask)
    if gate_ptr is not None:
        gate = tl.load(gate_ptr + (pid_b * D + offs), mask)

    x = x.to(tl.float32)
    var = tl.sum(x * x) * (1.0 / D)
    rrms = tl.rsqrt(var + eps)
    if GEMMA_NORM:
        scale = rrms + rrms * w.to(tl.float32)
    else:
        scale = rrms * w.to(tl.float32)
    x *= scale

    if gate_ptr is not None:
        if gate_type == "plus_one":
            x *= 1.0 + gate.to(tl.float32)
        elif gate_type == "tanh":
            x *= tl.extra.libdevice.tanh(gate.to(tl.float32))
        else:
            assert False

    if add_ptr is not None:
        add = tl.load(add_ptr + (pid_b * L * D + pid_l * D + offs), mask, other=0.0)
        x += add.to(tl.float32)

    tl.store(o_ptr + (pid_b * L * D + pid_l * D + offs), x, mask)


def gated_norm(
    x: Tensor,
    w: Tensor,
    gate: Tensor | None = None,
    add: Tensor | None = None,
    gate_type: str = "plus_one",
    gemma_norm: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    if torch.is_grad_enabled():
        # if True:
        if gemma_norm:
            x = F.rms_norm(x.float(), x.shape[:-1:], w.float() + 1.0, eps).to(x.dtype)
        else:
            x = F.rms_norm(x, x.shape[:-1:], w, eps)
        if gate is not None:
            if gate_type == "plus_one":
                x = x * (1.0 + gate)
            elif gate_type == "tanh":
                x = x * gate.tanh()
            else:
                raise ValueError(f"Unsupported {gate_type=}")
        if add is not None:
            x = x + add
        return x

    assert x.is_contiguous() and w.is_contiguous()
    if gate is not None:
        assert gate.is_contiguous()
    if add is not None:
        assert add.is_contiguous()
    B, L, D = x.shape
    out = torch.empty_like(x)
    _gated_norm_kernel[(L, B)](x, w, gate, add, out, L, D, gate_type, eps)
    return out
