import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _gated_norm_kernel(
    x_ptr,  # [B, L, D]
    w_ptr,  # [D]
    gate_ptr,  # [B or 1 or N, D], last dim contiguous
    gate_idx_ptr,  # [L] row of gate for each token, for packed varlen input (B=1)
    add_ptr,  # [B, L, D]
    o_ptr,  # [B, L, D]
    gate_stride,  # 0 to broadcast gate across batch
    L: tl.constexpr,
    D: tl.constexpr,
    gate_type: tl.constexpr,
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
        if gate_idx_ptr is not None:
            gate_row = tl.load(gate_idx_ptr + pid_l)
        else:
            gate_row = pid_b
        gate = tl.load(gate_ptr + (gate_row * gate_stride + offs), mask)

    x = x.to(tl.float32)
    var = tl.sum(x * x) * (1.0 / D)
    rrms = tl.rsqrt(var + eps)
    x *= rrms * w.to(tl.float32)

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
    eps: float = 1e-6,
    gate_idx: Tensor | None = None,
) -> Tensor:
    # x: [B, L, D]. gate: [B, 1, D] or [1, 1, D].
    # for packed varlen input, x is [1, L, D], gate is [N, 1, D] (per sample) and gate_idx is [L] (sample of each token)
    if torch.is_grad_enabled():
        # if True:
        if gate is not None and gate_idx is not None:
            gate = gate.flatten(0, 1)[gate_idx.long()].unsqueeze(0)
        x = F.rms_norm(x, x.shape[-1:], w, eps=eps)
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
    B, L, D = x.shape
    gate_stride = 0
    if gate is not None:
        gate = gate.flatten(0, -2)  # [B or 1 or N, D]
        assert gate.shape[-1] == D and gate.stride(-1) == 1, (gate.shape, gate.stride())
        if gate_idx is not None:
            assert B == 1 and gate_idx.shape == (L,), (x.shape, gate_idx.shape)
        else:
            assert gate.shape[0] in (1, B), (x.shape, gate.shape)
        gate_stride = 0 if gate.shape[0] == 1 else gate.stride(0)
    if add is not None:
        assert add.is_contiguous()
    out = torch.empty_like(x)
    _gated_norm_kernel[(L, B)](x, w, gate, gate_idx, add, out, gate_stride, L, D, gate_type, eps)
    return out
