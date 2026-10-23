import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _modulate_kernel(
    x_ptr,  # [B, L, D], strided
    shift_ptr,  # [B, D]
    scale_ptr,  # [B, D]
    res_ptr,  # [B, L, D], strided
    gate_ptr,  # [B, D]
    norm_ptr,  # [D]
    gated_ptr,  # [B, L, D], contiguous
    o_ptr,  # [B, L, D], contiguous
    stride_xb,
    stride_xl,
    stride_resb,
    stride_resl,
    shift_stride,
    scale_stride,
    gate_stride,
    L: tl.constexpr,
    D: tl.constexpr,
    RMS: tl.constexpr,
    eps: float = 1e-6,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)

    BLOCK_DIM: tl.constexpr = triton.next_power_of_2(D)
    offs = tl.arange(0, BLOCK_DIM)
    mask = offs < D
    x = tl.load(x_ptr + (pid_b * stride_xb + pid_l * stride_xl + offs), mask, other=0.0)

    if res_ptr is not None:
        res = tl.load(res_ptr + (pid_b * stride_resb + pid_l * stride_resl + offs), mask, other=0.0)
        gate = tl.load(gate_ptr + (pid_b * gate_stride + offs), mask)
        x = (x.to(tl.float32) + gate.to(tl.float32) * res.to(tl.float32)).to(x.dtype)
        tl.store(gated_ptr + (pid_b * L * D + pid_l * D + offs), x, mask)

    shift = tl.load(shift_ptr + (pid_b * shift_stride + offs), mask)
    scale = tl.load(scale_ptr + (pid_b * scale_stride + offs), mask)

    x = x.to(tl.float32)
    if not RMS:
        x -= tl.sum(x) * (1.0 / D)
    var = tl.sum(x * x) * (1.0 / D)
    x = x * tl.rsqrt(var + eps)
    if norm_ptr is not None:
        x = x * tl.load(norm_ptr + offs, mask).to(tl.float32)

    x = x * (1.0 + scale.to(tl.float32)) + shift.to(tl.float32)
    tl.store(o_ptr + (pid_b * L * D + pid_l * D + offs), x, mask)


def modulate(
    x: Tensor,
    shift: Tensor,
    scale: Tensor,
    res: Tensor | None = None,
    gate: Tensor | None = None,
    eps: float = 1e-6,
    norm_weight: Tensor | None = None,
    rms: bool = False,
) -> Tensor:
    # norm is LayerNorm (default) or RMSNorm, with optional weight
    if torch.is_grad_enabled():
        if res is not None:
            x = torch.addcmul(x, gate, res)
        if rms:
            out = F.rms_norm(x, x.shape[-1:], norm_weight, eps=eps)
        else:
            out = F.layer_norm(x, x.shape[-1:], norm_weight, eps=eps)
        out = torch.addcmul(out, out, scale) + shift
        return out, x

    B, L, D = x.shape
    assert x.stride(-1) == 1 and shift[0].is_contiguous() and scale[0].is_contiguous()
    assert shift.shape in ((B, 1, D), (1, 1, D))
    assert scale.shape in ((B, 1, D), (1, 1, D))
    if res is not None:
        assert gate is not None
        assert res.shape == x.shape and res.stride(-1) == 1 and gate[0].is_contiguous()
        assert gate.shape in ((B, 1, D), (1, 1, D))
    if norm_weight is not None:
        assert norm_weight.shape == (D,) and norm_weight.is_contiguous()
    gated = x.new_empty(B, L, D) if res is not None else x
    out = x.new_empty(B, L, D)
    _modulate_kernel[(L, B)](
        x,
        shift,
        scale,
        res,
        gate,
        norm_weight,
        gated,
        out,
        *x.stride()[:2],
        *(res.stride()[:2] if res is not None else (0, 0)),
        shift.stride(0) if shift.shape[0] > 1 else 0,
        scale.stride(0) if scale.shape[0] > 1 else 0,
        gate.stride(0) if gate is not None and gate.shape[0] > 1 else 0,
        L,
        D,
        rms,
        eps,
    )
    return out, gated
