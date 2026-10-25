import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _modulate_kernel(
    x_ptr,  # [B, L, D]
    shift_ptr,  # [B or N, D]
    scale_ptr,  # [B or N, D]
    res_ptr,  # [B, L, D]
    gate_ptr,  # [B or N, D]
    norm_ptr,  # [D]
    idx_ptr,  # [L] row of shift/scale/gate for each token, for packed varlen input (B=1)
    gated_ptr,  # [B, L, D]
    o_ptr,  # [B, L, D]
    x_stride_b,
    x_stride_l,
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
    x = tl.load(x_ptr + (pid_b * x_stride_b + pid_l * x_stride_l + offs), mask, other=0.0)

    if idx_ptr is not None:
        row = tl.load(idx_ptr + pid_l)
    else:
        row = pid_b

    if res_ptr is not None:
        res = tl.load(res_ptr + (pid_b * L * D + pid_l * D + offs), mask, other=0.0)
        gate = tl.load(gate_ptr + (row * gate_stride + offs), mask)
        x = (x.to(tl.float32) + gate.to(tl.float32) * res.to(tl.float32)).to(x.dtype)
        tl.store(gated_ptr + (pid_b * L * D + pid_l * D + offs), x, mask)

    shift = tl.load(shift_ptr + (row * shift_stride + offs), mask)
    scale = tl.load(scale_ptr + (row * scale_stride + offs), mask)

    x = x.to(tl.float32)
    if not RMS:
        x -= tl.sum(x) * (1.0 / D)
        x = tl.where(mask, x, 0.0)  # padding lanes must not contribute to the variance
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
    idx: Tensor | None = None,
) -> Tensor:
    # norm is LayerNorm (default) or RMSNorm, with optional weight
    # x: [B, L, D]. shift, scale, gate: [B, 1, D] or [1, 1, D].
    # for packed varlen input, x is [1, L, D], shift/scale/gate are [N, 1, D] (per sample)
    # and idx is [L] (sample of each token)
    if torch.is_grad_enabled():
        if idx is not None:
            shift, scale = [t[:, 0].index_select(0, idx).unsqueeze(0) for t in (shift, scale)]
            if gate is not None:
                gate = gate[:, 0].index_select(0, idx).unsqueeze(0)
        if res is not None:
            x = torch.addcmul(x, gate, res)
        if rms:
            out = F.rms_norm(x, x.shape[-1:], norm_weight, eps=eps)
        else:
            out = F.layer_norm(x, x.shape[-1:], norm_weight, eps=eps)
        out = torch.addcmul(out, out, scale) + shift
        return out, x

    B, L, D = x.shape
    x = x.contiguous()  # e.g. the image part of a dense [B, L, D] joint sequence

    def check(t: Tensor) -> int:  # returns row stride
        assert t.ndim == 3 and t.shape[1:] == (1, D) and t.stride(-1) == 1, (x.shape, t.shape, t.stride())
        if idx is None:
            assert t.shape[0] in (1, B), (x.shape, t.shape)
        return t.stride(0) if t.shape[0] > 1 else 0

    if idx is not None:
        assert B == 1 and idx.shape == (L,), (x.shape, idx.shape)
    shift_stride, scale_stride = check(shift), check(scale)
    gate_stride = 0
    if res is not None:
        assert gate is not None
        res = res.contiguous()
        gate_stride = check(gate)
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
        idx,
        gated,
        out,
        shift_stride,
        scale_stride,
        gate_stride,
        L,
        D,
        rms,
        eps,
    )
    return out, gated
