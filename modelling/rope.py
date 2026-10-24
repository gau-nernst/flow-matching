import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn


def compute_rope(
    length: int,
    dim: int,
    theta: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.types.Device = None,
) -> Tensor:
    # initial computations in fp64
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim))
    timestep = torch.arange(length, device=device, dtype=torch.float64)
    freqs = (timestep[:, None] * omega).to(dtype)
    return torch.polar(torch.ones_like(freqs), freqs)


@triton.jit
def _rope(x, cos, sin, bwd: tl.constexpr = False, interleaved: tl.constexpr = True, ROT_D: tl.constexpr = 0):
    # interleaved: rotate pairs (2i, 2i + 1). otherwise (i, i + ROT_D/2) within each chunk of ROT_D, like HF
    BLOCK_H: tl.constexpr = x.shape[0]
    BLOCK_D: tl.constexpr = x.shape[1]
    if interleaved:
        x0, x1 = x.reshape(BLOCK_H, BLOCK_D // 2, 2).split()
    else:
        x = x.reshape(BLOCK_H, BLOCK_D // ROT_D, 2, ROT_D // 2).permute(0, 1, 3, 2)
        x0, x1 = x.reshape(BLOCK_H, BLOCK_D // 2, 2).split()
    if not bwd:
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
    else:
        y0 = x0 * cos + x1 * sin
        y1 = -x0 * sin + x1 * cos
    y = tl.join(y0, y1)
    if not interleaved:
        y = y.reshape(BLOCK_H, BLOCK_D // ROT_D, ROT_D // 2, 2).permute(0, 1, 3, 2)
    return y.reshape(BLOCK_H, BLOCK_D)


@triton.jit
def _norm_rope_kernel(
    x_ptr,  # [B, L, H, D]
    rope_ptr,  # [L, D]
    norm_ptr,  # [D]
    y_ptr,
    stride_xb,
    stride_xl,
    stride_ob,
    stride_ol,
    H,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    interleaved: tl.constexpr,
    ROT_D: tl.constexpr,
    IS_GEMMA: tl.constexpr,
    eps=1e-6,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)

    BLOCK_D: tl.constexpr = triton.next_power_of_2(D)
    offs_h = tl.arange(0, BLOCK_H)[:, None]
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x_ptrs = x_ptr + (pid_b * stride_xb + pid_l * stride_xl + offs_h * D + offs_d)
    y_ptrs = y_ptr + (pid_b * stride_ob + pid_l * stride_ol + offs_h * D + offs_d)

    rope = tl.load(rope_ptr + (pid_l * D + offs_d), mask_d)
    cos, sin = rope.reshape(BLOCK_D // 2, 2).split()
    if norm_ptr is not None:
        norm = tl.load(norm_ptr + offs_d, mask_d).to(tl.float32)
        if IS_GEMMA:
            norm += 1.0

    for h in range(tl.cdiv(H, BLOCK_H)):
        mask_h = offs_h < H - h * BLOCK_H
        mask = mask_h & mask_d

        x = tl.load(x_ptrs, mask, other=0.0).to(tl.float32)

        if norm_ptr is not None:
            rrms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) * (1.0 / D) + eps)
            x = x * rrms * norm
            # x = x.to(tl.bfloat16).to(tl.float32)

        tl.store(y_ptrs, _rope(x, cos, sin, interleaved=interleaved, ROT_D=ROT_D), mask)

        x_ptrs += BLOCK_H * D
        y_ptrs += BLOCK_H * D


def _norm_rope_ref(
    x: Tensor,
    rope: Tensor,
    norm: Tensor | None = None,
    eps: float = 1e-6,
    interleaved: bool = True,
    rot_dim: int | None = None,
    gemma_norm: bool = False,
) -> Tensor:
    if norm is not None and gemma_norm:
        x = F.rms_norm(x.float(), x.shape[-1:], eps=eps) * (1.0 + norm.float())
    elif norm is not None:
        x = F.rms_norm(x, x.shape[-1:], norm, eps)
    cos, sin = torch.view_as_real(rope).unsqueeze(-3).unbind(-1)  # [L, 1, D/2] each
    R = rot_dim or x.shape[-1]
    if not interleaved:  # reorder to interleaved, then back
        x = x.unflatten(-1, (-1, 2, R // 2)).transpose(-1, -2).flatten(-3)
    x0, x1 = x.float().unflatten(-1, (-1, 2)).unbind(-1)  # [B, L, nH, D/2] each
    out = torch.stack(
        [
            torch.addcmul(x0 * cos, x1, sin, value=-1),
            torch.addcmul(x0 * sin, x1, cos),
        ],
        dim=-1,
    ).flatten(-2)
    if not interleaved:
        out = out.unflatten(-1, (-1, R // 2, 2)).transpose(-1, -2).flatten(-3)
    return out


def apply_rope(
    x: Tensor,
    rope: Tensor,
    norm: Tensor | None = None,
    eps: float = 1e-6,
    *,
    interleaved: bool = True,
    rot_dim: int | None = None,
    gemma_norm: bool = False,
    out: Tensor | None = None,
) -> Tensor:
    # x: [B, L, nH, D] in real
    # rope: [L, D/2] in complex
    # rot_dim: partial rotary for half-split layout. rope must have angle 0 beyond rot_dim/2
    # gemma_norm: zero-centered norm weight i.e. norm(x) * (1 + w)
    if any([x.requires_grad, rope.requires_grad, norm.requires_grad if norm is not None else False]):
        out_ = _norm_rope_ref(x, rope, norm, eps, interleaved, rot_dim, gemma_norm)
        return out.copy_(out_) if out is not None else out_.type_as(x)

    assert x[0, 0].is_contiguous() and rope.is_contiguous()
    if norm is not None:
        assert norm.is_contiguous()
    rope_real = torch.view_as_real(rope)
    if out is not None:
        assert out[0, 0].is_contiguous()
    else:
        out = torch.empty_like(x)
    B, L, H, D = x.shape
    ROT_D = rot_dim or D
    assert interleaved or (D & (D - 1) == 0 and D % ROT_D == 0)  # half split requires power-of-2 D
    BLOCK_H = 8  # 8x128 = 1024. so 4 warps issue 16B
    grid = (L, B)
    strides = (*x.stride()[:2], *out.stride()[:2])
    _norm_rope_kernel[grid](x, rope_real, norm, out, *strides, H, D, BLOCK_H, interleaved, ROT_D, gemma_norm, eps=eps)
    return out


class RopeND(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        max_lens: tuple[int, ...],
        theta: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert len(dims) == len(max_lens)
        self.dims = dims
        self.max_lens = max_lens
        self.theta = theta
        self.dtype = dtype
        self.precompute_rope()

    def precompute_rope(self, device: torch.types.Device = None) -> None:
        # don't create things on meta device to avoid weird cases...
        device = device or torch.get_default_device()
        if torch.device(device) == torch.device("meta"):
            device = "cpu"

        for i, (dim, length) in enumerate(zip(self.dims, self.max_lens)):
            # always compute on CPU, then move to the requested device
            rope = compute_rope(length, dim, self.theta, dtype=self.dtype, device="cpu")
            self.register_buffer(f"rope{i}", rope.to(device), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)

        # recompute rope if dtype is changed
        dtype = self.dtype.to_complex()
        if any(getattr(self, f"rope{i}").dtype != dtype for i in range(len(self.dims))):
            self.precompute_rope(self.rope0.device)

        return self

    def create(self, start_list: tuple[int, ...], length_list: tuple[int, ...]) -> Tensor:
        pos_list = [
            torch.arange(start, start + length, device=self.rope0.device)
            for start, length in zip(start_list, length_list)
        ]
        grids = torch.meshgrid(pos_list, indexing="ij")  # this returns list[Tensor]

        rope_list = [getattr(self, f"rope{i}")[grid.flatten()] for i, grid in enumerate(grids)]
        return torch.cat(rope_list, dim=-1)
